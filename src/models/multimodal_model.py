"""End-to-end multimodal dementia detection model (paper Fig. 1, Sec IV).

Data flow (base config; N = total chunks in batch, B = recordings):

    Audio chunks [N, T_chunk]
        -> SpeechEncoder (HuBERT, sum last 2 layers)   -> frames [N, L, 768]
        -> Pooling (ASP)                               -> [N, 1536]
        -> average chunks per recording                -> [B, 1536]
        -> Linear projection                           -> f_a [B, 768]

    Transcript [B, S]
        -> TextEncoder (BERT [CLS])                     -> f_t [B, 768]

    (f_a, f_t) -> MINE statistics net                   (MI maximization objective)
    (f_a, f_t) -> AT-Fusion                             -> h [B, 768]
        h -> Classifier                                 -> logits [B, 2]

The MINE network lives inside the model so its parameters are optimized; the DV
loss is computed in src/losses/mine_loss.py using ``self.mine``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..configs import ModelConfig
from ..modules import MINE, Classifier, build_fusion, build_pooling
from .speech_encoder import SpeechEncoder
from .text_encoder import TextEncoder


@dataclass
class ModelOutput:
    logits: torch.Tensor  # [B, num_classes]
    f_a: torch.Tensor  # [B, speech_embed_dim]
    f_t: torch.Tensor  # [B, text_embed_dim]


class MultimodalDementiaModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.speech_encoder = SpeechEncoder(
            model_name=cfg.speech_model_name,
            num_layers=cfg.num_speech_layers,
            freeze=cfg.freeze_speech,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )
        self.text_encoder = TextEncoder(
            model_name=cfg.text_model_name, freeze=cfg.freeze_text,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )

        self.pooling = build_pooling(
            cfg.pooling, input_dim=self.speech_encoder.output_dim,
            asp_bottleneck=cfg.asp_bottleneck,
        )
        self.speech_projection = nn.Linear(self.pooling.output_dim, cfg.speech_embed_dim)
        f_a_dim = cfg.speech_embed_dim
        f_t_dim = self.text_encoder.output_dim

        self.mine = MINE(dim_x=f_a_dim, dim_z=f_t_dim, hidden=cfg.mine_hidden)

        bilinear_params = {
            "mm_dim": cfg.bilinear_mm_dim,
            "mfb_mm_dim": cfg.bilinear_mfb_mm_dim,
            "rank": cfg.bilinear_rank,
            "factor": cfg.bilinear_factor,
            "chunks": cfg.bilinear_chunks,
            "dropout": cfg.bilinear_dropout,
        }
        self.fusion = build_fusion(
            cfg.fusion, dim_a=f_a_dim, dim_t=f_t_dim,
            output_dim=cfg.fused_dim, bilinear=bilinear_params,
        )
        self.classifier = Classifier(
            input_dim=self.fusion.output_dim, hidden=cfg.classifier_hidden,
            num_classes=cfg.num_classes, dropout=cfg.dropout,
        )

    def _average_chunks(self, chunk_embeddings: torch.Tensor,
                        chunk_to_sample: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Average per-chunk embeddings back to one vector per recording.

        chunk_embeddings: [N, P]; chunk_to_sample: [N]; returns [B, P].
        """
        P = chunk_embeddings.shape[1]
        device = chunk_embeddings.device
        sums = torch.zeros(batch_size, P, device=device, dtype=chunk_embeddings.dtype)
        sums.index_add_(0, chunk_to_sample, chunk_embeddings)  # [B, P]
        counts = torch.zeros(batch_size, device=device, dtype=chunk_embeddings.dtype)
        counts.index_add_(
            0, chunk_to_sample, torch.ones_like(chunk_to_sample, dtype=chunk_embeddings.dtype)
        )
        return sums / counts.clamp(min=1.0).unsqueeze(1)  # [B, P]

    def encode_speech(self, batch: dict) -> torch.Tensor:
        frames, frame_mask = self.speech_encoder(
            batch["input_values"], batch["audio_attention_mask"]
        )  # [N, L, D], [N, L]
        pooled = self.pooling(frames, frame_mask)  # [N, pool_out]
        per_recording = self._average_chunks(
            pooled, batch["chunk_to_sample"], batch["batch_size"]
        )  # [B, pool_out]
        return self.speech_projection(per_recording)  # [B, speech_embed_dim]

    def encode_text(self, batch: dict) -> torch.Tensor:
        return self.text_encoder(
            batch["input_ids"], batch["text_attention_mask"], batch.get("token_type_ids")
        )  # [B, text_embed_dim]

    def forward(self, batch: dict) -> ModelOutput:
        f_a = self.encode_speech(batch)  # [B, Da]
        f_t = self.encode_text(batch)  # [B, Dt]
        h = self.fusion(f_a, f_t)  # [B, fused_dim]
        logits = self.classifier(h)  # [B, num_classes]
        return ModelOutput(logits=logits, f_a=f_a, f_t=f_t)
