"""Speech encoder: pretrained SSL model + last-N-layer summation (paper Sec IV-b).

Why: extract contextualized frame-level acoustic representations from each 10s
chunk. The paper uses HuBERT and sums the last two hidden layers; the ablations
swap the SSL model (HuBERT / wav2vec2.0 / XLS-R) and the number of summed final
layers (1 / 2 / 3).

    H_a = H(-1) + H(-2)        # element-wise sum of last two hidden layers

Tensors:
    in:  input_values [N, T_chunk], attention_mask [N, T_chunk]
    out: frames [N, L, D], frame_mask [N, L]   (N = total chunks in the batch)

Fine-tuned end-to-end (user decision: freeze_speech=False).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


class SpeechEncoder(nn.Module):
    def __init__(self, model_name: str = "facebook/hubert-base-ls960",
                 num_layers: int = 2, freeze: bool = False) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.num_layers = num_layers
        self.hidden_size = self.backbone.config.hidden_size

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    @property
    def output_dim(self) -> int:
        return self.hidden_size

    def _frame_mask(self, attention_mask: torch.Tensor, feat_len: int) -> torch.Tensor:
        """Convert waveform-level mask [N, T] to frame-level mask [N, L]."""
        if attention_mask is None:
            return None
        get_len = getattr(self.backbone, "_get_feat_extract_output_lengths", None)
        if get_len is None:
            return None
        valid_samples = attention_mask.sum(dim=1)  # [N]
        out_lengths = get_len(valid_samples).to(torch.long)  # [N]
        # Guard: guarantee >=1 valid frame (a very short trailing chunk could
        # otherwise map to 0 frames and make ASP's softmax NaN), and <= L.
        out_lengths = out_lengths.clamp(min=1, max=feat_len)
        idx = torch.arange(feat_len, device=attention_mask.device).unsqueeze(0)  # [1, L]
        return (idx < out_lengths.unsqueeze(1)).long()  # [N, L]

    def forward(self, input_values: torch.Tensor,
                attention_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # input_values: [N, T_chunk]
        outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # (embeddings, layer_1, ..., layer_K)

        # Sum the last ``num_layers`` hidden layers (paper: last two).
        selected = hidden_states[-self.num_layers :]  # each [N, L, D]
        frames = torch.stack(selected, dim=0).sum(dim=0)  # [N, L, D]

        L = frames.shape[1]
        frame_mask = self._frame_mask(attention_mask, L)  # [N, L] or None
        if frame_mask is None:
            frame_mask = torch.ones(frames.shape[0], L, dtype=torch.long, device=frames.device)
        return frames, frame_mask
