"""Text encoder: pretrained BERT + [CLS] embedding (paper Sec IV-a).

    H = BERT(w_1, ..., w_N)
    f_t = H[CLS]              # hidden state of the [CLS] token

Tensors:
    in:  input_ids [B, S], attention_mask [B, S], token_type_ids [B, S]
    out: f_t [B, D]          (D = BERT hidden size, 768 for bert-base)

Fine-tuned end-to-end (user decision: freeze_text=False).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", freeze: bool = False,
                 gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.backbone.config.hidden_size

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        elif gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    @property
    def output_dim(self) -> int:
        return self.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor = None) -> torch.Tensor:
        # input_ids: [B, S]
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        # [CLS] is position 0 of the last hidden state (paper uses H[CLS], the
        # token hidden state, not the pooled/tanh output).
        return outputs.last_hidden_state[:, 0]  # [B, D]
