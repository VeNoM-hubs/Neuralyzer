"""End-to-end smoke test on the mock dataset (CURSOR.md "DEVELOPMENT GOAL").

Verifies the full pipeline runs, tensor shapes are correct at every stage, and a
training step reduces the loss on a tiny overfit batch. Run on Colab or locally
(if training deps are installed).

    python scripts/smoke_test.py

Note: first run downloads the pretrained HuBERT and BERT checkpoints.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoFeatureExtractor, AutoTokenizer  # noqa: E402

from src.configs import Config  # noqa: E402
from src.datasets import MockDementiaDataset, MultimodalCollator  # noqa: E402
from src.losses import combined_loss  # noqa: E402
from src.models import MultimodalDementiaModel  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main() -> None:
    set_seed(0)
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device={device}")

    dataset = MockDementiaDataset(num_samples=6, sample_rate=cfg.data.sample_rate, seed=0)
    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model.speech_model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_model_name)
    collator = MultimodalCollator(
        feature_extractor=feature_extractor, tokenizer=tokenizer,
        sample_rate=cfg.data.sample_rate, chunk_seconds=cfg.data.chunk_seconds,
        max_text_length=cfg.data.max_text_length,
    )
    batch = collator([dataset[i] for i in range(4)])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    B = batch["batch_size"]
    print(f"[smoke] batch_size={B}  num_chunks={batch['input_values'].shape[0]}")

    model = MultimodalDementiaModel(cfg.model).to(device)
    out = model(batch)

    D = cfg.model.speech_embed_dim
    assert out.f_a.shape == (B, D), f"f_a shape {out.f_a.shape} != {(B, D)}"
    assert out.f_t.shape == (B, cfg.model.text_embed_dim), f"f_t shape {out.f_t.shape}"
    assert out.logits.shape == (B, cfg.model.num_classes), f"logits shape {out.logits.shape}"
    print(f"[smoke] shapes OK: f_a={tuple(out.f_a.shape)} "
          f"f_t={tuple(out.f_t.shape)} logits={tuple(out.logits.shape)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        out = model(batch)
        loss, ce, mi = combined_loss(out.logits, batch["labels"], model.mine,
                                     out.f_a, out.f_t, mine_lambda=cfg.train.mine_lambda)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        optimizer.step()
        losses.append(float(loss.item()))
        print(f"[smoke] step {step}: loss={loss.item():.4f} ce={ce.item():.4f} mi={mi.item():.4f}")

    assert losses[-1] < losses[0] + 1e-3, "loss did not decrease over steps"
    print("[smoke] PASSED: pipeline runs end-to-end and loss decreases.")


if __name__ == "__main__":
    main()
