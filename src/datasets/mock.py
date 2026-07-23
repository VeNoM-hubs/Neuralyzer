"""Temporary synthetic dataset for development ONLY.

Purpose (CURSOR.md "DEVELOPMENT GOAL"): validate the full pipeline before real
ADReSS data is available. It fabricates waveforms + transcripts + labels purely
to exercise shapes and the training/eval loop. It is NOT a scientific substitute
for ADReSS.

KNOWN DIFFERENCES vs ADReSS (per CURSOR.md "WHEN USING A TEMPORARY DATASET"),
all touching ONLY the loader (architecture/preprocessing/training/eval unchanged):
  * synthetic audio (noise/tones), not real speech        -> meaningless metrics only
  * random-word transcripts, not Cookie-Theft descriptions
  * synthetic label balance
  * no predefined split (we apply the stratified 65/35 split)
"""

from __future__ import annotations

import random

import torch

from .base import BaseDementiaDataset, Sample

_VOCAB = (
    "the cookie boy girl mother kitchen water sink stool falling reaching "
    "curtain window plate dish dry overflow taking looking stealing laughing "
    "and is are a an on in over from with running spilling".split()
)


class MockDementiaDataset(BaseDementiaDataset):
    def __init__(
        self,
        num_samples: int = 40,
        sample_rate: int = 16000,
        min_seconds: float = 4.0,
        max_seconds: float = 24.0,
        seed: int = 0,
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        self.num_samples = num_samples
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds
        rng = random.Random(seed)
        self._specs = []
        for _ in range(num_samples):
            duration = rng.uniform(min_seconds, max_seconds)
            label = rng.randint(0, 1)
            n_words = rng.randint(8, 40)
            words = [rng.choice(_VOCAB) for _ in range(n_words)]
            self._specs.append((duration, label, " ".join(words)))

    def __len__(self) -> int:
        return self.num_samples

    def _load(self, index: int) -> Sample:
        duration, label, transcript = self._specs[index]
        n = int(duration * self.sample_rate)
        # Deterministic per-index waveform: low-amplitude noise + a tone whose
        # frequency correlates weakly with the label (gives a faint, learnable
        # signal so smoke-test loss can move).
        g = torch.Generator().manual_seed(index)
        noise = 0.01 * torch.randn(n, generator=g)
        t = torch.arange(n, dtype=torch.float32) / self.sample_rate
        freq = 220.0 + 110.0 * label
        tone = 0.05 * torch.sin(2 * torch.pi * freq * t)
        audio = (noise + tone).to(torch.float32)
        return {"audio": audio, "transcript": transcript, "label": label}
