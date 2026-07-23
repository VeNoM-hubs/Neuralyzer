"""Abstract dataset defining the ONLY contract the model depends on.

Per CURSOR.md "DEVELOPMENT DATASET POLICY", every dataset (mock / ADReSS /
PROCESS-2) MUST expose the same ``Sample`` interface:

    Sample = {
        "audio": waveform,      # 1-D float32 torch.Tensor at DataConfig.sample_rate
        "transcript": string,   # raw transcript text (no tokenization here)
        "label": integer,       # 0 = control/healthy, 1 = AD/impaired
    }

Switching datasets must require replacing ONLY a subclass of this base -- never
the model, trainer, or losses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

import torch
from torch.utils.data import Dataset


class Sample(TypedDict):
    audio: torch.Tensor  # shape [T], float32, mono, at target sample_rate
    transcript: str
    label: int


class BaseDementiaDataset(Dataset, ABC):
    """Base class enforcing the ``Sample`` contract."""

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate

    @abstractmethod
    def __len__(self) -> int:  # pragma: no cover - interface
        ...

    @abstractmethod
    def _load(self, index: int) -> Sample:
        """Return a fully-materialized ``Sample`` for ``index``.

        Subclasses load audio (resampled to ``self.sample_rate``, mono float32),
        the transcript string, and the integer label.
        """
        ...

    def __getitem__(self, index: int) -> Sample:
        sample = self._load(index)
        audio = sample["audio"]
        if not torch.is_tensor(audio):
            audio = torch.as_tensor(audio, dtype=torch.float32)
        audio = audio.to(torch.float32)
        if audio.dim() > 1:
            audio = audio.mean(dim=0)  # collapse channels -> mono [T]
        return {
            "audio": audio,
            "transcript": str(sample["transcript"]),
            "label": int(sample["label"]),
        }

    def labels(self) -> list:
        """Return all labels (used for stratified train/val splitting)."""
        return [self._load(i)["label"] for i in range(len(self))]
