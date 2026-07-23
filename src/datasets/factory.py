"""Dataset factory: the single place mapping a config to a Dataset class.

Switching from mock to ADReSS/PROCESS-2 changes ONLY this selection plus the
corresponding placeholder loader -- model/trainer/losses untouched.
"""

from __future__ import annotations

from ..configs import DataConfig
from .base import BaseDementiaDataset


def build_dataset(cfg: DataConfig, split: str = "train") -> BaseDementiaDataset:
    name = cfg.dataset.lower()

    if name == "mock":
        from .mock import MockDementiaDataset

        seed = 0 if split == "train" else 1
        return MockDementiaDataset(sample_rate=cfg.sample_rate, seed=seed)

    if name == "adress":
        from .adress import ADReSSDataset

        if cfg.data_root is None:
            raise ValueError("data.data_root must be set for the ADReSS dataset.")
        return ADReSSDataset(cfg.data_root, split=split, sample_rate=cfg.sample_rate)

    if name == "process2":
        from .process2 import Process2Dataset

        if cfg.data_root is None:
            raise ValueError("data.data_root must be set for the PROCESS-2 dataset.")
        return Process2Dataset(cfg.data_root, split=split, sample_rate=cfg.sample_rate)

    raise ValueError(f"Unknown dataset '{cfg.dataset}'. Expected mock|adress|process2.")
