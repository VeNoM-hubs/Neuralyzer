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

    if name == "process":
        from .process import ProcessDataset

        if cfg.data_root is None:
            raise ValueError("data.data_root must be set for the process dataset.")
        tasks = cfg.process_tasks or ["CTD"]
        return ProcessDataset(
            cfg.data_root, sample_rate=cfg.sample_rate, tasks=tasks,
            labels_csv=cfg.labels_csv, label_map=cfg.label_map, split=split,
        )

    if name == "kaggle_pitt":
        from .kaggle_pitt import KagglePittDataset

        if cfg.data_root is None:
            raise ValueError("data.data_root must be set for the kaggle_pitt dataset.")
        return KagglePittDataset(
            cfg.data_root, sample_rate=cfg.sample_rate,
            transcripts_file=cfg.transcripts_file, label_map=cfg.label_map, split=split,
        )

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

    raise ValueError(
        f"Unknown dataset '{cfg.dataset}'. Expected mock|process|kaggle_pitt|adress|process2."
    )
