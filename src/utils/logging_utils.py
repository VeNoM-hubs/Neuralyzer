"""Lightweight logging utilities.

Logging framework was unspecified by the paper; user chose CSV + console
(no TensorBoard / W&B dependency).
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List, Optional

import torch


def get_console_logger(name: str = "neuralyzer") -> logging.Logger:
    """Return a configured console logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def resolve_device(device: str) -> torch.device:
    """Resolve the "auto" device string to a concrete torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class CSVLogger:
    """Append-only CSV metric logger."""

    def __init__(self, path: str, fieldnames: Optional[List[str]] = None) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.fieldnames = fieldnames
        self._initialized = False

    def log(self, row: Dict[str, object]) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        mode = "a" if self._initialized else "w"
        with open(self.path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})
