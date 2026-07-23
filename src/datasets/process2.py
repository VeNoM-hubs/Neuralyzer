"""PROCESS-2 dataset loader (PLACEHOLDER - awaiting data access).

Hosted on HuggingFace (CognoSpeak/PROCESS-2, likely gated). ~21h audio, 400
participants (200 HC, 150 MCI, 50 AD).

Paper task (Sec III): binary classification with (MCI + AD) grouped as
"cognitively impaired" (label 1) vs healthy controls HC (label 0), Cookie Theft
Description task.

Deferred until the dataset schema (audio/transcript/label columns, official
split) is confirmed. Only this Dataset class needs completing.
"""

from __future__ import annotations

from .base import BaseDementiaDataset, Sample


class Process2Dataset(BaseDementiaDataset):
    def __init__(self, data_root: str, split: str = "train", sample_rate: int = 16000):
        super().__init__(sample_rate=sample_rate)
        raise NotImplementedError(
            "Process2Dataset is a placeholder. Confirm the HuggingFace dataset "
            "schema (audio/transcript/label columns, official split) before "
            "implementing. Group MCI+AD as label 1 and HC as label 0 per the paper."
        )

    def __len__(self) -> int:  # pragma: no cover - placeholder
        raise NotImplementedError

    def _load(self, index: int) -> Sample:  # pragma: no cover - placeholder
        raise NotImplementedError
