"""ADReSS Challenge dataset loader (PLACEHOLDER - awaiting data access).

Gated via DementiaBank/TalkBank: https://dementia.talkbank.org/ (signed
data-use agreement required; not publicly downloadable).

Typical layout once obtained:
    <data_root>/
        train/Full_wave_enhanced_audio/{cc,cd}/*.wav   # cc=control(0), cd=AD(1)
        train/transcription/{cc,cd}/*.cha              # CHAT transcripts
        test/...

Deferred until the real files (and exact CHAT format / official split) can be
inspected -- per CURSOR.md, do not infer unspecified details. Only this Dataset
class needs completing; model/trainer/losses stay unchanged.
"""

from __future__ import annotations

from .base import BaseDementiaDataset, Sample


class ADReSSDataset(BaseDementiaDataset):
    def __init__(self, data_root: str, split: str = "train", sample_rate: int = 16000):
        super().__init__(sample_rate=sample_rate)
        raise NotImplementedError(
            "ADReSSDataset is a placeholder. The dataset is gated "
            "(dementia.talkbank.org) and its exact layout / CHAT transcript format "
            "must be inspected before implementing. Only this Dataset class needs "
            "completing; the model, trainer, and losses stay unchanged."
        )

    def __len__(self) -> int:  # pragma: no cover - placeholder
        raise NotImplementedError

    def _load(self, index: int) -> Sample:  # pragma: no cover - placeholder
        raise NotImplementedError
