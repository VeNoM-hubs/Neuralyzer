"""Open-access Kaggle "Dementia_Detection_using_speech" loader (Pitt Corpus re-upload).

Source: kaggle.com/datasets/tahouramorovati/dementia-detection-using-speech
  * English Cookie-Theft picture description (same paradigm as the paper).
  * Derived from DementiaBank's Pitt Corpus, ~442 recordings, class-separated
    into sub-folders (dementia vs. control).
  * AUDIO ONLY -- no transcripts. Generate them once with
    ``scripts/transcribe_whisper.py`` (Whisper), exactly as the paper does for
    datasets without provided transcripts. This loader then reads the cached
    ``transcripts.json``.

LICENSE CAVEAT: this is a re-upload of DementiaBank data (originally under a
data-use agreement). Acceptable for personal reproduction/learning; not for
publication or production without proper DementiaBank access.

Only this loader is dataset-specific; the model/trainer/losses are unchanged.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import torch

from .base import BaseDementiaDataset, Sample

_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a")

# Default folder-name -> label mapping (case-insensitive). 1 = dementia/AD, 0 = control.
_DEFAULT_LABEL_MAP: Dict[str, int] = {
    "dementia": 1, "ad": 1, "cd": 1, "patient": 1, "patients": 1, "alzheimer": 1,
    "control": 0, "controls": 0, "hc": 0, "cn": 0, "cc": 0, "healthy": 0, "normal": 0,
}


def _lazy_load_audio(path: str, target_sr: int) -> torch.Tensor:
    """Load an audio file as mono float32 at target_sr (torchaudio, soundfile fallback)."""
    try:
        import torchaudio

        waveform, sr = torchaudio.load(path)  # [C, T]
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform.mean(dim=0).to(torch.float32)
    except Exception:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32")
        wav = torch.as_tensor(data, dtype=torch.float32)
        if wav.dim() > 1:
            wav = wav.mean(dim=1)
        if sr != target_sr:
            import torchaudio

            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav


class KagglePittDataset(BaseDementiaDataset):
    def __init__(
        self,
        data_root: str,
        sample_rate: int = 16000,
        transcripts_file: Optional[str] = None,
        label_map: Optional[Dict[str, int]] = None,
        split: str = "train",
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"data_root '{data_root}' is not a directory.")
        self.data_root = data_root
        self.label_map = {k.lower(): int(v) for k, v in (label_map or _DEFAULT_LABEL_MAP).items()}

        # --- Discover (audio_path, label) pairs by scanning class sub-folders ---
        self._items: List[Tuple[str, int]] = []
        unknown_folders = set()
        for dirpath, _dirnames, filenames in os.walk(data_root):
            audio_files = [f for f in filenames if f.lower().endswith(_AUDIO_EXTS)]
            if not audio_files:
                continue
            label = self._label_for_path(dirpath, data_root, unknown_folders)
            if label is None:
                continue
            for f in sorted(audio_files):
                self._items.append((os.path.join(dirpath, f), label))

        if not self._items:
            hint = (f" Found unmapped class folders: {sorted(unknown_folders)}."
                    if unknown_folders else "")
            raise ValueError(
                f"No labeled audio found under '{data_root}'.{hint} "
                f"Set data.label_map to map each class folder name to 0 (control) or 1 (dementia)."
            )
        self._items.sort()

        # --- Load cached Whisper transcripts (keyed by file stem) ---
        tpath = transcripts_file or os.path.join(data_root, "transcripts.json")
        if not os.path.exists(tpath):
            raise FileNotFoundError(
                f"Transcripts file not found: '{tpath}'. The Kaggle dataset is audio-only; "
                f"generate transcripts first:\n"
                f"    python scripts/transcribe_whisper.py --data-root {data_root}\n"
                f"or set data.transcripts_file to an existing JSON of {{wav_stem: transcript}}."
            )
        with open(tpath, "r", encoding="utf-8") as fh:
            self._transcripts: Dict[str, str] = json.load(fh)

    def _label_for_path(self, dirpath: str, root: str, unknown: set) -> Optional[int]:
        """Derive a label from the path components between root and the file."""
        rel = os.path.relpath(dirpath, root)
        parts = [p.lower() for p in rel.split(os.sep) if p not in (".", "")]
        for part in parts:
            if part in self.label_map:
                return self.label_map[part]
        for part in parts:
            unknown.add(part)
        return None

    def _transcript_for(self, audio_path: str) -> str:
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        if stem in self._transcripts:
            return self._transcripts[stem]
        # also try the relative path key
        rel = os.path.relpath(audio_path, self.data_root)
        if rel in self._transcripts:
            return self._transcripts[rel]
        raise KeyError(
            f"No transcript for '{stem}'. Re-run scripts/transcribe_whisper.py to cover all files."
        )

    def __len__(self) -> int:
        return len(self._items)

    def _load(self, index: int) -> Sample:
        audio_path, label = self._items[index]
        audio = _lazy_load_audio(audio_path, self.sample_rate)
        transcript = self._transcript_for(audio_path)
        return {"audio": audio, "transcript": transcript, "label": label}
