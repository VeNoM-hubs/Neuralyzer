"""PROCESS Challenge dataset loader (open Kaggle re-upload).

Matches the actual layout of the Kaggle "dementia-detection-using-speech" upload,
which is a re-upload of the PROCESS Challenge data:

    <data_root>/
        Archive/
            Process-rec-001/
                Process-rec-001__CTD.wav   Process-rec-001__CTD.txt   (Cookie Theft Description)
                Process-rec-001__PFT.wav   Process-rec-001__PFT.txt   (Phonemic Fluency)
                Process-rec-001__SFT.wav   Process-rec-001__SFT.txt   (Semantic Fluency)
            ...
        test_Set/test_Set/Process-test-XXX/...   (no labels -> ignored)
        Data_AUG_13.11.2024_output.csv           (Record-ID, Class, Transcript_*, ...)

Design choices (faithful to the paper's picture-description paradigm):
  * Default task = CTD (Cookie Theft Description); PFT/SFT are word-list fluency
    tasks with very different transcripts, so they're excluded by default
    (configurable via ``tasks``).
  * Labels come from the CSV's ``Class`` column, mapped to the paper's binary
    grouping: HC -> 0, (MCI + Dementia/AD) -> 1.
  * Transcript is read from the co-located ``__CTD.txt`` (fallback to the CSV's
    ``Transcript_CTD``). No ASR needed -- transcripts already exist.

Only labeled ``Process-rec-*`` records are used; the trainer applies the
stratified 65/35 train/val split. Model/trainer/losses are unchanged.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

from .base import BaseDementiaDataset, Sample

# Class name -> binary label. 1 = cognitively impaired (paper groups MCI + AD).
_DEFAULT_LABEL_MAP: Dict[str, int] = {
    "hc": 0, "control": 0, "healthy": 0, "cn": 0, "cc": 0, "normal": 0,
    "mci": 1, "dementia": 1, "ad": 1, "probablead": 1, "alzheimer": 1, "pd": 1,
}


def _detect_labels_csv(data_root: str) -> Optional[str]:
    """Find a CSV under data_root that has both a Record-ID and a Class column."""
    for path in sorted(glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True)):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
                header = f.readline()
        except OSError:
            continue
        delim = ";" if header.count(";") > header.count(",") else ","
        cols = {c.strip().lower() for c in header.split(delim)}
        if "record-id" in cols and "class" in cols:
            return path
    return None


def _read_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        delim = ";" if sample.count(";") > sample.count(",") else ","
        return list(csv.DictReader(f, delimiter=delim))


class ProcessDataset(BaseDementiaDataset):
    def __init__(
        self,
        data_root: str,
        sample_rate: int = 16000,
        tasks: Sequence[str] = ("CTD",),
        labels_csv: Optional[str] = None,
        label_map: Optional[Dict[str, int]] = None,
        split: str = "train",
    ) -> None:
        super().__init__(sample_rate=sample_rate)
        if not os.path.isdir(data_root):
            raise FileNotFoundError(f"data_root '{data_root}' is not a directory.")
        self.data_root = data_root
        self.tasks = [t.upper() for t in tasks]
        self.label_map = {k.lower(): int(v) for k, v in (label_map or _DEFAULT_LABEL_MAP).items()}

        csv_path = labels_csv or _detect_labels_csv(data_root)
        if csv_path is None:
            raise FileNotFoundError(
                f"No labels CSV with 'Record-ID' + 'Class' columns found under '{data_root}'. "
                f"Set data.labels_csv to point at it (e.g. Data_AUG_13.11.2024_output.csv)."
            )
        rows = _read_rows(csv_path)

        # Map lower-cased headers so we're robust to exact casing.
        def key_of(row: Dict[str, str], name: str) -> Optional[str]:
            for k in row:
                if k and k.strip().lower() == name:
                    return k
            return None

        self._items: List[Tuple[str, str, int]] = []  # (audio_path, transcript, label)
        unknown_classes = set()
        missing_audio = 0

        for row in rows:
            rid_key = key_of(row, "record-id")
            cls_key = key_of(row, "class")
            if rid_key is None or cls_key is None:
                continue
            rec_id = (row[rid_key] or "").strip()
            cls = (row[cls_key] or "").strip().lower()
            if not rec_id or not cls:
                continue
            if cls not in self.label_map:
                unknown_classes.add(cls)
                continue
            label = self.label_map[cls]

            rec_dir = self._record_dir(rec_id)
            if rec_dir is None:
                continue
            for task in self.tasks:
                audio_path = os.path.join(rec_dir, f"{rec_id}__{task}.wav")
                if not os.path.exists(audio_path):
                    missing_audio += 1
                    continue
                transcript = self._transcript(rec_dir, rec_id, task, row)
                self._items.append((audio_path, transcript, label))

        if not self._items:
            hint = f" Unknown Class values: {sorted(unknown_classes)}." if unknown_classes else ""
            raise ValueError(
                f"No usable (audio, label) pairs built from '{csv_path}'.{hint} "
                f"Checked tasks={self.tasks}, missing_audio={missing_audio}. "
                f"Verify data.label_map covers the Class values and Archive/<Record-ID>/ exists."
            )
        self._items.sort()

    def _record_dir(self, rec_id: str) -> Optional[str]:
        for cand in (os.path.join(self.data_root, "Archive", rec_id),
                     os.path.join(self.data_root, rec_id)):
            if os.path.isdir(cand):
                return cand
        return None

    def _transcript(self, rec_dir: str, rec_id: str, task: str, row: Dict[str, str]) -> str:
        txt_path = os.path.join(rec_dir, f"{rec_id}__{task}.txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if text:
                return text
        # Fallback: CSV column Transcript_<TASK>
        for k in row:
            if k and k.strip().lower() == f"transcript_{task.lower()}":
                return (row[k] or "").strip()
        return ""

    def __len__(self) -> int:
        return len(self._items)

    def _load(self, index: int) -> Sample:
        from .kaggle_pitt import _lazy_load_audio

        audio_path, transcript, label = self._items[index]
        audio = _lazy_load_audio(audio_path, self.sample_rate)
        return {"audio": audio, "transcript": transcript, "label": label}
