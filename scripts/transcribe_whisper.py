"""Generate transcripts for an audio-only dataset using OpenAI Whisper.

The Kaggle Pitt re-upload ships audio without transcripts. This script runs
Whisper over every audio file under ``--data-root`` and writes a
``transcripts.json`` mapping {file_stem: transcript}, which KagglePittDataset
then reads. This mirrors the paper's own use of ASR transcripts for datasets
that don't provide them.

Run this ONCE on Colab (GPU makes it fast) before training:
    python scripts/transcribe_whisper.py --data-root /content/pitt --model base

Install: pip install -U openai-whisper   (or faster-whisper for speed)
"""

from __future__ import annotations

import argparse
import json
import os

_AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a")


def find_audio(root: str):
    for dirpath, _dirs, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(_AUDIO_EXTS):
                yield os.path.join(dirpath, f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Whisper-transcribe an audio dataset.")
    p.add_argument("--data-root", type=str, required=True, help="Root folder of audio files.")
    p.add_argument("--out", type=str, default=None, help="Output JSON (default <data_root>/transcripts.json).")
    p.add_argument("--model", type=str, default="base",
                   help="Whisper size: tiny|base|small|medium|large-v3.")
    p.add_argument("--language", type=str, default="en")
    p.add_argument("--overwrite", action="store_true", help="Re-transcribe files already cached.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = args.out or os.path.join(args.data_root, "transcripts.json")

    transcripts = {}
    if os.path.exists(out_path) and not args.overwrite:
        with open(out_path, "r", encoding="utf-8") as f:
            transcripts = json.load(f)
        print(f"[transcribe] loaded {len(transcripts)} existing transcripts from {out_path}")

    import whisper  # imported here so the dependency is only needed for this step

    model = whisper.load_model(args.model)
    files = list(find_audio(args.data_root))
    print(f"[transcribe] {len(files)} audio files under {args.data_root}")

    for i, path in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in transcripts and not args.overwrite:
            continue
        result = model.transcribe(path, language=args.language, fp16=False)
        transcripts[stem] = result.get("text", "").strip()
        if i % 20 == 0 or i == len(files):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(transcripts, f, ensure_ascii=False, indent=0)
            print(f"[transcribe] {i}/{len(files)} done (checkpointed)")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=0)
    print(f"[transcribe] wrote {len(transcripts)} transcripts -> {out_path}")


if __name__ == "__main__":
    main()
