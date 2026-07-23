"""Local inference entrypoint.

Loads a checkpoint trained on Colab (best_model.pt, which bundles the full
config) and predicts the dementia class for one (audio, transcript) pair. This
is the ONLY thing you run locally -- no training dependencies needed.

Usage:
    python inference.py --checkpoint best_model.pt \
        --audio path/to/recording.wav \
        --transcript "the boy is taking cookies from the jar ..."

    # or read the transcript from a text file:
    python inference.py --checkpoint best_model.pt --audio rec.wav --transcript-file rec.txt
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoFeatureExtractor, AutoTokenizer

from src.configs import config_from_dict
from src.datasets import MultimodalCollator
from src.models import MultimodalDementiaModel
from src.utils import resolve_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run local inference with a trained checkpoint.")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    p.add_argument("--audio", type=str, required=True, help="Path to a .wav file.")
    p.add_argument("--transcript", type=str, default=None)
    p.add_argument("--transcript-file", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    """Load a wav as mono float32 at target_sr. Uses torchaudio if available,
    else soundfile, so local deps stay minimal."""
    try:
        import torchaudio

        waveform, sr = torchaudio.load(path)  # [C, T]
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform.mean(dim=0)  # mono [T]
    except Exception:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32")  # [T] or [T, C]
        wav = torch.as_tensor(data, dtype=torch.float32)
        if wav.dim() > 1:
            wav = wav.mean(dim=1)
        if sr != target_sr:
            import torchaudio

            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if args.transcript is None and args.transcript_file is None:
        raise SystemExit("Provide --transcript or --transcript-file.")
    transcript = args.transcript
    if args.transcript_file:
        with open(args.transcript_file, "r", encoding="utf-8") as f:
            transcript = f.read().strip()

    state = torch.load(args.checkpoint, map_location=device)
    cfg = config_from_dict(state["config"])

    model = MultimodalDementiaModel(cfg.model).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model.speech_model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_model_name)
    collator = MultimodalCollator(
        feature_extractor=feature_extractor, tokenizer=tokenizer,
        sample_rate=cfg.data.sample_rate, chunk_seconds=cfg.data.chunk_seconds,
        max_text_length=cfg.data.max_text_length,
        max_chunks_per_recording=cfg.data.max_chunks_per_recording,
    )

    waveform = load_audio(args.audio, cfg.data.sample_rate)  # [T]
    batch = collator([{"audio": waveform, "transcript": transcript, "label": 0}])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    with torch.no_grad():
        out = model(batch)
        probs = torch.softmax(out.logits, dim=1)[0]
        pred = int(probs.argmax().item())

    label_name = {0: "control / healthy", 1: "AD / cognitively impaired"}[pred]
    print(f"Prediction: {pred} ({label_name})")
    print(f"Probabilities: control={probs[0].item():.4f}  impaired={probs[1].item():.4f}")


if __name__ == "__main__":
    main()
