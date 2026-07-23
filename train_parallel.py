"""Parallel multi-GPU launcher for the multi-seed protocol.

Splits the seed list across the available GPUs and runs ONE `train.py` process
per GPU, each pinned to a single device via CUDA_VISIBLE_DEVICES. Because MINE
needs the full batch, we do NOT shard a single model across GPUs; instead we run
independent seeds in parallel (embarrassingly parallel, ~Nx wall-clock speedup).

A background thread streams each child's stdout live with a [gpuN] prefix, and a
copy is written to <output-dir>/gpuN.log. When all processes finish, results are
aggregated across every gpuN/summary.csv into one mean +/- std table.

Usage (Kaggle 2xT4):
    python train_parallel.py --dataset process \
        --data-root /kaggle/input/<folder> \
        --output-dir /kaggle/working/outputs \
        --seeds 42,43,44,45,46

Any flag not consumed here (e.g. --dataset, --data-root, --tasks, --labels-csv,
--max-chunks, --batch-size, --max-epochs, --lr, --patience, --scheduler) is
forwarded verbatim to each train.py child.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from typing import Dict, List


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(description="Run the multi-seed protocol across multiple GPUs.")
    p.add_argument("--config", type=str, default="src/configs/default.yaml")
    p.add_argument("--output-dir", type=str, default="outputs",
                   help="Base dir; each GPU writes to <output-dir>/gpuN.")
    p.add_argument("--seeds", type=str, default="42,43,44,45,46",
                   help="Comma-separated seeds to distribute across GPUs.")
    p.add_argument("--gpus", type=str, default=None,
                   help="Comma-separated GPU indices, e.g. '0,1'. Default: all visible GPUs.")
    p.add_argument("--no-warm", action="store_true",
                   help="Skip pre-downloading HuBERT/BERT (skip only if the HF cache is warm).")
    return p.parse_known_args()


def discover_gpus(explicit: str | None) -> List[int]:
    if explicit:
        return [int(g.strip()) for g in explicit.split(",") if g.strip()]
    try:
        import torch

        n = torch.cuda.device_count()
    except Exception:
        n = 0
    return list(range(n))


def warm_hf_cache(config_path: str) -> None:
    """Download the encoders once so parallel children don't race the HF cache."""
    from src.configs import load_config

    cfg = load_config(config_path)
    from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

    print(f"[launcher] warming HF cache: {cfg.model.speech_model_name}, {cfg.model.text_model_name}")
    AutoModel.from_pretrained(cfg.model.speech_model_name)
    AutoModel.from_pretrained(cfg.model.text_model_name)
    AutoFeatureExtractor.from_pretrained(cfg.model.speech_model_name)
    AutoTokenizer.from_pretrained(cfg.model.text_model_name)
    print("[launcher] HF cache warm.")


def assign_seeds(seeds: List[int], gpus: List[int]) -> Dict[int, List[int]]:
    """Round-robin seeds onto GPUs; drop GPUs that get nothing."""
    buckets: Dict[int, List[int]] = {g: [] for g in gpus}
    for i, s in enumerate(seeds):
        buckets[gpus[i % len(gpus)]].append(s)
    return {g: ss for g, ss in buckets.items() if ss}


def stream_output(proc: subprocess.Popen, prefix: str, log_path: str) -> None:
    with open(log_path, "w", encoding="utf-8") as log:
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(f"{prefix} {line}")
            sys.stdout.flush()
            log.write(line)
            log.flush()


def main() -> None:
    args, passthrough = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    gpus = discover_gpus(args.gpus)

    if len(gpus) <= 1:
        print(f"[launcher] {len(gpus)} GPU(s) visible; nothing to parallelize. "
              f"Run train.py directly instead.")
        sys.exit(1)

    buckets = assign_seeds(seeds, gpus)
    print(f"[launcher] seed assignment: "
          + ", ".join(f"gpu{g}->{ss}" for g, ss in buckets.items()))

    if not args.no_warm:
        warm_hf_cache(args.config)

    os.makedirs(args.output_dir, exist_ok=True)
    repo_root = os.path.dirname(os.path.abspath(__file__))

    procs: List[tuple[int, subprocess.Popen, threading.Thread]] = []
    for gpu, seed_subset in buckets.items():
        out_dir = os.path.join(args.output_dir, f"gpu{gpu}")
        cmd = [
            sys.executable, "train.py",
            "--config", args.config,
            "--seeds", ",".join(str(s) for s in seed_subset),
            "--output-dir", out_dir,
        ] + passthrough

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)  # child sees exactly one GPU as cuda:0

        print(f"[launcher] gpu{gpu}: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, cwd=repo_root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        t = threading.Thread(
            target=stream_output,
            args=(proc, f"[gpu{gpu}]", os.path.join(args.output_dir, f"gpu{gpu}.log")),
            daemon=True,
        )
        t.start()
        procs.append((gpu, proc, t))

    failed = []
    for gpu, proc, t in procs:
        proc.wait()
        t.join()
        if proc.returncode != 0:
            failed.append(gpu)
            print(f"[launcher] gpu{gpu} FAILED (exit {proc.returncode}); see "
                  f"{os.path.join(args.output_dir, f'gpu{gpu}.log')}")

    if failed:
        print(f"[launcher] {len(failed)} process(es) failed: {failed}")
        sys.exit(1)

    # Aggregate across every gpuN/summary.csv.
    import csv

    from src.evaluation import METRIC_KEYS, aggregate_runs

    runs = []
    for gpu in buckets:
        summ = os.path.join(args.output_dir, f"gpu{gpu}", "summary.csv")
        if not os.path.exists(summ):
            continue
        with open(summ, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                runs.append({k: float(row[k]) for k in METRIC_KEYS if k in row})

    if runs:
        agg = aggregate_runs(runs)
        print(f"\n===== Aggregated over {len(runs)} run(s), all GPUs (mean +/- std) =====")
        for key in METRIC_KEYS:
            print(f"{key:12s}: {agg[key]['mean'] * 100:.2f} +/- {agg[key]['std'] * 100:.2f}")
    print("\n[launcher] done. Best checkpoints: "
          + ", ".join(os.path.join(args.output_dir, f"gpu{g}", "best_model.pt") for g in buckets))


if __name__ == "__main__":
    main()
