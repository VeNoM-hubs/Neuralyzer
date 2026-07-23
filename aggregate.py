"""Merge per-seed results from one or more summary.csv files into a single
mean +/- std table (paper Sec V-C).

Use this when the 5-seed protocol was split across GPUs/processes (each writes
its own summary.csv), so no single run printed the full aggregate.

Usage:
    python aggregate.py /kaggle/working/gpu0/summary.csv /kaggle/working/gpu1/summary.csv
    python aggregate.py outputs/**/summary.csv          # shell-expanded globs
"""

from __future__ import annotations

import argparse
import csv
import glob
from typing import Dict, List

from src.evaluation import METRIC_KEYS, aggregate_runs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate summary.csv files into mean +/- std.")
    p.add_argument("paths", nargs="+", help="One or more summary.csv paths (globs allowed).")
    return p.parse_args()


def load_rows(paths: List[str]) -> List[Dict[str, float]]:
    runs: List[Dict[str, float]] = []
    seen_seeds = set()
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    seed = row.get("seed")
                    # De-dupe seeds that appear in overlapping files.
                    if seed is not None and seed in seen_seeds:
                        continue
                    if seed is not None:
                        seen_seeds.add(seed)
                    runs.append({k: float(row[k]) for k in METRIC_KEYS if k in row})
    return runs


def main() -> None:
    args = parse_args()
    runs = load_rows(args.paths)
    if not runs:
        raise SystemExit("No rows found in the provided summary.csv file(s).")

    agg = aggregate_runs(runs)
    print(f"Aggregated over {len(runs)} run(s):")
    for key in METRIC_KEYS:
        print(f"  {key:12s}: {agg[key]['mean'] * 100:.2f} +/- {agg[key]['std'] * 100:.2f}")


if __name__ == "__main__":
    main()
