"""Training entrypoint (run on Google Colab GPU).

Runs the multi-seed protocol (paper: 5 runs, report mean +/- std). For each
seed it builds a stratified 65/35 split, trains with early stopping, and
evaluates the best checkpoint. The single best run (by validation F1) is copied
to <output_dir>/best_model.pt -- that is the ONE file you download to run
inference locally.

Usage (Colab):
    python train.py --config src/configs/default.yaml
    python train.py --config src/configs/default.yaml --dataset mock --max-epochs 3 --single-seed
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

# Reduce CUDA fragmentation (frees "reserved but unallocated" memory). Must be
# set before torch initializes its CUDA allocator, i.e. before importing src.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from src.configs import Config, load_config, save_config
from src.evaluation import METRIC_KEYS, aggregate_runs
from src.trainer import Trainer, build_dataloaders
from src.utils import CSVLogger, get_console_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the multimodal dementia model.")
    p.add_argument("--config", type=str, default="src/configs/default.yaml")
    p.add_argument("--dataset", type=str, default=None, help="Override data.dataset.")
    p.add_argument("--data-root", type=str, default=None, help="Override data.data_root.")
    p.add_argument("--transcripts-file", type=str, default=None,
                   help="Override data.transcripts_file (kaggle_pitt).")
    p.add_argument("--label-map", type=str, default=None,
                   help='Override data.label_map as JSON, e.g. \'{"Dementia":1,"Control":0}\'.')
    p.add_argument("--labels-csv", type=str, default=None,
                   help="Override data.labels_csv (process dataset).")
    p.add_argument("--tasks", type=str, default=None,
                   help="process tasks, comma-separated, e.g. 'CTD' or 'CTD,SFT,PFT'.")
    p.add_argument("--output-dir", type=str, default=None, help="Override train.output_dir.")
    p.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size.")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="Override data.max_chunks_per_recording (lower if OOM; 0 = no cap).")
    p.add_argument("--max-epochs", type=int, default=None, help="Override train.max_epochs.")
    p.add_argument("--lr", type=float, default=None, help="Override train.lr (peak LR).")
    p.add_argument("--patience", type=int, default=None,
                   help="Override train.early_stopping_patience.")
    p.add_argument("--scheduler", type=str, default=None,
                   choices=["linear_warmup", "cosine_warmup", "steplr"],
                   help="Override train.scheduler.")
    p.add_argument("--single-seed", action="store_true", help="Run only the first seed.")
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seeds to run, e.g. '42,43,44'. Overrides the "
                        "config list; handy for pinning seed subsets to separate GPUs "
                        "(run two processes with different CUDA_VISIBLE_DEVICES).")
    return p.parse_args()


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    if args.dataset is not None:
        cfg.data.dataset = args.dataset
    if args.data_root is not None:
        cfg.data.data_root = args.data_root
    if args.transcripts_file is not None:
        cfg.data.transcripts_file = args.transcripts_file
    if args.label_map is not None:
        cfg.data.label_map = {k: int(v) for k, v in json.loads(args.label_map).items()}
    if args.labels_csv is not None:
        cfg.data.labels_csv = args.labels_csv
    if args.tasks is not None:
        cfg.data.process_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if args.output_dir is not None:
        cfg.train.output_dir = args.output_dir
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.max_chunks is not None:
        cfg.data.max_chunks_per_recording = None if args.max_chunks <= 0 else args.max_chunks
    if args.max_epochs is not None:
        cfg.train.max_epochs = args.max_epochs
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.patience is not None:
        cfg.train.early_stopping_patience = args.patience
    if args.scheduler is not None:
        cfg.train.scheduler = args.scheduler
    if args.seeds is not None:
        cfg.train.seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    logger = get_console_logger()
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    save_config(cfg, os.path.join(cfg.train.output_dir, "config.yaml"))

    seeds = cfg.train.seeds[:1] if args.single_seed else cfg.train.seeds
    summary = CSVLogger(os.path.join(cfg.train.output_dir, "summary.csv"))
    run_results = []
    best_run = {"f1": -1.0, "ckpt": None, "seed": None}

    for run_idx, seed in enumerate(seeds):
        logger.info(f"===== Run {run_idx + 1}/{len(seeds)} (seed={seed}) =====")
        cfg.train.seed = seed
        set_seed(seed)

        train_loader, val_loader = build_dataloaders(cfg)
        run_dir = os.path.join(cfg.train.output_dir, f"seed_{seed}")
        trainer = Trainer(cfg, run_dir=run_dir)
        trainer.fit(train_loader, val_loader)
        metrics = trainer.evaluate(val_loader, load_best=True)

        logger.info(f"[seed {seed}] " + " ".join(f"{k}={metrics[k]:.4f}" for k in METRIC_KEYS))
        summary.log({"seed": seed, **{k: metrics[k] for k in METRIC_KEYS}})
        run_results.append(metrics)

        if metrics["f1"] > best_run["f1"]:
            best_run = {"f1": metrics["f1"], "ckpt": trainer.ckpt_path, "seed": seed}

    agg = aggregate_runs(run_results)
    logger.info("===== Aggregated (mean +/- std) =====")
    for key in METRIC_KEYS:
        logger.info(f"{key:12s}: {agg[key]['mean'] * 100:.2f} +/- {agg[key]['std'] * 100:.2f}")

    # Copy the best run's checkpoint to a single, easy-to-download file.
    if best_run["ckpt"] and os.path.exists(best_run["ckpt"]):
        final_path = os.path.join(cfg.train.output_dir, "best_model.pt")
        shutil.copyfile(best_run["ckpt"], final_path)
        logger.info(f"Best checkpoint (seed {best_run['seed']}, F1={best_run['f1']:.4f}) "
                    f"copied to: {final_path}")
        logger.info("Download this best_model.pt to run inference locally.")


if __name__ == "__main__":
    main()
