"""Ablation-study entrypoint (paper Sec VII; run on Colab).

Runs one ablation dimension end-to-end, training each variant with the
multi-seed protocol and printing a mean +/- std table.

Usage:
    python ablate.py --study pooling
    python ablate.py --study lambda --config src/configs/default.yaml
    python ablate.py --study ssl --single-seed --max-epochs 3

Studies: pooling | ssl | lambda | fusion | layers
"""

from __future__ import annotations

import argparse
import copy
import os

from src.configs import Config, load_config
from src.evaluation import METRIC_KEYS, aggregate_runs
from src.evaluation.ablations import STUDIES
from src.trainer import Trainer, build_dataloaders
from src.utils import get_console_logger
from src.utils.seed import set_seed


def _apply(cfg: Config, overrides: dict) -> Config:
    cfg = copy.deepcopy(cfg)
    for section, values in overrides.items():
        target = getattr(cfg, section)
        for k, v in values.items():
            setattr(target, k, v)
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a paper ablation study.")
    p.add_argument("--study", type=str, required=True, choices=sorted(STUDIES.keys()))
    p.add_argument("--config", type=str, default="src/configs/default.yaml")
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--single-seed", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    if args.dataset is not None:
        base.data.dataset = args.dataset
    if args.max_epochs is not None:
        base.train.max_epochs = args.max_epochs

    logger = get_console_logger()
    variants = STUDIES[args.study]()
    table = {}

    for variant_name, overrides in variants:
        cfg = _apply(base, overrides)
        seeds = cfg.train.seeds[:1] if args.single_seed else cfg.train.seeds
        logger.info(f"----- ablation[{args.study}] variant={variant_name} -----")

        run_results = []
        try:
            for seed in seeds:
                cfg.train.seed = seed
                set_seed(seed)
                train_loader, val_loader = build_dataloaders(cfg)
                run_dir = os.path.join(cfg.train.output_dir, f"ablation_{args.study}",
                                       variant_name, f"seed_{seed}")
                trainer = Trainer(cfg, run_dir=run_dir)
                trainer.fit(train_loader, val_loader)
                run_results.append(trainer.evaluate(val_loader, load_best=True))
            table[variant_name] = aggregate_runs(run_results)
        except NotImplementedError as exc:
            logger.warning(f"Skipping '{variant_name}': {exc}")
            table[variant_name] = None

    logger.info(f"===== Ablation '{args.study}' summary (mean +/- std, %) =====")
    logger.info("variant".ljust(14) + "".join(k[:6].ljust(10) for k in METRIC_KEYS))
    for variant_name, agg in table.items():
        if agg is None:
            logger.info(variant_name.ljust(14) + "(not implemented)")
            continue
        row = variant_name.ljust(14) + "".join(
            f"{agg[k]['mean'] * 100:.2f}".ljust(10) for k in METRIC_KEYS
        )
        logger.info(row)


if __name__ == "__main__":
    main()
