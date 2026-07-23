"""Training loop, checkpointing, early stopping, and scheduler (runs on Colab).

Paper-specified: batch size 8; early stopping if val loss doesn't improve for 8
epochs; StepLR (step_size=4, gamma=0.1); loss = CE + lambda * MINE (lambda=0.25).
User-decided: AdamW (lr 2e-5, wd 0.01, betas (0.9,0.999)), grad clip 1.0,
best-val-loss checkpointing, CSV + console logging.

The saved checkpoint (best.pt) bundles the model weights AND the full config, so
it can be loaded locally for inference with no extra metadata.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoTokenizer

from ..configs import Config
from ..datasets import MultimodalCollator, build_dataset
from ..evaluation import compute_metrics
from ..losses import combined_loss
from ..models import MultimodalDementiaModel
from ..utils import CSVLogger, get_console_logger, resolve_device


def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders with a stratified 65/35 split."""
    from sklearn.model_selection import train_test_split

    feature_extractor = AutoFeatureExtractor.from_pretrained(cfg.model.speech_model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_model_name)
    collator = MultimodalCollator(
        feature_extractor=feature_extractor, tokenizer=tokenizer,
        sample_rate=cfg.data.sample_rate, chunk_seconds=cfg.data.chunk_seconds,
        max_text_length=cfg.data.max_text_length,
    )

    full = build_dataset(cfg.data, split="train")
    labels = full.labels()
    indices = list(range(len(full)))
    train_idx, val_idx = train_test_split(
        indices, test_size=cfg.data.val_ratio, stratify=labels, random_state=cfg.train.seed,
    )

    train_subset = torch.utils.data.Subset(full, train_idx)
    val_subset = torch.utils.data.Subset(full, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory, collate_fn=collator,
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=cfg.data.pin_memory, collate_fn=collator,
    )
    return train_loader, val_loader


class Trainer:
    def __init__(self, cfg: Config, run_dir: str) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.device = resolve_device(cfg.train.device)
        self.logger = get_console_logger()
        self.csv = CSVLogger(os.path.join(run_dir, "train_log.csv"))
        self.ckpt_path = os.path.join(run_dir, "best.pt")

        self.model = MultimodalDementiaModel(cfg.model).to(self.device)
        self.optimizer = self._build_optimizer()
        # Scheduler is built in fit() (needs steps-per-epoch for warmup schedules).
        self.scheduler = None
        self.step_scheduler_per_batch = False
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        name = self.cfg.train.optimizer.lower()
        if name == "adamw":
            return torch.optim.AdamW(params, lr=self.cfg.train.lr,
                                     weight_decay=self.cfg.train.weight_decay, betas=self.cfg.train.betas)
        if name == "adam":
            return torch.optim.Adam(params, lr=self.cfg.train.lr,
                                    weight_decay=self.cfg.train.weight_decay, betas=self.cfg.train.betas)
        if name == "sgd":
            return torch.optim.SGD(params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)
        raise ValueError(f"Unknown optimizer '{self.cfg.train.optimizer}'.")

    def _build_scheduler(self, steps_per_epoch: int) -> None:
        """Build the LR scheduler once the dataloader size is known.

        - steplr: paper-exact StepLR, stepped once per epoch.
        - linear_warmup / cosine_warmup: warmup then decay across the full run
          (total_steps = steps_per_epoch * max_epochs), stepped once per batch.
        """
        name = self.cfg.train.scheduler.lower()
        if name == "steplr":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.cfg.train.scheduler_step_size,
                gamma=self.cfg.train.scheduler_gamma,
            )
            self.step_scheduler_per_batch = False
            return

        total_steps = max(1, steps_per_epoch * self.cfg.train.max_epochs)
        warmup_steps = int(self.cfg.train.warmup_ratio * total_steps)
        if name == "linear_warmup":
            from transformers import get_linear_schedule_with_warmup

            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
            )
        elif name == "cosine_warmup":
            from transformers import get_cosine_schedule_with_warmup

            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
            )
        else:
            raise ValueError(
                f"Unknown scheduler '{self.cfg.train.scheduler}'. "
                f"Expected linear_warmup|cosine_warmup|steplr."
            )
        self.step_scheduler_per_batch = True

    def _to_device(self, batch: Dict) -> Dict:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total_loss = total_ce = total_mi = 0.0
        n_batches = 0
        all_preds: List[int] = []
        all_labels: List[int] = []

        grad_context = torch.enable_grad() if train else torch.no_grad()
        with grad_context:
            for batch in loader:
                batch = self._to_device(batch)
                labels = batch["labels"]

                if train:
                    self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.cfg.train.use_amp):
                    out = self.model(batch)
                    loss, ce, mi = combined_loss(
                        out.logits, labels, self.model.mine, out.f_a, out.f_t,
                        mine_lambda=self.cfg.train.mine_lambda,
                    )

                if train:
                    self.scaler.scale(loss).backward()
                    if self.cfg.train.grad_clip and self.cfg.train.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.step_scheduler_per_batch and self.scheduler is not None:
                        self.scheduler.step()

                total_loss += float(loss.item())
                total_ce += float(ce.item())
                total_mi += float(mi.item())
                n_batches += 1
                preds = out.logits.argmax(dim=1)
                all_preds.extend(preds.detach().cpu().tolist())
                all_labels.extend(labels.detach().cpu().tolist())

        metrics = compute_metrics(all_preds, all_labels)
        metrics.update({
            "loss": total_loss / max(n_batches, 1),
            "ce": total_ce / max(n_batches, 1),
            "mi": total_mi / max(n_batches, 1),
        })
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        best_val_loss = float("inf")
        epochs_without_improve = 0
        best_val_metrics: Dict[str, float] = {}

        self._build_scheduler(steps_per_epoch=max(1, len(train_loader)))

        for epoch in range(1, self.cfg.train.max_epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)
            if self.scheduler is not None and not self.step_scheduler_per_batch:
                self.scheduler.step()  # steplr: once per epoch (paper)

            self.logger.info(
                f"epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} "
                f"acc {train_metrics['accuracy']:.3f} | val loss {val_metrics['loss']:.4f} "
                f"acc {val_metrics['accuracy']:.3f} f1 {val_metrics['f1']:.3f}"
            )
            self.csv.log({
                "epoch": epoch, "lr": self.optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"], "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"], "val_recall": val_metrics["recall"],
                "val_specificity": val_metrics["specificity"],
            })

            if val_metrics["loss"] < best_val_loss - 1e-6:
                best_val_loss = val_metrics["loss"]
                best_val_metrics = val_metrics
                epochs_without_improve = 0
                self.save_checkpoint(val_metrics)
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= self.cfg.train.early_stopping_patience:
                    self.logger.info(
                        f"Early stopping at epoch {epoch} "
                        f"(no val-loss improvement for {epochs_without_improve} epochs)."
                    )
                    break

        return best_val_metrics

    def save_checkpoint(self, val_metrics: Dict[str, float]) -> None:
        """Save best.pt bundling weights + full config (portable for inference)."""
        torch.save({
            "model": self.model.state_dict(),
            "config": self.cfg.to_dict(),
            "val_metrics": val_metrics,
        }, self.ckpt_path)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, load_best: bool = True) -> Dict[str, float]:
        if load_best and os.path.exists(self.ckpt_path):
            state = torch.load(self.ckpt_path, map_location=self.device)
            self.model.load_state_dict(state["model"])
        return self._run_epoch(loader, train=False)
