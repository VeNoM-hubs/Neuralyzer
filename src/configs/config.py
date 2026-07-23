"""Configuration dataclasses for the multimodal dementia detection framework.

All tunable knobs live here so that no path or hyperparameter is ever
hardcoded inside a module (see CURSOR.md "Coding Rules"). Values map directly
to the paper's specified settings; every previously-unspecified detail has been
resolved via explicit user decision and is annotated below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass
class DataConfig:
    """Dataset + preprocessing configuration.

    Only the ``dataset`` implementation is allowed to change between the
    temporary development dataset and ADReSS/PROCESS-2 (see CURSOR.md
    "DEVELOPMENT DATASET POLICY"). All fields here are dataset-agnostic.
    """

    dataset: str = "mock"  # "mock" | "kaggle_pitt" | "adress" | "process2"
    data_root: Optional[str] = None  # never hardcoded; supplied via config/CLI

    # Audio (user decision): 16 kHz, resample if needed, no silence removal.
    sample_rate: int = 16000
    chunk_seconds: float = 10.0  # paper: fixed 10-second segments

    # Text (user decision): BERT max_length 512, truncation on.
    max_text_length: int = 512

    # --- kaggle_pitt (open-access Kaggle Pitt re-upload) options ---
    # JSON file mapping {wav_stem: transcript}, produced by scripts/transcribe_whisper.py.
    # If None, the loader looks for <data_root>/transcripts.json. The Kaggle dataset
    # ships audio only, so transcripts must be generated with Whisper first.
    transcripts_file: Optional[str] = None
    # Maps class sub-folder name -> label (0 control / 1 dementia). If None, the
    # loader uses a default map of common names and errors on any unknown folder
    # (so nothing is silently mislabeled).
    label_map: Optional[Dict[str, int]] = None

    # Validation split (paper: 65/35), stratified by label (user decision).
    val_ratio: float = 0.35

    num_workers: int = 0
    pin_memory: bool = False


@dataclass
class ModelConfig:
    """Architecture configuration.

    Checkpoints (user decision): base variants, both fine-tuned end-to-end.
    Sizes stay swappable so large variants can be tried later on a big GPU.
    """

    # --- Speech branch ---
    # Ablation "Speech Model": swap this name for wav2vec2.0 / XLS-R.
    speech_model_name: str = "facebook/hubert-base-ls960"
    # Ablation "HuBERT Layers": sum the last N hidden layers (1 / 2 / 3).
    num_speech_layers: int = 2
    # Ablation "Pooling": "asp" | "mean" | "max".
    pooling: str = "asp"
    asp_bottleneck: int = 128  # ASP attention hidden size (unspecified -> small default)
    speech_embed_dim: int = 768  # linear projection output (matches figure [B,768])

    # --- Text branch ---
    text_model_name: str = "bert-base-uncased"
    text_embed_dim: int = 768  # BERT-base [CLS] dimension

    # --- Fusion ---
    # Ablation "Fusion": at_fusion | concat | gmu | mutan | mfb | mfh | block.
    fusion: str = "at_fusion"
    fused_dim: int = 768

    # Bilinear-fusion hyperparameters (MUTAN/MFB/MFH/BLOCK). The paper does NOT
    # specify these; values below are the reference defaults from the original
    # papers / the block.bootstrap.pytorch library (explicit user-approved
    # assumption). Only used when ``fusion`` selects a bilinear method.
    bilinear_mm_dim: int = 1600  # MUTAN/BLOCK projection dim
    bilinear_mfb_mm_dim: int = 1200  # MFB/MFH projection dim
    bilinear_rank: int = 15  # MUTAN/BLOCK rank
    bilinear_factor: int = 2  # MFB/MFH pooling factor k
    bilinear_chunks: int = 20  # BLOCK number of block-term chunks
    bilinear_dropout: float = 0.1

    # --- Classifier (user decision: 768 -> 128 -> 2, matches PROCESS-2 baseline hint) ---
    classifier_hidden: int = 128
    num_classes: int = 2
    dropout: float = 0.1  # user decision

    # --- MINE statistics network (user decision: 2 hidden Linear+ReLU, hidden 512) ---
    mine_hidden: int = 512

    # --- Fine-tune vs freeze (user decision: fine-tune both) ---
    freeze_speech: bool = False
    freeze_text: bool = False


@dataclass
class TrainConfig:
    """Training configuration (used on Colab).

    Paper-specified: batch size 8, early-stopping patience 8, StepLR
    (step_size=4, gamma=0.1), loss = CE + 0.25 * MINE.
    User-decided optimizer: AdamW, lr 2e-5, wd 0.01, betas (0.9, 0.999),
    grad clip 1.0. Seeds 42-46 across 5 runs.
    """

    batch_size: int = 8
    # Upper bound; early stopping terminates first. Increased from the paper's
    # setup (user request) so a lower LR has room to converge.
    max_epochs: int = 250
    early_stopping_patience: int = 20  # raised from 8 so longer runs aren't cut short

    # Optimizer (user decision)
    optimizer: str = "adamw"
    # Reduced from 2e-5 -> 1e-5 (user request): the more stable rate for jointly
    # fine-tuning HuBERT + BERT on a small dataset. Used as the PEAK LR when a
    # warmup scheduler is active.
    lr: float = 1e-5
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0

    # Scheduler. "linear_warmup" (default, user-optimized) or "cosine_warmup"
    # do warmup then decay over the full horizon (stepped per batch); "steplr"
    # reproduces the paper exactly (StepLR step_size=4, gamma=0.1, per epoch).
    scheduler: str = "linear_warmup"  # linear_warmup | cosine_warmup | steplr
    warmup_ratio: float = 0.1  # fraction of total steps spent warming up
    scheduler_step_size: int = 4  # used only when scheduler == "steplr" (paper)
    scheduler_gamma: float = 0.1  # used only when scheduler == "steplr" (paper)

    # Loss weighting (paper: lambda = 0.25). Ablation "Lambda": 0/0.1/0.2/0.25/0.3.
    mine_lambda: float = 0.25

    # Reproducibility (user decision)
    seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])

    device: str = "auto"  # "auto" resolves to cuda if available else cpu
    output_dir: str = "runs"
    use_amp: bool = False  # mixed precision (unspecified by paper -> off by default)


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def config_from_dict(d: dict) -> Config:
    """Rebuild a Config from a plain dict (e.g. loaded from a checkpoint)."""
    data = DataConfig(**(d.get("data") or {}))
    model = ModelConfig(**(d.get("model") or {}))
    train_raw = dict(d.get("train") or {})
    if train_raw.get("betas") is not None:
        train_raw["betas"] = tuple(train_raw["betas"])
    train = TrainConfig(**train_raw)
    return Config(data=data, model=model, train=train)


def load_config(path: str) -> Config:
    """Load a YAML file into a ``Config`` (missing keys fall back to defaults)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return config_from_dict(raw)


def save_config(config: Config, path: str) -> None:
    """Serialize a ``Config`` to YAML for run reproducibility."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
