"""Ablation grids (paper Sec VII) as config overrides.

Each function returns a list of (variant_name, overrides) applied on top of a
base config. The ``ablate.py`` entrypoint iterates a chosen study, trains each
variant with the multi-seed protocol, and reports mean +/- std.

Bilinear fusions (mutan/mfb/mfh/block) use reference-default factorization
hyperparameters (paper does not specify them); see modules/bilinear_fusion.py.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Override = Tuple[str, Dict[str, Dict]]


def pooling_study() -> List[Override]:
    return [
        ("mean", {"model": {"pooling": "mean"}}),
        ("max", {"model": {"pooling": "max"}}),
        ("asp", {"model": {"pooling": "asp"}}),
    ]


def ssl_study() -> List[Override]:
    return [
        ("hubert", {"model": {"speech_model_name": "facebook/hubert-base-ls960"}}),
        ("wav2vec2", {"model": {"speech_model_name": "facebook/wav2vec2-base-960h"}}),
        ("xls-r", {"model": {"speech_model_name": "facebook/wav2vec2-xls-r-300m"}}),
    ]


def lambda_study() -> List[Override]:
    return [(f"lambda_{lam}", {"train": {"mine_lambda": lam}})
            for lam in (0.0, 0.1, 0.2, 0.25, 0.3)]


def fusion_study() -> List[Override]:
    return [
        ("concat", {"model": {"fusion": "concat"}}),
        ("gmu", {"model": {"fusion": "gmu"}}),
        ("mutan", {"model": {"fusion": "mutan"}}),
        ("mfb", {"model": {"fusion": "mfb"}}),
        ("mfh", {"model": {"fusion": "mfh"}}),
        ("block", {"model": {"fusion": "block"}}),
        ("at_fusion", {"model": {"fusion": "at_fusion"}}),
    ]


def hubert_layers_study() -> List[Override]:
    return [
        ("last_1", {"model": {"num_speech_layers": 1}}),
        ("last_2", {"model": {"num_speech_layers": 2}}),
        ("last_3", {"model": {"num_speech_layers": 3}}),
    ]


STUDIES = {
    "pooling": pooling_study,
    "ssl": ssl_study,
    "lambda": lambda_study,
    "fusion": fusion_study,
    "layers": hubert_layers_study,
}
