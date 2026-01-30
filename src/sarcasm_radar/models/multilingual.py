"""Multilingual transformer support for Hinglish code-switching.

DistilBERT-base-uncased never saw ``haa``, ``bhai``, ``mast``, or
``arrey`` during pretraining — those tokens get fragmented into rare
subword pieces and the model treats them as near-noise.
XLM-RoBERTa-base was pretrained on 100 languages including Hindi, so
romanised Hinglish lands on tokens that already carry meaning.

This module exposes:

- :data:`XLMR_BASE_MODEL_NAME` — the canonical checkpoint.
- :func:`make_xlmr_config` — a config preset with the right defaults
  for XLM-R on this corpus (lower LR, longer max_length).
- :func:`per_register_metrics` — splits the val set by
  ``language_register`` and reports macro-F1 per slice, so the
  DistilBERT vs XLM-R comparison is fair on the en-IN / hi-en
  subsets specifically rather than just on the aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from sarcasm_radar.config import settings
from sarcasm_radar.evaluation.metrics import macro_f1
from sarcasm_radar.models.transformer import TransformerTrainConfig

XLMR_BASE_MODEL_NAME: str = "xlm-roberta-base"


def make_xlmr_config(
    *,
    learning_rate: float = 2e-5,
    num_epochs: int = 4,
    batch_size: int = 16,
    max_length: int = 160,
    output_dir: Path | None = None,
) -> TransformerTrainConfig:
    """XLM-RoBERTa preset.

    Tweaks compared to the DistilBERT defaults:

    - lower learning rate (2e-5 vs 5e-5) — XLM-R is more sensitive
    - one more epoch (4) — the larger model is slightly slower to
      converge on this dataset size
    - longer max_length (160) — Hinglish often runs longer per
      utterance than the English-only iSarcasm tweets
    """
    return TransformerTrainConfig(
        model_name=XLMR_BASE_MODEL_NAME,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_length=max_length,
        output_dir=output_dir or settings.models_dir / "xlmr",
    )


@dataclass(frozen=True, slots=True)
class PerRegisterScore:
    """Macro-F1 of one model on one language register."""

    register: str
    n: int
    macro_f1: float


def per_register_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    registers: pd.Series,
) -> list[PerRegisterScore]:
    """Compute macro-F1 per language register.

    The whole point of pulling XLM-R in is that it should outperform
    DistilBERT specifically on ``en-IN`` and ``hi-en`` rows. This
    helper splits the eval set by register and reports macro-F1 per
    slice so the comparison is honest.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    registers_arr = registers.reset_index(drop=True)
    if len(registers_arr) != len(y_true_arr):
        raise ValueError(
            "registers must align with y_true / y_pred; "
            f"got {len(registers_arr)} registers vs {len(y_true_arr)} labels"
        )

    rows: list[PerRegisterScore] = []
    for register in sorted(registers_arr.unique()):
        mask = (registers_arr == register).to_numpy()
        if not mask.any():
            continue
        rows.append(
            PerRegisterScore(
                register=str(register),
                n=int(mask.sum()),
                macro_f1=macro_f1(y_true_arr[mask], y_pred_arr[mask]),
            )
        )
    return rows


def compare_per_register(
    y_true: ArrayLike,
    predictions: dict[str, ArrayLike],
    registers: pd.Series,
) -> pd.DataFrame:
    """Side-by-side macro-F1 per register, one column per model.

    Example
    -------
    >>> compare_per_register(
    ...     y_test,
    ...     {"distilbert": distilbert_preds, "xlm-r": xlmr_preds},
    ...     registers=test_df["language_register"],
    ... )
    """
    frames = []
    for model_name, preds in predictions.items():
        per_reg = per_register_metrics(y_true, preds, registers)
        frame = pd.DataFrame(
            {
                "register": [r.register for r in per_reg],
                "n": [r.n for r in per_reg],
                model_name: [r.macro_f1 for r in per_reg],
            }
        )
        frames.append(frame.set_index(["register", "n"]))
    out = pd.concat(frames, axis=1).reset_index()
    return out
