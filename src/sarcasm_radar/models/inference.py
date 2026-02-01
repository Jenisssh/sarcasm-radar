"""Unified inference wrapper over the baseline + transformer models.

Both the FastAPI endpoint and the Streamlit demo need to "score a
piece of text and return a prediction", and neither of them should
care which model produced the answer. This module is the seam.

The wrapper accepts any artifact bundle produced by either training
path:

- Baseline: ``models/baseline.joblib`` — a fitted sklearn ``Pipeline``
- Transformer: ``models/distilbert/`` or ``models/xlmr/`` — a HuggingFace
  ``save_pretrained`` directory

``predict_single`` returns a frozen :class:`PredictionResult` regardless
of which path was taken, so downstream code stays uniform.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import joblib

from sarcasm_radar.config import settings

ModelKind = Literal["baseline", "transformer"]
Decision = Literal["SARCASTIC", "NOT_SARCASTIC"]


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """One prediction, JSON-friendly."""

    text: str
    probability: float
    decision: Decision
    model_version: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ModelArtifacts:
    """Bundle the inference layer needs at request time."""

    classifier: Any
    kind: ModelKind
    model_version: str
    threshold: float

    @property
    def is_transformer(self) -> bool:
        return self.kind == "transformer"


def load_artifacts(
    path: Path | None = None,
    *,
    kind: ModelKind = "transformer",
    model_version: str = "v0.1.0",
    threshold: float = 0.5,
) -> ModelArtifacts:
    """Load a saved model from disk.

    For ``kind='baseline'`` the path points at a joblib file. For
    ``kind='transformer'`` it points at a HuggingFace checkpoint
    directory.
    """
    if path is None:
        path = (
            settings.models_dir / "baseline.joblib"
            if kind == "baseline"
            else settings.models_dir / "distilbert"
        )

    if not path.exists():
        raise FileNotFoundError(f"model artifact not found at {path}")

    if kind == "baseline":
        classifier = joblib.load(path)
    else:
        from sarcasm_radar.models.transformer import TransformerSarcasmClassifier

        classifier = TransformerSarcasmClassifier.load(path)

    return ModelArtifacts(
        classifier=classifier,
        kind=kind,
        model_version=model_version,
        threshold=threshold,
    )


def predict_single(text: str, artifacts: ModelArtifacts) -> PredictionResult:
    """Score one text and wrap the result."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    probability = _score_one(text, artifacts)
    decision: Decision = "SARCASTIC" if probability >= artifacts.threshold else "NOT_SARCASTIC"
    return PredictionResult(
        text=text,
        probability=float(probability),
        decision=decision,
        model_version=artifacts.model_version,
    )


def predict_batch(texts: list[str], artifacts: ModelArtifacts) -> list[PredictionResult]:
    """Score many texts in one call.

    Falls back to repeated single calls — the baseline pipeline already
    vectorises internally, and HuggingFace's tokeniser batches at the
    Series level, so we only pay the per-call overhead, not the per-row
    compute cost.
    """
    if not texts:
        return []
    return [predict_single(t, artifacts) for t in texts]


def _score_one(text: str, artifacts: ModelArtifacts) -> float:
    """Backend-specific scoring → P(sarcastic)."""
    if artifacts.is_transformer:
        import pandas as pd

        proba = artifacts.classifier.predict_proba(pd.Series([text]))
        return float(proba[0, 1])

    # baseline (sklearn Pipeline)
    proba = artifacts.classifier.predict_proba([text])
    return float(proba[0, 1])
