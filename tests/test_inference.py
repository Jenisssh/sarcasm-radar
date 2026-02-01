"""Tests for sarcasm_radar.models.inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from sarcasm_radar.models.inference import (
    ModelArtifacts,
    PredictionResult,
    load_artifacts,
    predict_batch,
    predict_single,
)


class _FakeBaselinePipeline:
    """sklearn-Pipeline-shaped mock with controllable output probability."""

    def __init__(self, prob_sarcastic: float) -> None:
        self.prob = prob_sarcastic

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        return np.column_stack([np.full(n, 1.0 - self.prob), np.full(n, self.prob)])


class _FakeTransformerClassifier:
    """TransformerSarcasmClassifier-shaped mock."""

    def __init__(self, prob_sarcastic: float) -> None:
        self.prob = prob_sarcastic

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.prob), np.full(n, self.prob)])


class TestPredictionResult:
    def test_is_frozen(self) -> None:
        r = PredictionResult(text="x", probability=0.5, decision="SARCASTIC", model_version="v1")
        with pytest.raises((AttributeError, Exception)):
            r.probability = 0.9  # type: ignore[misc]

    def test_as_dict_round_trips_through_json(self) -> None:
        import json

        r = PredictionResult(
            text="haa beta", probability=0.92, decision="SARCASTIC", model_version="v1"
        )
        d = r.as_dict()
        # JSON-friendly
        loaded = json.loads(json.dumps(d))
        assert loaded["text"] == "haa beta"
        assert loaded["probability"] == 0.92


class TestModelArtifacts:
    def test_is_transformer_true_for_transformer_kind(self) -> None:
        artifacts = ModelArtifacts(
            classifier=None, kind="transformer", model_version="v1", threshold=0.5
        )
        assert artifacts.is_transformer is True

    def test_is_transformer_false_for_baseline_kind(self) -> None:
        artifacts = ModelArtifacts(
            classifier=None, kind="baseline", model_version="v1", threshold=0.5
        )
        assert artifacts.is_transformer is False

    def test_is_frozen(self) -> None:
        artifacts = ModelArtifacts(
            classifier=None, kind="baseline", model_version="v1", threshold=0.5
        )
        with pytest.raises((AttributeError, Exception)):
            artifacts.threshold = 0.9  # type: ignore[misc]


class TestLoadArtifacts:
    def test_baseline_loads_joblib(self, tmp_path: Path) -> None:
        clf = _FakeBaselinePipeline(prob_sarcastic=0.8)
        path = tmp_path / "baseline.joblib"
        joblib.dump(clf, path)

        artifacts = load_artifacts(path, kind="baseline")
        assert artifacts.kind == "baseline"
        # The loaded object retains the original probability
        loaded_proba = artifacts.classifier.predict_proba(["x"])
        assert loaded_proba[0, 1] == pytest.approx(0.8)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_artifacts(tmp_path / "does-not-exist.joblib", kind="baseline")

    def test_default_threshold_is_half(self, tmp_path: Path) -> None:
        clf = _FakeBaselinePipeline(prob_sarcastic=0.5)
        path = tmp_path / "baseline.joblib"
        joblib.dump(clf, path)
        artifacts = load_artifacts(path, kind="baseline")
        assert artifacts.threshold == 0.5

    def test_threshold_can_be_overridden(self, tmp_path: Path) -> None:
        clf = _FakeBaselinePipeline(prob_sarcastic=0.5)
        path = tmp_path / "baseline.joblib"
        joblib.dump(clf, path)
        artifacts = load_artifacts(path, kind="baseline", threshold=0.7)
        assert artifacts.threshold == 0.7


class TestPredictSingle:
    def test_returns_prediction_result(self) -> None:
        artifacts = ModelArtifacts(
            classifier=_FakeBaselinePipeline(prob_sarcastic=0.8),
            kind="baseline",
            model_version="v1",
            threshold=0.5,
        )
        result = predict_single("haa beta nice plan", artifacts)
        assert isinstance(result, PredictionResult)
        assert result.text == "haa beta nice plan"
        assert result.probability == pytest.approx(0.8)
        assert result.decision == "SARCASTIC"
        assert result.model_version == "v1"

    def test_decision_flips_at_threshold(self) -> None:
        below = ModelArtifacts(
            classifier=_FakeBaselinePipeline(prob_sarcastic=0.3),
            kind="baseline",
            model_version="v1",
            threshold=0.5,
        )
        above = ModelArtifacts(
            classifier=_FakeBaselinePipeline(prob_sarcastic=0.6),
            kind="baseline",
            model_version="v1",
            threshold=0.5,
        )
        assert predict_single("x", below).decision == "NOT_SARCASTIC"
        assert predict_single("x", above).decision == "SARCASTIC"

    def test_decision_uses_custom_threshold(self) -> None:
        artifacts = ModelArtifacts(
            classifier=_FakeBaselinePipeline(prob_sarcastic=0.4),
            kind="baseline",
            model_version="v1",
            threshold=0.3,
        )
        # 0.4 >= 0.3 → SARCASTIC, even though it's below 0.5
        assert predict_single("x", artifacts).decision == "SARCASTIC"

    def test_empty_text_raises(self) -> None:
        artifacts = ModelArtifacts(
            classifier=_FakeBaselinePipeline(prob_sarcastic=0.5),
            kind="baseline",
            model_version="v1",
            threshold=0.5,
        )
        with pytest.raises(ValueError, match="non-empty"):
            predict_single("", artifacts)

    def test_whitespace_only_raises(self) -> None:
        artifacts = ModelArtifacts(
            classifier=_FakeBaselinePipeline(prob_sarcastic=0.5),
            kind="baseline",
            model_version="v1",
            threshold=0.5,
        )
        with pytest.raises(ValueError, match="non-empty"):
            predict_single("   \t\n  ", artifacts)

    def test_transformer_kind_routes_correctly(self) -> None:
        artifacts = ModelArtifacts(
            classifier=_FakeTransformerClassifier(prob_sarcastic=0.9),
            kind="transformer",
            model_version="distilbert-v1",
            threshold=0.5,
        )
        result = predict_single("test", artifacts)
        assert result.probability == pytest.approx(0.9)
        assert result.decision == "SARCASTIC"


class TestPredictBatch:
    def test_empty_list_returns_empty(self) -> None:
        artifacts = ModelArtifacts(
            classifier=_FakeBaselinePipeline(prob_sarcastic=0.5),
            kind="baseline",
            model_version="v1",
            threshold=0.5,
        )
        assert predict_batch([], artifacts) == []

    def test_returns_one_result_per_input(self) -> None:
        artifacts = ModelArtifacts(
            classifier=_FakeBaselinePipeline(prob_sarcastic=0.7),
            kind="baseline",
            model_version="v1",
            threshold=0.5,
        )
        results = predict_batch(["a", "b", "c"], artifacts)
        assert len(results) == 3
        assert all(r.decision == "SARCASTIC" for r in results)
