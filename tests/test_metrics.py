"""Tests for sarcasm_radar.evaluation.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from sarcasm_radar.evaluation.metrics import (
    ClassificationReport,
    PerClassMetrics,
    evaluate,
    macro_f1,
    per_class_metrics,
)


class TestMacroF1:
    def test_perfect_predictions_score_one(self) -> None:
        y = np.array([0, 1, 0, 1, 1, 0])
        assert macro_f1(y, y) == pytest.approx(1.0)

    def test_all_wrong_predictions_score_zero(self) -> None:
        y = np.array([0, 1, 0, 1])
        y_hat = 1 - y
        assert macro_f1(y, y_hat) == 0.0

    def test_returns_python_float(self) -> None:
        y = np.array([0, 1, 0, 1])
        assert isinstance(macro_f1(y, y), float)


class TestPerClassMetrics:
    def test_returns_two_perclass_objects(self) -> None:
        y = np.array([0, 0, 1, 1])
        y_hat = np.array([0, 1, 1, 1])
        a, b = per_class_metrics(y, y_hat)
        assert isinstance(a, PerClassMetrics)
        assert isinstance(b, PerClassMetrics)

    def test_support_counts_each_class(self) -> None:
        y = np.array([0, 0, 0, 1, 1])
        y_hat = np.array([0, 0, 0, 1, 1])
        not_sarc, sarc = per_class_metrics(y, y_hat)
        assert not_sarc.support == 3
        assert sarc.support == 2

    def test_perfect_per_class(self) -> None:
        y = np.array([0, 1, 0, 1])
        not_sarc, sarc = per_class_metrics(y, y)
        assert not_sarc.precision == 1.0
        assert sarc.recall == 1.0
        assert sarc.f1 == 1.0


class TestEvaluate:
    def test_returns_classification_report(self) -> None:
        y = np.array([0, 1, 0, 1, 1, 0])
        report = evaluate(y, y)
        assert isinstance(report, ClassificationReport)

    def test_perfect_predictions_full_scores(self) -> None:
        y = np.array([0, 1, 0, 1, 1, 0])
        report = evaluate(y, y)
        assert report.accuracy == 1.0
        assert report.macro_f1 == pytest.approx(1.0)
        assert report.not_sarcastic.f1 == pytest.approx(1.0)
        assert report.sarcastic.f1 == pytest.approx(1.0)

    def test_confusion_matrix_shape(self) -> None:
        y = np.array([0, 1])
        y_hat = np.array([1, 0])
        report = evaluate(y, y_hat)
        # 2x2 matrix
        assert len(report.confusion) == 2
        assert all(len(row) == 2 for row in report.confusion)
        # Both predictions wrong: TN=0, FP=1, FN=1, TP=0
        assert report.confusion == [[0, 1], [1, 0]]

    def test_as_dict_returns_serialisable(self) -> None:
        y = np.array([0, 1, 1, 0])
        d = evaluate(y, y).as_dict()
        # All leaf values must be JSON-friendly (no numpy types)
        import json

        json.dumps(d)  # raises if anything isn't serialisable

    def test_immutable_report(self) -> None:
        y = np.array([0, 1])
        report = evaluate(y, y)
        with pytest.raises((AttributeError, Exception)):
            report.accuracy = 0.0  # type: ignore[misc]
