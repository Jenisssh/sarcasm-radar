"""Tests for sarcasm_radar.models.baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sarcasm_radar.evaluation.metrics import macro_f1
from sarcasm_radar.models.baseline import build_baseline_pipeline, train_baseline


@pytest.fixture
def predictable_corpus() -> tuple[pd.Series, pd.Series]:
    """Sarcastic class consistently uses 'great', 'amazing', 'love' (ironic).

    Non-sarcastic uses neutral / sincere phrasing. A model with any signal
    should easily separate these — gives us a meaningful 'beats chance'
    assertion below.
    """
    rng = np.random.default_rng(0)
    sarcastic_seeds = [
        "great another monday",
        "amazing service so far",
        "love how slow this is",
        "great timing as always",
        "obviously the wifi works fine",
        "wow such impressive results",
        "yeah right like that will work",
        "amazing how fast this is",
        "great way to start a week",
        "love these delays",
    ]
    sincere_seeds = [
        "the report is ready for review",
        "meeting is at three pm",
        "submitted the assignment today",
        "having coffee with a friend",
        "the new bridge is open",
        "finished my run feeling fine",
        "weather is pleasant today",
        "got the books from the library",
        "team lunch tomorrow at noon",
        "presentation went well",
    ]
    n_each = 50
    sarc = rng.choice(sarcastic_seeds, size=n_each).tolist()
    sinc = rng.choice(sincere_seeds, size=n_each).tolist()
    X = pd.Series(sarc + sinc)
    y = pd.Series([1] * n_each + [0] * n_each)
    return X, y


class TestBuildBaselinePipeline:
    def test_returns_sklearn_pipeline(self) -> None:
        pipe = build_baseline_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_has_expected_steps(self) -> None:
        pipe = build_baseline_pipeline()
        names = [n for n, _ in pipe.steps]
        assert names == ["tfidf", "clf"]

    def test_tfidf_is_tfidfvectorizer(self) -> None:
        pipe = build_baseline_pipeline()
        assert isinstance(pipe.named_steps["tfidf"], TfidfVectorizer)

    def test_clf_is_logistic_regression(self) -> None:
        pipe = build_baseline_pipeline()
        assert isinstance(pipe.named_steps["clf"], LogisticRegression)

    def test_default_uses_balanced_class_weight(self) -> None:
        pipe = build_baseline_pipeline()
        assert pipe.named_steps["clf"].class_weight == "balanced"

    def test_default_ngrams_include_bigrams(self) -> None:
        pipe = build_baseline_pipeline()
        assert pipe.named_steps["tfidf"].ngram_range == (1, 2)

    def test_strip_accents_disabled_for_hindi_script(self) -> None:
        pipe = build_baseline_pipeline()
        assert pipe.named_steps["tfidf"].strip_accents is None

    def test_ngram_range_can_be_overridden(self) -> None:
        pipe = build_baseline_pipeline(ngram_range=(1, 3))
        assert pipe.named_steps["tfidf"].ngram_range == (1, 3)

    def test_class_weight_can_be_disabled(self) -> None:
        pipe = build_baseline_pipeline(class_weight=None)
        assert pipe.named_steps["clf"].class_weight is None


class TestTrainBaseline:
    def test_fit_returns_pipeline(self, predictable_corpus: tuple[pd.Series, pd.Series]) -> None:
        X, y = predictable_corpus
        pipe = train_baseline(X, y)
        # Confirm fitted by predicting without raising
        preds = pipe.predict(X.head(5))
        assert len(preds) == 5

    def test_predicts_in_valid_label_space(
        self, predictable_corpus: tuple[pd.Series, pd.Series]
    ) -> None:
        X, y = predictable_corpus
        pipe = train_baseline(X, y)
        preds = pipe.predict(X)
        assert set(preds.tolist()) <= {0, 1}

    def test_predict_proba_in_unit_interval(
        self, predictable_corpus: tuple[pd.Series, pd.Series]
    ) -> None:
        X, y = predictable_corpus
        pipe = train_baseline(X, y)
        proba = pipe.predict_proba(X)
        assert (proba >= 0).all()
        assert (proba <= 1).all()
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_beats_random_on_separable_corpus(
        self, predictable_corpus: tuple[pd.Series, pd.Series]
    ) -> None:
        X, y = predictable_corpus
        pipe = train_baseline(X, y)
        preds = pipe.predict(X)
        # On a separable corpus, even the baseline should hit very high F1
        assert macro_f1(y, preds) > 0.9

    def test_handles_hinglish_input(self) -> None:
        # Make sure the cleaner / vectoriser don't choke on Devanagari /
        # romanised Hindi. Use min_df=1 since the toy corpus is tiny.
        X = pd.Series(
            [
                "haa beta nice plan",
                "mast yaar bahut accha",
                "the weather is pleasant",
                "submitted my report today",
                "great another monday",
                "amazing service so far",
            ]
        )
        y = pd.Series([1, 1, 0, 0, 1, 1])
        pipe = train_baseline(X, y, min_df=1)
        proba = pipe.predict_proba(X)
        assert proba.shape == (6, 2)
