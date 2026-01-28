"""TF-IDF + Logistic Regression baseline.

The role of this model is to *fail well*. Any transformer-based approach
has to beat this on the same evaluation harness; if it doesn't, the
extra complexity isn't earning its keep.

Pipeline steps:

1. Text cleaning via :func:`sarcasm_radar.data.clean.clean_text` (URLs,
   mentions, hashtags, repeated chars, HTML, whitespace).
2. :class:`sklearn.feature_extraction.text.TfidfVectorizer` with 1-2 ngrams
   so short ironic phrasings ('yeah right', 'kya baat') survive.
3. :class:`sklearn.linear_model.LogisticRegression` with
   ``class_weight='balanced'``.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sarcasm_radar.config import settings
from sarcasm_radar.data.clean import clean_text

ClassWeight = Literal["balanced"] | dict[int, float] | None


def build_baseline_pipeline(
    *,
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    class_weight: ClassWeight = "balanced",
    C: float = 1.0,
    max_iter: int = 1_000,
    random_state: int | None = None,
) -> Pipeline:
    """Construct the unfit baseline pipeline.

    Parameters
    ----------
    max_features:
        Vocabulary cap for the TF-IDF vectorizer.
    ngram_range:
        ``(1, 2)`` captures both unigrams and bigrams — bigrams matter for
        ironic collocations like ``yeah right`` or ``kya baat``.
    min_df:
        Drop terms appearing in fewer than this many docs.
    class_weight, C, max_iter:
        Forwarded to ``LogisticRegression``.
    random_state:
        Defaults to ``settings.random_seed``.
    """
    seed = settings.random_seed if random_state is None else random_state
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    max_features=max_features,
                    ngram_range=ngram_range,
                    min_df=min_df,
                    sublinear_tf=True,
                    strip_accents=None,  # preserve Hindi script
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    class_weight=class_weight,
                    C=C,
                    max_iter=max_iter,
                    random_state=seed,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_baseline(
    X_train: pd.Series,
    y_train: pd.Series,
    **pipeline_kwargs: Any,
) -> Pipeline:
    """Fit the baseline pipeline and return it."""
    pipe = build_baseline_pipeline(**pipeline_kwargs)
    pipe.fit(X_train, y_train)
    return pipe
