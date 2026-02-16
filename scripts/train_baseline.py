"""Train the TF-IDF + Logistic Regression baseline and persist it for serving.

Downloads the SemEval-2022 Task 6 iSarcasmEval English training set
directly from its GitHub repo (no HuggingFace auth needed), fits the
baseline pipeline on the full set, and writes it to
``models/baseline.joblib``.

Unlike ``notebooks/03_train_colab.ipynb`` — which holds out 20% to
*measure* the model — this script trains on every available row,
because the artifact it produces is the one the API actually serves.

Usage
-----
    python -m scripts.train_baseline

The API's lifespan picks up ``models/baseline.joblib`` automatically
when no transformer checkpoint is present.
"""

from __future__ import annotations

import sys

import joblib
import pandas as pd

from sarcasm_radar.config import settings
from sarcasm_radar.models.baseline import train_baseline
from sarcasm_radar.utils.logging import get_logger

log = get_logger("sarcasm_radar.scripts.train_baseline")

# iSarcasmEval English train split — try both default branch names.
SEMEVAL_URLS: tuple[str, ...] = (
    "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/main/train/train.En.csv",
    "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/master/train/train.En.csv",
)


def load_corpus() -> pd.DataFrame:
    """Download iSarcasmEval and return a (text, label) frame."""
    for url in SEMEVAL_URLS:
        try:
            raw = pd.read_csv(url)
        except Exception as e:  # network or parse error — try the next URL
            log.warning("download_failed", url=url, error=str(e))
            continue
        if {"tweet", "sarcastic"}.issubset(raw.columns):
            frame = raw.dropna(subset=["tweet"]).loc[:, ["tweet", "sarcastic"]]
            frame = frame.rename(columns={"tweet": "text", "sarcastic": "label"})
            frame["text"] = frame["text"].astype(str)
            frame["label"] = frame["label"].astype(int)
            log.info("corpus_loaded", url=url, rows=len(frame))
            return frame
    raise RuntimeError("could not download iSarcasmEval from any known URL — check connectivity")


def main() -> int:
    corpus = load_corpus()

    log.info("training_baseline", rows=len(corpus))
    pipeline = train_baseline(corpus["text"], corpus["label"])

    settings.models_dir.mkdir(parents=True, exist_ok=True)
    out = settings.models_dir / "baseline.joblib"
    joblib.dump(pipeline, out)
    log.info("baseline_saved", path=str(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
