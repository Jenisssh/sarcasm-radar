"""Loaders for the merged sarcasm corpus.

Three sources end up in a single normalised frame with the schema:

    text                str
    label               int   (0 = not sarcastic, 1 = sarcastic)
    source              str   ('isarcasm' | 'semeval' | 'curated_indian_english')
    language_register   str   ('en' | 'en-IN' | 'hi-en')

The raw corpora come from HuggingFace via ``scripts/download_data.py`` and
land as parquet files in ``data/raw/``. The curated Indian English
supplement is checked into the repo under ``data/curated/`` because it's
the unique contribution of this project.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sarcasm_radar.config import settings

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
SOURCE_COLUMN = "source"
REGISTER_COLUMN = "language_register"
EXPECTED_COLUMNS: tuple[str, ...] = (
    TEXT_COLUMN,
    LABEL_COLUMN,
    SOURCE_COLUMN,
    REGISTER_COLUMN,
)


def load_isarcasm(path: Path | None = None) -> pd.DataFrame:
    """Load the iSarcasm (Oprea & Magdy 2020) corpus.

    Expects a parquet file produced by ``scripts/download_data.py``. Returns
    a normalised frame tagged with ``source='isarcasm'`` and
    ``language_register='en'``.
    """
    path = path or settings.data_raw / "isarcasm.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"iSarcasm parquet not found at {path}. Run `make data` first."
        )
    return normalize(pd.read_parquet(path), source="isarcasm", register="en")


def load_semeval_isarcasm(path: Path | None = None) -> pd.DataFrame:
    """Load the SemEval-2022 Task 6 iSarcasmEval English partition."""
    path = path or settings.data_raw / "semeval_isarcasm.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"SemEval iSarcasm parquet not found at {path}. Run `make data` first."
        )
    return normalize(pd.read_parquet(path), source="semeval", register="en")


def load_curated_supplement(path: Path | None = None) -> pd.DataFrame:
    """Load the hand-curated Indian English supplement.

    When the CSV doesn't exist yet, returns an empty frame with the correct
    schema rather than raising — early in the project the curation pipeline
    may not have any labels yet.
    """
    path = path or settings.data_curated / "indian_english.csv"
    if not path.exists():
        return pd.DataFrame(
            {
                TEXT_COLUMN: pd.Series(dtype="object"),
                LABEL_COLUMN: pd.Series(dtype="int64"),
                SOURCE_COLUMN: pd.Series(dtype="object"),
                REGISTER_COLUMN: pd.Series(dtype="object"),
            }
        )
    return normalize(
        pd.read_csv(path),
        source="curated_indian_english",
        register="en-IN",
    )


def normalize(df: pd.DataFrame, *, source: str, register: str) -> pd.DataFrame:
    """Coerce an upstream frame to the project schema.

    Public corpora use varying column names for the text and label columns
    (``tweet``, ``content``, ``sentence`` for text; ``sarcastic``,
    ``is_sarcastic``, ``target`` for the label). This function picks the
    first match it finds and renames.
    """
    text_candidates = (TEXT_COLUMN, "tweet", "content", "sentence")
    label_candidates = (LABEL_COLUMN, "sarcastic", "is_sarcastic", "target")

    text_col = _first_present(df, text_candidates)
    label_col = _first_present(df, label_candidates)
    if text_col is None:
        raise KeyError(
            f"No text column found in {list(df.columns)}; "
            f"expected one of {text_candidates}"
        )
    if label_col is None:
        raise KeyError(
            f"No label column found in {list(df.columns)}; "
            f"expected one of {label_candidates}"
        )

    out = pd.DataFrame(
        {
            TEXT_COLUMN: df[text_col].astype(str),
            LABEL_COLUMN: df[label_col].astype(int),
        }
    )
    out[SOURCE_COLUMN] = source
    out[REGISTER_COLUMN] = register
    return out


def merge_corpora(*frames: pd.DataFrame, deduplicate: bool = True) -> pd.DataFrame:
    """Concatenate normalised frames; optionally drop duplicate texts.

    Duplicates are matched on the exact ``text`` string. When the same text
    appears in multiple sources, the first one wins (so put the most
    trustworthy frame first).
    """
    if not frames:
        return pd.DataFrame(
            {
                TEXT_COLUMN: pd.Series(dtype="object"),
                LABEL_COLUMN: pd.Series(dtype="int64"),
                SOURCE_COLUMN: pd.Series(dtype="object"),
                REGISTER_COLUMN: pd.Series(dtype="object"),
            }
        )
    merged = pd.concat(frames, ignore_index=True)
    if deduplicate:
        merged = merged.drop_duplicates(subset=[TEXT_COLUMN], keep="first").reset_index(
            drop=True
        )
    return merged


def _first_present(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None
