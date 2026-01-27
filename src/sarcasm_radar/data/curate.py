"""Curation pipeline for the Indian English sarcasm supplement.

This is the distinctive piece of the project. Most public sarcasm corpora
are Western Reddit / Twitter; this module owns the process for adding
hand-labeled Indian English examples on top.

Each curated row needs four fields:

    text                str   the cleaned tweet/utterance
    label               int   0 = not sarcastic, 1 = sarcastic
    register            str   'en-IN' for Indian English, 'hi-en' for Hinglish
    rationale           str   short note on *why* this is (not) sarcastic,
                              kept for quality control and inter-annotator
                              comparison later

The seed file ``data/curated/indian_english.csv`` is checked into the
repo. The functions here support:

- loading the file with schema validation
- appending new rows from a CLI / notebook without breaking the schema
- a summary helper for class balance, register mix, and rationale coverage
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from sarcasm_radar.config import settings

Register = Literal["en-IN", "hi-en"]

CURATION_COLUMNS: tuple[str, ...] = ("text", "label", "register", "rationale")
VALID_LABELS: frozenset[int] = frozenset({0, 1})
VALID_REGISTERS: frozenset[str] = frozenset({"en-IN", "hi-en"})


def load_curated_labels(path: Path | None = None) -> pd.DataFrame:
    """Load the curated supplement with schema validation.

    Raises ``FileNotFoundError`` if the CSV doesn't exist; raises
    ``ValueError`` if the schema is malformed.
    """
    path = path or settings.data_curated / "indian_english.csv"
    if not path.exists():
        raise FileNotFoundError(f"curated supplement not found at {path}")

    df = pd.read_csv(path)
    _validate_dataframe(df, source=str(path))
    return df


def validate_curation_row(
    text: str,
    label: int,
    register: Register,
    rationale: str,
) -> None:
    """Sanity-check a single proposed row. Raises ``ValueError`` on bad input."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if label not in VALID_LABELS:
        raise ValueError(f"label must be 0 or 1; got {label!r}")
    if register not in VALID_REGISTERS:
        raise ValueError(
            f"register must be one of {sorted(VALID_REGISTERS)}; got {register!r}"
        )
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("rationale must be a non-empty string (quality control)")


def append_label(
    text: str,
    label: int,
    register: Register,
    rationale: str,
    *,
    path: Path | None = None,
) -> Path:
    """Append a single validated row to the curated CSV.

    Creates the file with the header if it doesn't exist yet. Raises
    ``ValueError`` if the row fails validation (so the CSV never gets a
    half-bad row appended).
    """
    validate_curation_row(text, label, register, rationale)
    path = path or settings.data_curated / "indian_english.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(CURATION_COLUMNS)
        writer.writerow([text, label, register, rationale])
    return path


def summarize_curation(df: pd.DataFrame) -> dict[str, Any]:
    """Compute a quick summary of the curated supplement.

    Useful as a Make target / notebook cell to track how much progress
    has been made on the Indian English labelling pass.
    """
    _validate_dataframe(df, source="<in-memory>")
    total = len(df)
    n_sarcastic = int((df["label"] == 1).sum())
    by_register = df["register"].value_counts().to_dict()
    avg_rationale_len = float(df["rationale"].str.len().mean()) if total else 0.0
    return {
        "total": total,
        "n_sarcastic": n_sarcastic,
        "n_not_sarcastic": total - n_sarcastic,
        "pct_sarcastic": round(n_sarcastic / total * 100, 2) if total else 0.0,
        "by_register": by_register,
        "avg_rationale_length_chars": round(avg_rationale_len, 1),
    }


def _validate_dataframe(df: pd.DataFrame, *, source: str) -> None:
    missing = set(CURATION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    extras = set(df.columns) - set(CURATION_COLUMNS)
    if extras:
        raise ValueError(f"{source} has unexpected columns: {sorted(extras)}")
    bad_labels = set(df["label"].unique()) - VALID_LABELS
    if bad_labels:
        raise ValueError(f"{source} has invalid labels: {sorted(bad_labels)}")
    bad_registers = set(df["register"].unique()) - VALID_REGISTERS
    if bad_registers:
        raise ValueError(f"{source} has invalid registers: {sorted(bad_registers)}")
