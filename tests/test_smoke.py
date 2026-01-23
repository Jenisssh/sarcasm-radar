"""Smoke tests — verify the package imports and the basic fixture works."""

from __future__ import annotations

import pandas as pd

import sarcasm_radar
from sarcasm_radar.config import settings


def test_version_is_set() -> None:
    assert sarcasm_radar.__version__ == "0.1.0"


def test_settings_has_project_root_with_pyproject() -> None:
    assert (settings.project_root / "pyproject.toml").is_file()


def test_settings_seed_is_42() -> None:
    assert settings.random_seed == 42


def test_tiny_sarcasm_df_shape(tiny_sarcasm_df: pd.DataFrame) -> None:
    assert tiny_sarcasm_df.shape == (20, 2)
    assert set(tiny_sarcasm_df.columns) == {"text", "label"}
    assert tiny_sarcasm_df["label"].isin([0, 1]).all()


def test_tiny_sarcasm_df_is_balanced(tiny_sarcasm_df: pd.DataFrame) -> None:
    counts = tiny_sarcasm_df["label"].value_counts()
    assert counts[0] == counts[1] == 10
