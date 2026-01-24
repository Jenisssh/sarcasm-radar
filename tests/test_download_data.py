"""Tests for the HuggingFace download script.

We don't hit HuggingFace in tests — the network paths are mocked. What's
worth testing is the file-already-present skip behaviour and the
argparse surface, which are pure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
from scripts.download_data import (
    DEFAULT_ISARCASM_ID,
    DEFAULT_SEMEVAL_ID,
    download_isarcasm,
    download_semeval,
    main,
)


class _FakeDataset:
    """Stand-in for ``datasets.Dataset`` — enough surface to be saved as parquet."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df


def test_default_dataset_ids_are_strings() -> None:
    assert isinstance(DEFAULT_ISARCASM_ID, str)
    assert isinstance(DEFAULT_SEMEVAL_ID, str)
    assert "/" in DEFAULT_ISARCASM_ID  # huggingface owner/name pattern
    assert "/" in DEFAULT_SEMEVAL_ID


def test_isarcasm_skips_when_file_already_present(tmp_path: Path) -> None:
    out = tmp_path / "isarcasm.parquet"
    pd.DataFrame({"text": ["x"], "label": [0]}).to_parquet(out)
    # Should return without invoking the loader
    with patch("scripts.download_data._load_split") as mock:
        result = download_isarcasm(tmp_path)
    assert result == out
    mock.assert_not_called()


def test_semeval_skips_when_file_already_present(tmp_path: Path) -> None:
    out = tmp_path / "semeval_isarcasm.parquet"
    pd.DataFrame({"text": ["x"], "label": [1]}).to_parquet(out)
    with patch("scripts.download_data._load_split") as mock:
        result = download_semeval(tmp_path)
    assert result == out
    mock.assert_not_called()


def test_isarcasm_writes_parquet_on_first_run(tmp_path: Path) -> None:
    fake_df = pd.DataFrame({"tweet": ["a", "b"], "sarcastic": [0, 1]})
    with patch("scripts.download_data._load_split", return_value=_FakeDataset(fake_df)):
        out = download_isarcasm(tmp_path)
    assert out.exists()
    loaded = pd.read_parquet(out)
    assert len(loaded) == 2


def test_force_redownload_overwrites(tmp_path: Path) -> None:
    out = tmp_path / "isarcasm.parquet"
    pd.DataFrame({"text": ["old"], "label": [0]}).to_parquet(out)

    fresh = pd.DataFrame({"tweet": ["new"], "sarcastic": [1]})
    with patch("scripts.download_data._load_split", return_value=_FakeDataset(fresh)):
        download_isarcasm(tmp_path, force=True)

    loaded = pd.read_parquet(out)
    assert list(loaded["tweet"]) == ["new"]


def test_main_returns_zero_on_skip_all(tmp_path: Path, monkeypatch: Any) -> None:
    # If both skip flags are set, main returns 0 without hitting HF
    from sarcasm_radar import config

    monkeypatch.setattr(config.settings, "data_raw", tmp_path)
    with patch("scripts.download_data._load_split") as mock:
        rc = main(["--skip-isarcasm", "--skip-semeval"])
    assert rc == 0
    mock.assert_not_called()


def test_main_returns_1_on_missing_datasets_dep(tmp_path: Path, monkeypatch: Any) -> None:
    from sarcasm_radar import config

    monkeypatch.setattr(config.settings, "data_raw", tmp_path)
    with patch(
        "scripts.download_data._load_split",
        side_effect=RuntimeError("huggingface 'datasets' is required"),
    ):
        rc = main([])
    assert rc == 1
