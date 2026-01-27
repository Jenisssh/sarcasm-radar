"""Tests for sarcasm_radar.data.curate and the seed labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sarcasm_radar.data.curate import (
    CURATION_COLUMNS,
    VALID_LABELS,
    VALID_REGISTERS,
    append_label,
    load_curated_labels,
    summarize_curation,
    validate_curation_row,
)


class TestValidateCurationRow:
    def test_accepts_valid_row(self) -> None:
        # Should not raise
        validate_curation_row(
            text="haa beta nice plan",
            label=1,
            register="hi-en",
            rationale="ironic praise",
        )

    def test_rejects_empty_text(self) -> None:
        with pytest.raises(ValueError, match="text must be"):
            validate_curation_row(text="", label=1, register="en-IN", rationale="x")

    def test_rejects_whitespace_only_text(self) -> None:
        with pytest.raises(ValueError, match="text must be"):
            validate_curation_row(text="   \t  ", label=0, register="en-IN", rationale="x")

    def test_rejects_bad_label(self) -> None:
        with pytest.raises(ValueError, match="label must be"):
            validate_curation_row(text="x", label=2, register="en-IN", rationale="r")

    def test_rejects_bad_register(self) -> None:
        with pytest.raises(ValueError, match="register must be"):
            validate_curation_row(
                text="x",
                label=0,
                register="fr-FR",
                rationale="r",  # type: ignore[arg-type]
            )

    def test_rejects_empty_rationale(self) -> None:
        # Rationale is enforced for quality control
        with pytest.raises(ValueError, match="rationale"):
            validate_curation_row(text="x", label=0, register="en-IN", rationale="")


class TestAppendLabel:
    def test_creates_file_with_header_on_first_call(self, tmp_path: Path) -> None:
        path = tmp_path / "new.csv"
        out = append_label(
            "first row",
            label=1,
            register="en-IN",
            rationale="r",
            path=path,
        )
        assert out == path
        df = pd.read_csv(path)
        assert list(df.columns) == list(CURATION_COLUMNS)
        assert len(df) == 1
        assert df.loc[0, "text"] == "first row"

    def test_appends_to_existing_file_without_duplicating_header(self, tmp_path: Path) -> None:
        path = tmp_path / "x.csv"
        append_label("row 1", 1, "en-IN", "r1", path=path)
        append_label("row 2", 0, "hi-en", "r2", path=path)
        df = pd.read_csv(path)
        assert len(df) == 2
        # Confirm row order preserved
        assert df.loc[0, "text"] == "row 1"
        assert df.loc[1, "text"] == "row 2"

    def test_validation_failure_does_not_write(self, tmp_path: Path) -> None:
        path = tmp_path / "x.csv"
        with pytest.raises(ValueError):
            append_label("", 1, "en-IN", "r", path=path)
        # File must not have been created
        assert not path.exists()


class TestSummarizeCuration:
    def test_basic_counts(self) -> None:
        df = pd.DataFrame(
            {
                "text": ["a", "b", "c", "d"],
                "label": [1, 1, 0, 0],
                "register": ["en-IN", "hi-en", "en-IN", "en-IN"],
                "rationale": ["x" * 10, "y" * 20, "z" * 5, "w" * 15],
            }
        )
        summary = summarize_curation(df)
        assert summary["total"] == 4
        assert summary["n_sarcastic"] == 2
        assert summary["n_not_sarcastic"] == 2
        assert summary["pct_sarcastic"] == 50.0
        assert summary["by_register"] == {"en-IN": 3, "hi-en": 1}

    def test_empty_dataframe_summary(self) -> None:
        df = pd.DataFrame({col: pd.Series(dtype="object") for col in CURATION_COLUMNS})
        df["label"] = df["label"].astype("int64")
        summary = summarize_curation(df)
        assert summary["total"] == 0
        assert summary["pct_sarcastic"] == 0.0


class TestSeedFile:
    """Tests that the committed seed file passes its own schema."""

    def test_seed_file_loads(self) -> None:
        # Uses default path (data/curated/indian_english.csv)
        df = load_curated_labels()
        assert len(df) == 50

    def test_seed_file_is_balanced(self) -> None:
        df = load_curated_labels()
        counts = df["label"].value_counts()
        assert counts[0] == 25
        assert counts[1] == 25

    def test_seed_file_labels_are_valid(self) -> None:
        df = load_curated_labels()
        assert set(df["label"].unique()) <= VALID_LABELS

    def test_seed_file_registers_are_valid(self) -> None:
        df = load_curated_labels()
        assert set(df["register"].unique()) <= VALID_REGISTERS

    def test_seed_file_every_row_has_rationale(self) -> None:
        df = load_curated_labels()
        assert df["rationale"].str.strip().str.len().min() > 0

    def test_seed_file_includes_both_registers(self) -> None:
        df = load_curated_labels()
        regs = set(df["register"].unique())
        # The Hinglish examples are the differentiator — we want both present
        assert "hi-en" in regs
        assert "en-IN" in regs


class TestLoadCuratedLabels:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="curated supplement"):
            load_curated_labels(tmp_path / "absent.csv")

    def test_bad_label_value_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.csv"
        path.write_text("text,label,register,rationale\nfoo,2,en-IN,r\n")
        with pytest.raises(ValueError, match="invalid labels"):
            load_curated_labels(path)

    def test_extra_column_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "extra.csv"
        path.write_text("text,label,register,rationale,extra\nfoo,1,en-IN,r,oops\n")
        with pytest.raises(ValueError, match="unexpected columns"):
            load_curated_labels(path)
