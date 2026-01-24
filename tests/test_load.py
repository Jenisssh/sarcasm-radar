"""Tests for sarcasm_radar.data.load."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sarcasm_radar.data.load import (
    EXPECTED_COLUMNS,
    LABEL_COLUMN,
    REGISTER_COLUMN,
    SOURCE_COLUMN,
    TEXT_COLUMN,
    load_curated_supplement,
    load_isarcasm,
    load_semeval_isarcasm,
    merge_corpora,
    normalize,
)


class TestNormalize:
    def test_accepts_canonical_columns(self) -> None:
        df = pd.DataFrame({"text": ["a", "b"], "label": [0, 1]})
        out = normalize(df, source="x", register="en")
        assert list(out.columns) == list(EXPECTED_COLUMNS)
        assert list(out[TEXT_COLUMN]) == ["a", "b"]
        assert list(out[LABEL_COLUMN]) == [0, 1]
        assert (out[SOURCE_COLUMN] == "x").all()
        assert (out[REGISTER_COLUMN] == "en").all()

    def test_renames_tweet_to_text(self) -> None:
        df = pd.DataFrame({"tweet": ["hello"], "sarcastic": [1]})
        out = normalize(df, source="x", register="en")
        assert out.loc[0, TEXT_COLUMN] == "hello"
        assert out.loc[0, LABEL_COLUMN] == 1

    def test_renames_is_sarcastic(self) -> None:
        df = pd.DataFrame({"content": ["x"], "is_sarcastic": [0]})
        out = normalize(df, source="x", register="en")
        assert out.loc[0, LABEL_COLUMN] == 0

    def test_renames_target(self) -> None:
        df = pd.DataFrame({"sentence": ["s"], "target": [1]})
        out = normalize(df, source="x", register="en")
        assert out.loc[0, LABEL_COLUMN] == 1

    def test_raises_when_no_text_column(self) -> None:
        df = pd.DataFrame({"label": [0]})
        with pytest.raises(KeyError, match="No text column"):
            normalize(df, source="x", register="en")

    def test_raises_when_no_label_column(self) -> None:
        df = pd.DataFrame({"text": ["x"]})
        with pytest.raises(KeyError, match="No label column"):
            normalize(df, source="x", register="en")

    def test_coerces_text_to_string(self) -> None:
        df = pd.DataFrame({"text": [123, 456], "label": [0, 1]})
        out = normalize(df, source="x", register="en")
        assert out[TEXT_COLUMN].dtype == object
        assert out.loc[0, TEXT_COLUMN] == "123"

    def test_coerces_label_to_int(self) -> None:
        df = pd.DataFrame({"text": ["a", "b"], "label": [True, False]})
        out = normalize(df, source="x", register="en")
        assert out[LABEL_COLUMN].dtype == "int64"
        assert list(out[LABEL_COLUMN]) == [1, 0]


class TestMergeCorpora:
    def test_no_frames_returns_empty_with_correct_schema(self) -> None:
        merged = merge_corpora()
        assert set(merged.columns) == set(EXPECTED_COLUMNS)
        assert len(merged) == 0

    def test_concatenates_frames(self) -> None:
        a = pd.DataFrame(
            {
                TEXT_COLUMN: ["alpha", "beta"],
                LABEL_COLUMN: [0, 1],
                SOURCE_COLUMN: ["a", "a"],
                REGISTER_COLUMN: ["en", "en"],
            }
        )
        b = pd.DataFrame(
            {
                TEXT_COLUMN: ["gamma"],
                LABEL_COLUMN: [1],
                SOURCE_COLUMN: ["b"],
                REGISTER_COLUMN: ["en-IN"],
            }
        )
        merged = merge_corpora(a, b)
        assert len(merged) == 3
        assert set(merged[TEXT_COLUMN]) == {"alpha", "beta", "gamma"}

    def test_deduplicates_by_default(self) -> None:
        a = pd.DataFrame(
            {
                TEXT_COLUMN: ["dup"],
                LABEL_COLUMN: [0],
                SOURCE_COLUMN: ["a"],
                REGISTER_COLUMN: ["en"],
            }
        )
        b = pd.DataFrame(
            {
                TEXT_COLUMN: ["dup"],
                LABEL_COLUMN: [1],
                SOURCE_COLUMN: ["b"],
                REGISTER_COLUMN: ["en"],
            }
        )
        merged = merge_corpora(a, b)
        assert len(merged) == 1
        # first frame wins
        assert merged.loc[0, SOURCE_COLUMN] == "a"
        assert merged.loc[0, LABEL_COLUMN] == 0

    def test_dedup_can_be_disabled(self) -> None:
        a = pd.DataFrame(
            {
                TEXT_COLUMN: ["dup"],
                LABEL_COLUMN: [0],
                SOURCE_COLUMN: ["a"],
                REGISTER_COLUMN: ["en"],
            }
        )
        b = pd.DataFrame(
            {
                TEXT_COLUMN: ["dup"],
                LABEL_COLUMN: [1],
                SOURCE_COLUMN: ["b"],
                REGISTER_COLUMN: ["en"],
            }
        )
        merged = merge_corpora(a, b, deduplicate=False)
        assert len(merged) == 2


class TestLoadCuratedSupplement:
    def test_returns_empty_frame_when_file_missing(self, tmp_path: Path) -> None:
        df = load_curated_supplement(tmp_path / "does-not-exist.csv")
        assert set(df.columns) == set(EXPECTED_COLUMNS)
        assert len(df) == 0

    def test_loads_csv_and_normalises(self, tmp_path: Path) -> None:
        csv = tmp_path / "indian_english.csv"
        csv.write_text("text,label\nhaa beta nice,1\nworking late again,1\nnice weather,0\n")
        df = load_curated_supplement(csv)
        assert len(df) == 3
        assert (df[SOURCE_COLUMN] == "curated_indian_english").all()
        assert (df[REGISTER_COLUMN] == "en-IN").all()


class TestLoadParquetCorpora:
    def test_isarcasm_raises_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="iSarcasm"):
            load_isarcasm(tmp_path / "nope.parquet")

    def test_semeval_raises_when_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="SemEval"):
            load_semeval_isarcasm(tmp_path / "nope.parquet")

    def test_isarcasm_loads_parquet(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"tweet": ["a", "b"], "sarcastic": [0, 1]})
        path = tmp_path / "isarcasm.parquet"
        df.to_parquet(path)
        out = load_isarcasm(path)
        assert len(out) == 2
        assert (out[SOURCE_COLUMN] == "isarcasm").all()
        assert (out[REGISTER_COLUMN] == "en").all()
