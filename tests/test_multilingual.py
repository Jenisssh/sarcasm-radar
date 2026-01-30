"""Tests for sarcasm_radar.models.multilingual."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sarcasm_radar.models.multilingual import (
    XLMR_BASE_MODEL_NAME,
    PerRegisterScore,
    compare_per_register,
    make_xlmr_config,
    per_register_metrics,
)


class TestMakeXLMRConfig:
    def test_uses_xlmr_checkpoint(self) -> None:
        config = make_xlmr_config()
        assert config.model_name == XLMR_BASE_MODEL_NAME

    def test_default_lower_lr_than_distilbert(self) -> None:
        # XLM-R is more sensitive — preset uses a lower LR than the
        # default DistilBERT 5e-5
        config = make_xlmr_config()
        assert config.learning_rate < 5e-5

    def test_default_longer_max_length(self) -> None:
        # Hinglish utterances run longer; preset bumps max_length
        config = make_xlmr_config()
        assert config.max_length >= 160

    def test_custom_output_dir(self, tmp_path: Path) -> None:
        config = make_xlmr_config(output_dir=tmp_path / "x")
        assert config.output_dir == tmp_path / "x"

    def test_overrides_respected(self) -> None:
        config = make_xlmr_config(
            learning_rate=3e-5,
            num_epochs=5,
            batch_size=8,
            max_length=200,
        )
        assert config.learning_rate == 3e-5
        assert config.num_epochs == 5
        assert config.batch_size == 8
        assert config.max_length == 200


class TestPerRegisterMetrics:
    def test_returns_one_score_per_register(self) -> None:
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0, 1, 0])
        registers = pd.Series(["en", "en", "en-IN", "en-IN", "hi-en", "hi-en"])
        scores = per_register_metrics(y_true, y_pred, registers)
        assert len(scores) == 3
        for s in scores:
            assert isinstance(s, PerRegisterScore)
            assert s.macro_f1 == pytest.approx(1.0)

    def test_n_counts_rows_in_each_register(self) -> None:
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        registers = pd.Series(["en", "en", "en", "en-IN", "en-IN"])
        scores = per_register_metrics(y_true, y_pred, registers)
        by_register = {s.register: s.n for s in scores}
        assert by_register == {"en": 3, "en-IN": 2}

    def test_imperfect_predictions_per_register(self) -> None:
        # en is perfect, en-IN gets one wrong
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        registers = pd.Series(["en", "en", "en-IN", "en-IN"])
        scores = per_register_metrics(y_true, y_pred, registers)
        by_register = {s.register: s.macro_f1 for s in scores}
        assert by_register["en"] == pytest.approx(1.0)
        assert by_register["en-IN"] < 1.0

    def test_misaligned_lengths_raise(self) -> None:
        y = np.array([0, 1, 0])
        registers = pd.Series(["en", "en"])
        with pytest.raises(ValueError, match="align"):
            per_register_metrics(y, y, registers)

    def test_per_register_score_is_frozen(self) -> None:
        s = PerRegisterScore(register="en", n=10, macro_f1=0.5)
        with pytest.raises((AttributeError, Exception)):
            s.macro_f1 = 1.0  # type: ignore[misc]


class TestComparePerRegister:
    @pytest.fixture
    def sample(self) -> dict:
        y_true = np.array([0, 1, 1, 0, 1, 0])
        registers = pd.Series(["en", "en", "en-IN", "en-IN", "hi-en", "hi-en"])
        # DistilBERT: perfect on en, struggles on en-IN
        distilbert = np.array([0, 1, 0, 0, 1, 1])
        # XLM-R: perfect on en-IN/hi-en, struggles on en
        xlmr = np.array([1, 1, 1, 0, 1, 0])
        return {
            "y_true": y_true,
            "registers": registers,
            "predictions": {"distilbert": distilbert, "xlm-r": xlmr},
        }

    def test_returns_dataframe_with_one_row_per_register(self, sample: dict) -> None:
        out = compare_per_register(sample["y_true"], sample["predictions"], sample["registers"])
        assert isinstance(out, pd.DataFrame)
        # 3 registers in the fixture
        assert len(out) == 3

    def test_has_column_per_model(self, sample: dict) -> None:
        out = compare_per_register(sample["y_true"], sample["predictions"], sample["registers"])
        assert "distilbert" in out.columns
        assert "xlm-r" in out.columns
        assert "register" in out.columns
        assert "n" in out.columns

    def test_xlmr_beats_distilbert_on_en_in(self, sample: dict) -> None:
        out = compare_per_register(sample["y_true"], sample["predictions"], sample["registers"])
        en_in = out[out["register"] == "en-IN"].iloc[0]
        # The fixture is constructed so XLM-R does better on en-IN
        assert en_in["xlm-r"] > en_in["distilbert"]
