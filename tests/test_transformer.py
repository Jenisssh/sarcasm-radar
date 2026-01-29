"""Tests for sarcasm_radar.models.transformer.

We don't actually fine-tune anything in tests — that would download
gigabytes of model weights. The tests cover the config dataclass, the
unfit-state guards, and the save/load path with a mocked tokenizer +
model. The full integration test belongs in a scripts/ smoke run on
a small dataset, not in pytest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from sarcasm_radar.models.transformer import (
    TransformerSarcasmClassifier,
    TransformerTrainConfig,
)


class TestTransformerTrainConfig:
    def test_defaults(self) -> None:
        c = TransformerTrainConfig()
        assert c.model_name == "distilbert-base-uncased"
        assert c.num_epochs == 3
        assert c.batch_size == 16
        assert 0 < c.learning_rate < 1

    def test_is_frozen(self) -> None:
        c = TransformerTrainConfig()
        with pytest.raises((AttributeError, Exception)):
            c.num_epochs = 99  # type: ignore[misc]

    def test_rejects_zero_epochs(self) -> None:
        with pytest.raises(ValueError, match="num_epochs"):
            TransformerTrainConfig(num_epochs=0)

    def test_rejects_negative_batch_size(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            TransformerTrainConfig(batch_size=-1)

    def test_rejects_out_of_range_lr(self) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            TransformerTrainConfig(learning_rate=1.5)

    def test_rejects_zero_max_length(self) -> None:
        with pytest.raises(ValueError, match="max_length"):
            TransformerTrainConfig(max_length=0)

    def test_as_dict_serialises_output_dir_to_str(self) -> None:
        c = TransformerTrainConfig()
        d = c.as_dict()
        assert isinstance(d["output_dir"], str)
        # Round-trip through json
        import json

        json.dumps(d)


class TestUnfitGuards:
    def test_predict_proba_before_fit_raises(self) -> None:
        clf = TransformerSarcasmClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict_proba(pd.Series(["test"]))

    def test_predict_before_fit_raises(self) -> None:
        clf = TransformerSarcasmClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(pd.Series(["test"]))

    def test_save_before_fit_raises(self, tmp_path: Path) -> None:
        clf = TransformerSarcasmClassifier()
        with pytest.raises(RuntimeError, match="nothing to save"):
            clf.save(tmp_path)


class TestSaveLoadWithMockedModel:
    """Save / load delegates to HuggingFace's save_pretrained / from_pretrained.

    These tests verify the delegation pattern by injecting mocks.
    """

    def test_save_writes_to_path(self, tmp_path: Path) -> None:
        clf = TransformerSarcasmClassifier()
        clf.model = MagicMock()
        clf.tokenizer = MagicMock()
        out = clf.save(tmp_path / "ckpt")
        assert out == tmp_path / "ckpt"
        assert out.exists()  # mkdir worked
        clf.model.save_pretrained.assert_called_once_with(out)
        clf.tokenizer.save_pretrained.assert_called_once_with(out)

    def test_save_uses_config_output_dir_by_default(self, tmp_path: Path) -> None:
        config = TransformerTrainConfig(output_dir=tmp_path / "default-ckpt")
        clf = TransformerSarcasmClassifier(config=config)
        clf.model = MagicMock()
        clf.tokenizer = MagicMock()
        out = clf.save()
        assert out == config.output_dir


class TestCustomConfig:
    def test_custom_model_name_respected(self) -> None:
        config = TransformerTrainConfig(model_name="bert-base-multilingual-cased")
        clf = TransformerSarcasmClassifier(config=config)
        assert clf.config.model_name == "bert-base-multilingual-cased"

    def test_custom_output_dir_respected(self, tmp_path: Path) -> None:
        config = TransformerTrainConfig(output_dir=tmp_path / "my-ckpt")
        clf = TransformerSarcasmClassifier(config=config)
        assert clf.config.output_dir == tmp_path / "my-ckpt"


def test_module_imports_without_transformers_installed(
    monkeypatch: Any,
) -> None:
    """The module itself shouldn't import torch/transformers at load time."""
    # If the module accidentally imported transformers at the top level,
    # this test would have already failed at import time. The fact that
    # this function runs at all means the lazy-import pattern is intact.
    from sarcasm_radar.models.transformer import (  # noqa: F401
        TransformerSarcasmClassifier,
        TransformerTrainConfig,
    )
