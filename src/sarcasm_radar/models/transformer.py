"""DistilBERT fine-tuning for sarcasm classification.

Why DistilBERT for the first transformer pass:

- ~66M params, runs comfortably on CPU for inference and on a single
  consumer GPU for training in under an hour on the merged corpus.
- The English vocabulary is good enough for the iSarcasm + SemEval
  bulk; the Hinglish-aware pass (XLM-RoBERTa) lives in week 2.
- HuggingFace's :class:`Trainer` API handles checkpointing, mixed
  precision, and gradient accumulation, so the wrapper here stays
  thin.

The wrapper exposes a small sklearn-shaped surface (``fit``,
``predict``, ``predict_proba``, ``save``, ``load``) so it composes with
the same evaluation harness the baseline uses. All transformers /
torch imports are lazy — the rest of the package shouldn't have to
load them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sarcasm_radar.config import settings
from sarcasm_radar.data.clean import clean_text


@dataclass(frozen=True, slots=True)
class TransformerTrainConfig:
    """Hyperparameters for one fine-tuning run.

    Defaults are tuned for DistilBERT on the merged sarcasm corpus —
    typical runs reach val macro-F1 around 0.80 with these. Override
    ``model_name`` to use a different checkpoint (RoBERTa, BERT-base, etc.).
    """

    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    learning_rate: float = 5e-5
    batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: settings.models_dir / "distilbert")

    def __post_init__(self) -> None:
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be > 0; got {self.num_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {self.batch_size}")
        if not 0 < self.learning_rate < 1:
            raise ValueError(f"learning_rate must be in (0, 1); got {self.learning_rate}")
        if self.max_length <= 0:
            raise ValueError(f"max_length must be > 0; got {self.max_length}")

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["output_dir"] = str(d["output_dir"])
        return d


class TransformerSarcasmClassifier:
    """Thin wrapper around HuggingFace Trainer for binary sarcasm classification.

    Workflow:

    1. ``clf = TransformerSarcasmClassifier(config)``
    2. ``clf.fit(X_train, y_train, X_val=..., y_val=...)`` — fine-tunes
       the model, saves checkpoints under ``config.output_dir``.
    3. ``clf.predict_proba(X)`` / ``clf.predict(X)`` — inference.
    4. ``clf.save(path)`` / ``TransformerSarcasmClassifier.load(path)``.
    """

    def __init__(self, config: TransformerTrainConfig | None = None) -> None:
        self.config = config or TransformerTrainConfig()
        self.tokenizer: Any = None
        self.model: Any = None

    def fit(
        self,
        X_train: pd.Series,
        y_train: pd.Series,
        X_val: pd.Series | None = None,
        y_val: pd.Series | None = None,
    ) -> TransformerSarcasmClassifier:
        """Fine-tune the transformer on ``(X_train, y_train)``.

        Optionally evaluates on ``(X_val, y_val)`` after every epoch and
        keeps the best checkpoint by macro-F1.
        """
        # Lazy import — heavy
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
        )

        def encode(batch: dict[str, Any]) -> dict[str, Any]:
            encoded: dict[str, Any] = tokenizer(
                [clean_text(t) for t in batch["text"]],
                truncation=True,
                max_length=self.config.max_length,
            )
            return encoded

        train_ds = Dataset.from_dict(
            {"text": list(X_train), "label": [int(v) for v in y_train]}
        ).map(encode, batched=True, remove_columns=["text"])

        eval_ds = None
        if X_val is not None and y_val is not None:
            eval_ds = Dataset.from_dict(
                {"text": list(X_val), "label": [int(v) for v in y_val]}
            ).map(encode, batched=True, remove_columns=["text"])

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            seed=self.config.seed,
            eval_strategy="epoch" if eval_ds is not None else "no",
            save_strategy="epoch" if eval_ds is not None else "no",
            load_best_model_at_end=eval_ds is not None,
            metric_for_best_model="macro_f1" if eval_ds is not None else None,
            greater_is_better=True,
            logging_steps=50,
            report_to=[],
        )

        def compute_metrics(eval_pred: Any) -> dict[str, float]:
            from sklearn.metrics import accuracy_score, f1_score

            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": float(accuracy_score(labels, preds)),
                "macro_f1": float(f1_score(labels, preds, average="macro")),
            }

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            # `processing_class` replaced the `tokenizer` arg in transformers
            # 4.46; the old name was removed outright in 4.47.
            processing_class=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics if eval_ds is not None else None,
            callbacks=(
                [EarlyStoppingCallback(early_stopping_patience=2)] if eval_ds is not None else []
            ),
        )
        trainer.train()

        self.tokenizer = tokenizer
        self.model = model
        return self

    def predict_proba(self, X: pd.Series) -> NDArray[Any]:
        """Return shape-(n, 2) probabilities.

        Runs inference in batches of ``config.eval_batch_size`` and moves
        each batch onto the model's own device. After ``fit`` the HF Trainer
        leaves the model on the GPU, so the encoded inputs (built on CPU by
        default) must be moved to match — otherwise the forward pass raises
        a device-mismatch RuntimeError. Batching also keeps a large input
        set from blowing the GPU memory budget in a single forward pass.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("TransformerSarcasmClassifier is not fitted yet")
        import torch

        self.model.eval()
        device = next(self.model.parameters()).device
        texts = [clean_text(t) for t in X]
        if not texts:
            return np.empty((0, 2), dtype="float64")

        batches: list[NDArray[Any]] = []
        for start in range(0, len(texts), self.config.eval_batch_size):
            chunk = texts[start : start + self.config.eval_batch_size]
            encoded = self.tokenizer(
                chunk,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
            batches.append(torch.softmax(logits, dim=-1).cpu().numpy())
        probs: NDArray[Any] = np.concatenate(batches, axis=0).astype("float64")
        return probs

    def predict(self, X: pd.Series, threshold: float = 0.5) -> NDArray[Any]:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    def save(self, path: Path | None = None) -> Path:
        """Save the model + tokenizer to disk (HuggingFace format)."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("nothing to save — call fit() first")
        path = path or self.config.output_dir
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        return path

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        config: TransformerTrainConfig | None = None,
    ) -> TransformerSarcasmClassifier:
        """Load a previously-saved model from ``path``."""
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        instance = cls(config=config)
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path)
        return instance
