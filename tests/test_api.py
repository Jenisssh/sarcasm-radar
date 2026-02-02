"""Integration tests for the FastAPI inference service.

Uses TestClient + dependency_overrides to inject stub artifacts so the
tests don't need a trained model on disk.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sarcasm_radar.api.dependencies import get_artifacts
from sarcasm_radar.api.main import app
from sarcasm_radar.models.inference import ModelArtifacts


class _StubClassifier:
    """sklearn-Pipeline-compatible mock that returns a fixed P(sarcastic)."""

    def __init__(self, prob: float) -> None:
        self.prob = prob

    def predict_proba(self, texts: Any) -> np.ndarray:
        n = len(texts)
        return np.column_stack([np.full(n, 1.0 - self.prob), np.full(n, self.prob)])


@pytest.fixture
def sarcastic_artifacts() -> ModelArtifacts:
    return ModelArtifacts(
        classifier=_StubClassifier(prob=0.85),
        kind="baseline",
        model_version="v0.1.0-test",
        threshold=0.5,
    )


@pytest.fixture
def sincere_artifacts() -> ModelArtifacts:
    return ModelArtifacts(
        classifier=_StubClassifier(prob=0.15),
        kind="baseline",
        model_version="v0.1.0-test",
        threshold=0.5,
    )


@pytest.fixture
def client(sarcastic_artifacts: ModelArtifacts) -> Iterator[TestClient]:
    app.dependency_overrides[get_artifacts] = lambda: sarcastic_artifacts
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def sincere_client(sincere_artifacts: ModelArtifacts) -> Iterator[TestClient]:
    app.dependency_overrides[get_artifacts] = lambda: sincere_artifacts
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


class TestHealth:
    def test_returns_200_with_status_ok(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_version"] == "v0.1.0-test"
        assert body["model_kind"] == "baseline"
        assert body["threshold"] == 0.5

    def test_response_keys_locked(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert set(body.keys()) == {"status", "model_version", "model_kind", "threshold"}


class TestPredict:
    def test_returns_score_and_decision(self, client: TestClient) -> None:
        r = client.post("/predict", json={"text": "haa beta nice plan"})
        assert r.status_code == 200
        body = r.json()
        assert body["probability"] == pytest.approx(0.85)
        assert body["decision"] == "SARCASTIC"
        assert body["text"] == "haa beta nice plan"

    def test_decision_flips_for_sincere_input(self, sincere_client: TestClient) -> None:
        r = sincere_client.post("/predict", json={"text": "the weather is pleasant"})
        body = r.json()
        assert body["probability"] == pytest.approx(0.15)
        assert body["decision"] == "NOT_SARCASTIC"

    def test_422_on_empty_text(self, client: TestClient) -> None:
        r = client.post("/predict", json={"text": ""})
        assert r.status_code == 422

    def test_422_on_missing_text(self, client: TestClient) -> None:
        r = client.post("/predict", json={})
        assert r.status_code == 422

    def test_422_on_extra_field(self, client: TestClient) -> None:
        r = client.post("/predict", json={"text": "x", "sneaky": 1})
        assert r.status_code == 422

    def test_422_on_oversize_text(self, client: TestClient) -> None:
        r = client.post("/predict", json={"text": "x" * 5_000})
        assert r.status_code == 422


class TestDegradedMode:
    def test_503_when_artifacts_missing(self) -> None:
        # No dependency override + no lifespan (TestClient without 'with')
        c = TestClient(app)
        for path, method in (("/health", "get"), ("/predict", "post"), ("/explain", "post")):
            r = c.get(path) if method == "get" else c.post(path, json={"text": "anything"})
            assert r.status_code == 503, f"{path} should be 503 when degraded"
            assert "not loaded" in r.json()["detail"].lower()


class TestOpenAPI:
    def test_three_documented_routes(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert "/health" in schema["paths"]
        assert "/predict" in schema["paths"]
        assert "/explain" in schema["paths"]
