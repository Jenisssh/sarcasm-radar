"""FastAPI inference service for sarcasm-radar.

Three endpoints:

  GET  /health    liveness + model version + active threshold
  POST /predict   probability + binary decision
  POST /explain   probability + decision + LIME per-token weights

Model artifacts load once on startup via the lifespan hook. Missing
artifacts don't crash the process — they cause /health, /predict,
and /explain to return 503 (degraded mode). That's the production
default: come up cleanly, surface a clear error.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Request, Response
from numpy.typing import NDArray

from sarcasm_radar import __version__
from sarcasm_radar.api.dependencies import get_artifacts
from sarcasm_radar.api.schemas import (
    Decision,
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    TokenContribution,
)
from sarcasm_radar.models.inference import (
    ModelArtifacts,
    ModelKind,
    load_artifacts,
    predict_single,
)
from sarcasm_radar.utils.logging import get_logger

log = get_logger("sarcasm_radar.api")

LIME_NUM_SAMPLES = 200  # tradeoff: explanation quality vs latency
LIME_NUM_FEATURES = 10


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model artifacts once at startup; clear them on shutdown.

    Tries the transformer checkpoint first, then the TF-IDF baseline.
    Whichever artifact is present on disk gets served — so a local dev
    box can run the lightweight baseline while production serves the
    transformer, with no config change. If neither is found the service
    still starts (degraded mode) and the endpoints return 503.
    """
    app.state.artifacts = None
    candidates: tuple[ModelKind, ...] = ("transformer", "baseline")
    for kind in candidates:
        try:
            artifacts = load_artifacts(kind=kind)
        except FileNotFoundError:
            continue
        app.state.artifacts = artifacts
        log.info(
            "model_loaded",
            kind=artifacts.kind,
            version=artifacts.model_version,
            threshold=artifacts.threshold,
        )
        break
    else:
        log.warning("model_artifacts_missing", searched=list(candidates))
    yield
    app.state.artifacts = None


app = FastAPI(
    title="sarcasm-radar API",
    description=(
        "Sarcasm detection for Indian English tweets, with Hinglish support. "
        "Powered by DistilBERT / XLM-RoBERTa."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Response:
    """Structured request log: method, path, status, duration_ms."""
    started = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - started) * 1000.0
    log.info(
        "request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    return response


def _decision(probability: float, threshold: float) -> Decision:
    return "SARCASTIC" if probability >= threshold else "NOT_SARCASTIC"


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health(artifacts: ModelArtifacts = Depends(get_artifacts)) -> HealthResponse:
    """Liveness + the model version and operating threshold."""
    return HealthResponse(
        status="ok",
        model_version=artifacts.model_version,
        model_kind=artifacts.kind,
        threshold=artifacts.threshold,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(
    payload: PredictRequest,
    artifacts: ModelArtifacts = Depends(get_artifacts),
) -> PredictResponse:
    """Score a single utterance."""
    result = predict_single(payload.text, artifacts)
    return PredictResponse(
        text=result.text,
        probability=result.probability,
        decision=result.decision,
        threshold=artifacts.threshold,
        model_version=result.model_version,
    )


@app.post("/explain", response_model=ExplainResponse, tags=["inference"])
def explain(
    payload: PredictRequest,
    artifacts: ModelArtifacts = Depends(get_artifacts),
) -> ExplainResponse:
    """Score + LIME per-token weights.

    LIME perturbs the input by removing tokens, scores each perturbation,
    and fits a local linear surrogate. The returned weights are the
    surrogate's coefficients on the SARCASTIC class — positive means
    the token pushed toward sarcastic, negative means toward not.
    """
    from lime.lime_text import LimeTextExplainer

    def lime_predict_fn(texts: list[str]) -> NDArray[Any]:
        if artifacts.is_transformer:
            import pandas as pd

            return np.asarray(artifacts.classifier.predict_proba(pd.Series(texts)))
        return np.asarray(artifacts.classifier.predict_proba(texts))

    explainer = LimeTextExplainer(class_names=["not_sarcastic", "sarcastic"])
    explanation = explainer.explain_instance(
        payload.text,
        lime_predict_fn,
        num_features=LIME_NUM_FEATURES,
        num_samples=LIME_NUM_SAMPLES,
        labels=(1,),
    )
    probs = lime_predict_fn([payload.text])[0]
    probability = float(probs[1])
    token_weights = explanation.as_list(label=1)

    return ExplainResponse(
        text=payload.text,
        probability=probability,
        decision=_decision(probability, artifacts.threshold),
        threshold=artifacts.threshold,
        tokens=[TokenContribution(token=t, weight=w) for t, w in token_weights],
        model_version=artifacts.model_version,
    )
