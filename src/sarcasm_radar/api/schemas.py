"""Pydantic request/response models for the FastAPI service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Decision = Literal["SARCASTIC", "NOT_SARCASTIC"]


class PredictRequest(BaseModel):
    """One transaction-shaped payload: a piece of text to score."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=4000, description="The text to score")


class PredictResponse(BaseModel):
    """Score + decision returned by /predict."""

    model_config = ConfigDict(extra="forbid")

    text: str
    probability: float = Field(ge=0, le=1, description="P(sarcastic)")
    decision: Decision
    threshold: float = Field(ge=0, le=1)
    model_version: str


class TokenContribution(BaseModel):
    """One token's contribution to the LIME explanation."""

    model_config = ConfigDict(extra="forbid")

    token: str
    weight: float = Field(
        description="LIME weight on the SARCASTIC class. Positive = pushes "
        "toward sarcastic, negative = pushes toward not sarcastic."
    )


class ExplainResponse(BaseModel):
    """LIME-style per-token explanation."""

    model_config = ConfigDict(extra="forbid")

    text: str
    probability: float = Field(ge=0, le=1)
    decision: Decision
    threshold: float = Field(ge=0, le=1)
    tokens: list[TokenContribution]
    model_version: str


class HealthResponse(BaseModel):
    """Returned by /health."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    model_version: str
    model_kind: Literal["baseline", "transformer"]
    threshold: float = Field(ge=0, le=1)
