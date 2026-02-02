"""FastAPI dependencies — artifact loading and per-request access.

The service holds one ``ModelArtifacts`` for the lifetime of the
process. It's loaded once in the lifespan hook (see ``main.py``) and
stashed on ``app.state.artifacts``. ``get_artifacts`` is the dependency
every endpoint declares; integration tests override it to inject a
stub so they don't need a trained model on disk.
"""

from __future__ import annotations

from fastapi import HTTPException, Request, status

from sarcasm_radar.models.inference import ModelArtifacts


def get_artifacts(request: Request) -> ModelArtifacts:
    """FastAPI dependency that returns the per-app artifacts or 503s.

    Tests substitute this with ``app.dependency_overrides[get_artifacts]
    = stub``.
    """
    artifacts: ModelArtifacts | None = getattr(request.app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts not loaded — train and persist before serving",
        )
    return artifacts
