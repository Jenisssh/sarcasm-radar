# Multi-stage build for the sarcasm-radar FastAPI inference service.
#
# Stage 1 (builder) installs build deps and the editable package wheel.
# Stage 2 (runtime) copies the venv onto a slim Python base.
#
# The torch + transformers footprint is ~2.5GB. To keep the image
# manageable we install the CPU-only torch wheel from PyTorch's
# dedicated CPU index, which saves ~2GB over the default CUDA build.

# --------------------------------------------------------------- Stage 1
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update \
 && apt-get install --no-install-recommends -y \
        build-essential git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN /opt/venv/bin/pip install --no-deps .

# --------------------------------------------------------------- Stage 2
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    SARCASM_RADAR_MODELS_DIR=/app/models \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

RUN useradd --create-home --shell /usr/sbin/nologin --uid 1000 radar

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
RUN mkdir -p /app/models && chown -R radar:radar /app

USER radar
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "sarcasm_radar.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
