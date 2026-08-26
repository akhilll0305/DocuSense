# DocuSense application image.
#
# Two stages so build tooling and pip's cache stay out of the runtime layer.
# The embedding and reranker models are baked in at build time: downloading
# them on first request would otherwise add a minute to the first query and
# make the container depend on HuggingFace being reachable at runtime.

# ==============================================================================
# Stage 1 — build dependencies
# ==============================================================================
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# CPU-only torch. The default wheel pulls ~2GB of CUDA libraries that are dead
# weight in a container with no GPU.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt


# ==============================================================================
# Stage 2 — runtime
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/app/.cache/huggingface

# libmagic backs file-type detection; tesseract is the OCR fallback for figures.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmagic1 \
        tesseract-ocr \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

# Run as a non-root user; the image should not be able to write to itself.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY --chown=app:app docusense ./docusense
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app pyproject.toml README.md ./

RUN mkdir -p /app/data /app/logs && chown -R app:app /app
USER app

# Pre-download the models so the first query is not also a download.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8000

# Hits the one endpoint that needs no authentication.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "docusense.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
