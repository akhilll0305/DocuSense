# DocuSense application image.
#
# Two stages so build tooling and pip's cache stay out of the runtime layer.
# The embedding and reranker models are baked in at build time: downloading
# them on first request would otherwise add a minute to the first query and
# make the container depend on HuggingFace being reachable at runtime.
#
# The models run on ONNX Runtime rather than torch. Same weights, same outputs
# — measured, cosine 1.000000 on embeddings and byte-identical cross-encoder
# scores — for 326MB resident instead of 758MB, which is the difference between
# needing a host that will rent you a gigabyte and fitting in a free tier.

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

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt


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
COPY --chown=app:app docker/entrypoint.sh ./docker/entrypoint.sh
COPY --chown=app:app pyproject.toml README.md ./

# The demo papers a public instance seeds itself with on start. Free hosting
# tiers have ephemeral storage, so these belong in the image.
COPY --chown=app:app data/demo ./data/demo

RUN chmod +x ./docker/entrypoint.sh

RUN mkdir -p /app/data /app/logs && chown -R app:app /app
USER app

# Pre-download the models so the first query is not also a download. Through
# DocuSense's own loaders rather than fastembed's, because the loader is what
# sets the tokenizer's truncation and padding to match the torch runtime — a
# model cached by any other path would be cached with the wrong settings.
RUN python -c "\
from docusense.embeddings.backends import load_embedding_backend, load_cross_encoder_backend; \
load_embedding_backend('all-MiniLM-L6-v2'); \
load_cross_encoder_backend('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Hugging Face Spaces and most PaaS hosts inject the port to listen on; 8000
# is the local default.
ENV PORT=8000
EXPOSE 8000

# Hits the one endpoint that needs no authentication.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/api/health" || exit 1

# Seeds the demo shelf when SEED_DEMO is set, then serves. Seeding happens on
# every start because the free tiers this targets have ephemeral storage.
CMD ["./docker/entrypoint.sh"]
