# syntax=docker/dockerfile:1
#
# Heavy Celery worker image — GPU (CUDA) variant. NOT built by docker-compose
# (which is CPU-only for Mac/local dev); this is the image for a Linux + NVIDIA
# host / GKE GPU node pool (see docs/deployment.md §3.2). Same code and entrypoint
# as the CPU Dockerfile; the only differences are the CUDA base image and torch
# installed from the CUDA wheel index. Run the container with `--gpus all` and
# set CLIPSCRIBE_DEVICE=cuda.
#
# Build context is the REPOSITORY ROOT:
#   docker build -f backend/docker/core/gpu/Dockerfile -t clipscribe-worker-gpu .

# ── Stage 1: build the venv ───────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Bring Python 3.12 in from the official slim image (no deadsnakes PPA needed).
COPY --from=python:3.12-slim /usr/local /usr/local
COPY --from=python:3.12-slim /usr/lib /usr/lib
COPY --from=python:3.12-slim /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV VIRTUAL_ENV=/app/backend/.venv \
    PATH="/app/backend/.venv/bin:$PATH"
WORKDIR /app/backend

RUN uv venv --python python3.12 "$VIRTUAL_ENV"

# Install the locked worker runtime. The worker-gpu extra selects the CUDA 12.6
# Torch and Torchvision wheels.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --extra worker-gpu
RUN uv pip install --no-cache \
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
COPY --from=python:3.12-slim /usr/local /usr/local
COPY --from=python:3.12-slim /usr/lib /usr/lib
COPY --from=python:3.12-slim /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/backend/.venv \
    PATH="/app/backend/.venv/bin:$PATH" \
    PYTHONPATH=/app/backend \
    TORCH_HOME=/app/backend/checkpoints/torch_hub \
    HF_HOME=/app/backend/checkpoints/huggingface \
    NLTK_DATA=/app/backend/checkpoints/nltk \
    PADDLE_OCR_BASE_DIR=/app/backend/checkpoints/paddleocr

WORKDIR /app/backend

COPY --from=builder /app/backend/.venv /app/backend/.venv
COPY backend/scripts ./scripts

# Bake every model weight into an image layer (prod pattern): no checkpoints
# volume is mounted on the GPU node, so the container starts instantly with all
# weights present. Downloads several GB at build time; the download/load runs on
# CPU (no GPU needed to build). Uses the checkpoints/ env vars set above.
#
# This runs BEFORE copying app/ and src/ on purpose: prewarm only needs the venv
# + scripts/ (it imports torch/whisper/paddle from the venv and calls
# scripts/download_dino.py — never app or src). Keeping it above the code copy
# means an app/src edit doesn't invalidate this ~GB weights layer; Docker reuses
# the cache and only rebuilds the cheap code-copy layers below.
RUN python scripts/prewarm.py

COPY backend/app ./app
COPY backend/src ./src
COPY backend/main.py ./main.py
COPY backend/docker/core/gpu/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
