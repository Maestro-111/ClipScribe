# syntax=docker/dockerfile:1
#
# Heavy Celery worker image — CPU variant (web-app-plan §8).
#
# Runs the full extraction + parsing pipeline (torch, paddle, whisper, sam2,
# dino, sbert, ...). This is the CPU build: torch comes from the CPU wheel index
# so it stays Linux-container-safe. The GPU variant lives in gpu/Dockerfile
# (CUDA base + CUDA torch index).
#
# Model WEIGHTS are NOT baked here. In dev they arrive via the mounted
# ./backend/checkpoints volume and are fetched once with `scripts/prewarm.py`
# (`docker compose run --rm worker python scripts/prewarm.py`). A prod image can
# add a `RUN python scripts/prewarm.py` step to bake them into a layer.
#
# Build context is the REPOSITORY ROOT:
#   docker build -f backend/docker/core/Dockerfile -t clipscribe-worker .

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# System libraries the CV / OCR / audio stack needs at runtime:
#   ffmpeg            - video decode (OpenCV, scenedetect) + Whisper audio
#   libgl1 / libglib  - OpenCV (libGL.so.1) and PaddleOCR
#   ca-certificates   - HTTPS model downloads (torch.hub, HF, paddle)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/backend/.venv \
    PATH="/app/backend/.venv/bin:$PATH" \
    # backend/ on the import path so `app` and `src` (incl. the src.-prefixed
    # sam2/dino trees) resolve like a local run from backend/.
    PYTHONPATH=/app/backend \
    # Centralize every env-driven model cache under checkpoints/ (one mount in
    # dev, one baked layer in prod). Whisper uses an explicit download_root in
    # build_clip_scribe.py; MTCNN ships its weights inside the pip wheel.
    TORCH_HOME=/app/backend/checkpoints/torch_hub \
    HF_HOME=/app/backend/checkpoints/huggingface \
    NLTK_DATA=/app/backend/checkpoints/nltk \
    PADDLE_OCR_BASE_DIR=/app/backend/checkpoints/paddleocr

WORKDIR /app/backend

RUN uv venv "$VIRTUAL_ENV"

# Install the LOCKED `worker` group. We export the group from uv.lock rather
# than `uv pip install --group worker` because a fresh resolve backtracks numba
# to an ancient version whose llvmlite can't build on Python 3.12. Exporting
# pins the same 3.12-compatible versions the repo already resolved. Then: strip
# the CUDA torch stack from the export, install torch/torchvision from the CPU
# wheel index, and install the rest pinned.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv export --frozen --no-emit-project --no-hashes --only-group worker \
        -o /tmp/worker-reqs.txt && \
    sed -i '/^torch==/d; /^torchvision==/d; /^nvidia-/d' /tmp/worker-reqs.txt && \
    uv pip install --no-cache torch==2.9.1 torchvision==0.24.1 \
        --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --no-cache -r /tmp/worker-reqs.txt

# spaCy's en_core_web_sm is a pip wheel (not a cache download) — bake it into the
# venv so it never needs fetching at runtime (mirrors `make spacy`).
RUN uv pip install --no-cache \
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

# Full core source: extractor / ocr / dino / sam2 / parser / db / utils, the app
# package (Celery task + job lifecycle), the CLI entry, and prewarm.
COPY backend/app ./app
COPY backend/src ./src
COPY backend/main.py ./main.py
COPY backend/scripts ./scripts
COPY backend/docker/core/cpu/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
