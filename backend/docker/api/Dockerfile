# syntax=docker/dockerfile:1
#
# Slim FastAPI image (web-app-plan §8).
#
# Serves REST + SSE, reads the DB, serves artifacts, dispatches Celery tasks by
# name, and answers advisory chat. It loads NO pipeline models and is
# torch-free: the `api` dependency group (langgraph/langchain-openai included)
# resolves without torch/paddle/whisper/opencv. The same image also runs the
# one-shot Alembic `migrate` step (it has src/db + alembic, no ML deps needed).
#
# Build context is the REPOSITORY ROOT so the COPYs can reach backend/:
#   docker build -f backend/docker/api/Dockerfile -t clipscribe-api .

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/backend/.venv \
    PATH="/app/backend/.venv/bin:$PATH" \
    # backend/ on the import path so BOTH `app` (not a packaged module) and `src`
    # resolve exactly like running locally from backend/. See README "How imports
    # resolve in the containers".
    PYTHONPATH=/app/backend

WORKDIR /app/backend

RUN uv venv "$VIRTUAL_ENV"

# Install ONLY the slim `api` dependency group, from the LOCK (not a fresh
# resolve): `uv export --frozen` dumps uv.lock's pinned versions for the group,
# then we install exactly those — same lock-fidelity pattern as the worker image,
# minus the torch/CUDA handling (the api group is torch-free). `--only-group`
# excludes the heavy base `[project.dependencies]`.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv export --frozen --no-emit-project --no-hashes --only-group api \
        -o /tmp/api-reqs.txt && \
    uv pip install --no-cache -r /tmp/api-reqs.txt

# Copy just what the API imports at module load; the extractor/ocr/dino/sam2
# trees are intentionally omitted to keep the image slim and torch-free.
COPY backend/app ./app
COPY backend/src/db ./src/db
COPY backend/src/parser ./src/parser
COPY backend/src/utils ./src/utils
COPY backend/src/clip_scribe ./src/clip_scribe
COPY backend/alembic ./alembic
COPY backend/alembic.ini ./alembic.ini
COPY backend/docker/api/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
