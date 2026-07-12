#!/bin/sh
# Entrypoint for the slim API image. If a command is passed (e.g. the compose
# `migrate` one-shot runs `alembic upgrade head`), exec it; otherwise serve the
# API. Keeping the launch here — rather than a bare Dockerfile CMD — gives one
# place to add pre-flight steps (wait-for-db, etc.) later.
set -e

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
