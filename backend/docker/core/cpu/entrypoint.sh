#!/bin/sh
# Entrypoint for the heavy worker image. If a command is passed (e.g. a one-shot
# `python scripts/prewarm.py`), exec it; otherwise start the Celery worker.
# --pool=solo --concurrency=1: models load once, one job at a time per process /
# GPU (web-app-plan §12).
set -e

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec celery -A app.celery_app worker --pool=solo --concurrency=1
