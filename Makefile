PYTHON := uv run python
PYTHON_MODULE := uv run python -m
PIP := uv pip install

.PHONY: setup prewarm checkpoints spacy dinov2 sentence_transformers clean download_wordnet migrate revision

# -------------------------------------------------------------------------
# Database migrations (Alembic owns the schema; create_all was removed).
# Run this after a fresh checkout / new DB and after pulling new migrations.
# -------------------------------------------------------------------------
# Each backend is a separate database with its own alembic_version, so we run
# once per backend. SQLite is migrated always; Postgres is best-effort — if it
# is not up (or POSTGRESQL_URL is unset) it is skipped, not fatal. Postgres creds
# come from the repo-root .env (Alembic's env.py does not load it itself).
migrate:
	@echo "\n--- Applying migrations: SQLite (always) ---"
	@cd backend && CLIPSCRIBE_DB_BACKEND=sqlite uv run alembic upgrade head
	@echo "\n--- Applying migrations: Postgres (if reachable) ---"
	@cd backend && { set -a; [ -f ../.env ] && . ../.env; set +a; \
		CLIPSCRIBE_DB_BACKEND=postgresql uv run alembic upgrade head; } \
		|| echo "  ↳ skipped (Postgres unreachable or POSTGRESQL_URL unset)"
	@echo "\nDatabases are at head."

# Authoring step (dev only): diff schema.py against the DB and WRITE a new
# migration script under alembic/versions/. Does NOT change the DB. Requires
# the DB to already be at head (run `make migrate` first). Pass a message:
#   make revision m="add foo table"
# Always review the generated script; if the diff was empty, delete the file.
revision:
	@echo "\n--- Generating migration script (no DB changes) ---"
	@cd backend && CLIPSCRIBE_DB_BACKEND=sqlite uv run alembic revision --autogenerate -m "$(m)"
	@echo "Review the new file in backend/alembic/versions/ (delete it if empty)."

setup: spacy prewarm
	@echo "\nProject Setup Complete! You can now run the extractor."

prewarm:
	@echo "\n--- Prewarming ClipScribe model assets ---"
	@cd backend && uv run python scripts/prewarm.py


checkpoints:
	@echo "\n--- 1. Downloading Checkpoints ---"
	@cd backend && $(PYTHON) scripts/download_dino.py
	@chmod +x backend/scripts/download_sam_ckpts.sh
	@bash backend/scripts/download_sam_ckpts.sh
	@echo "Checkpoints downloaded."

spacy:
	@echo "\n--- 2. Installing spaCy Model ---"
	@cd backend && $(PIP) "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
	@echo "spaCy model installed."


dinov2:
	@echo "\n--- 5. Pre-fetching DINOv2 Model ---"
	@cd backend && $(PYTHON) -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context; \
	import torch; \
	print('Downloading DINOv2 to cache...'); \
	torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"
	@echo "DINOv2 model cached."


sentence_transformers:
	@echo "\n--- 6. Pre-fetching Sentence Transformer ---"
	@cd backend && $(PYTHON) -c "from sentence_transformers import SentenceTransformer; \
		print('Caching all-MiniLM-L6-v2...'); \
		SentenceTransformer('all-MiniLM-L6-v2')"
	@echo "Sentence Transformer model cached."


download_wordnet:
	@echo "Ensuring WordNet is installed..."
	@cd backend && $(PYTHON) -c "import nltk; nltk.download('wordnet')"

clean:
	@echo "Cleaning up..."
	@rm -f backend/checkpoints/*.pth
	@rm -f backend/checkpoints/*.pt
	@echo "Checkpoints removed."

.PHONY: help
help:
	@echo "🛠️  Available commands:"
