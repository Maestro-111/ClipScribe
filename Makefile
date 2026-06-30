PYTHON := uv run python
PYTHON_MODULE := uv run python -m
PIP := uv pip install

.PHONY: setup checkpoints spacy blip dinov2 sentence_transformers fix_mac_ssl clean run_extractor download_wordnet migrate revision

# -------------------------------------------------------------------------
# Database migrations (Alembic owns the schema; create_all was removed).
# Run this after a fresh checkout / new DB and after pulling new migrations.
# -------------------------------------------------------------------------
migrate:
	@echo "\n--- Applying database migrations ---"
	@cd backend && uv run alembic upgrade head
	@echo "Database is at head."

# Authoring step (dev only): diff schema.py against the DB and WRITE a new
# migration script under alembic/versions/. Does NOT change the DB. Requires
# the DB to already be at head (run `make migrate` first). Pass a message:
#   make revision m="add foo table"
# Always review the generated script; if the diff was empty, delete the file.
revision:
	@echo "\n--- Generating migration script (no DB changes) ---"
	@cd backend && uv run alembic revision --autogenerate -m "$(m)"
	@echo "Review the new file in backend/alembic/versions/ (delete it if empty)."

setup: checkpoints spacy blip dinov2 sentence_transformers fix_mac_ssl
	@echo "\nProject Setup Complete! You can now run the extractor."

# -------------------------------------------------------------------------
# 1. Download Model Checkpoints (DINO & SAM 2)
# -------------------------------------------------------------------------
checkpoints:
	@echo "\n--- 1. Downloading Checkpoints ---"
	@cd backend
	@$(PYTHON) checkpoints/download_dino.py
	@chmod +x checkpoints/download_sam_ckpts.sh
	@cd checkpoints && ./download_sam_ckpts.sh
	@echo "Checkpoints downloaded."

# -------------------------------------------------------------------------
# 2. Install spaCy Model (en_core_web_sm)
# -------------------------------------------------------------------------
spacy:
	@echo "\n--- 2. Installing spaCy Model ---"
	@$(PIP) "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
	@echo "spaCy model installed."

# -------------------------------------------------------------------------
# 4. Pre-fetch DINOv2 Model (Torch Hub)
# -------------------------------------------------------------------------
dinov2:
	@echo "\n--- 5. Pre-fetching DINOv2 Model ---"
	@$(PYTHON) -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context; \
	import torch; \
	print('Downloading DINOv2 to cache...'); \
	torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"
	@echo "DINOv2 model cached."

# -------------------------------------------------------------------------
# 5. Pre-fetch all-MiniLM-L6-v2 Sentence-Transformer
# -------------------------------------------------------------------------
sentence_transformers:
	@echo "\n--- 6. Pre-fetching Sentence Transformer ---"
	@$(PYTHON) -c "from sentence_transformers import SentenceTransformer; \
		print('Caching all-MiniLM-L6-v2...'); \
		SentenceTransformer('all-MiniLM-L6-v2')"
	@echo "Sentence Transformer model cached."

# -------------------------------------------------------------------------
# 6. Fix macOS SSL Certificates (Mac Specific)
# -------------------------------------------------------------------------
fix_mac_ssl:
	@echo "\n--- Checking for macOS SSL Certificate Issue ---"
	@if [ "$$(uname)" = "Darwin" ]; then \
		echo "macOS detected. Attempting to install certificates..."; \
		CERT_CMD="/Applications/Python 3.13/Install Certificates.command"; \
		if [ -f "$$CERT_CMD" ]; then \
			sh "$$CERT_CMD"; \
			echo "Certificates installed successfully."; \
		else \
			echo "Certificate command not found at standard path. Skipping."; \
			echo "If you get SSL errors, please run 'Install Certificates.command' manually."; \
		fi \
	else \
		echo "Not on macOS. Skipping certificate fix."; \
	fi

# -------------------------------------------------------------------------
# 7. NLTK WordNet corpus
# -------------------------------------------------------------------------
download_wordnet:
	@echo "Ensuring WordNet is installed..."
	@python -c "import nltk; nltk.download('wordnet')"

clean:
	@echo "Cleaning up..."
	@cd backend
	@rm -f checkpoints/*.pth
	@rm -f checkpoints/*.pt
	@echo "Checkpoints removed."

.PHONY: help
help:
	@echo "🛠️  Available commands:"
