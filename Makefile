PYTHON := uv run python
PYTHON_MODULE := uv run python -m
PIP := uv pip install

.PHONY: setup checkpoints spacy blip clean run_extractor

setup: checkpoints spacy blip taxonomy dinov2
	@echo "\nProject Setup Complete! You can now run the extractor."

# -------------------------------------------------------------------------
# 1. Download Model Checkpoints (DINO & SAM 2)
# -------------------------------------------------------------------------
checkpoints:
	@echo "\n--- 1. Downloading Checkpoints ---"
	@chmod +x checkpoints/download_dingo_ckpts.sh
	@chmod +x checkpoints/download_sam_ckpts.sh
	@cd checkpoints && ./download_dingo_ckpts.sh
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
# 3. Pre-fetch BLIP Model (Hugging Face)
# -------------------------------------------------------------------------
blip:
	@echo "\n--- 3. Pre-fetching BLIP Model ---"
	@echo "This triggers the download now so it doesn't hang during execution."
	@$(PYTHON) -c "from transformers import BlipProcessor, BlipForConditionalGeneration; \
	print('Downloading Processor...'); BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base'); \
	print('Downloading Model...'); BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base')"
	@echo "BLIP model cached."

# -------------------------------------------------------------------------
# 4. Pre-fetch BART-Large-MNLI Model (Hugging Face)
# -------------------------------------------------------------------------
taxonomy:
	@echo "\n--- 4. Pre-fetching BART-Large-MNLI Model ---"
	@$(PYTHON) -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
		model_name = 'facebook/bart-large-mnli'; \
		print(f'Downloading model {model_name}...'); \
		AutoModelForSequenceClassification.from_pretrained(model_name); \
		print('Downloading tokenizer...'); \
		AutoTokenizer.from_pretrained(model_name)"
	@echo "BART model cached successfully."

# -------------------------------------------------------------------------
# 5. Pre-fetch DINOv2 Model (Torch Hub)
# -------------------------------------------------------------------------
dinov2:
	@echo "\n--- 5. Pre-fetching DINOv2 Model ---"
	@$(PYTHON) -c "import ssl; ssl._create_default_https_context = ssl._create_unverified_context; \
	import torch; \
	print('Downloading DINOv2 to cache...'); \
	torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')"
	@echo "DINOv2 model cached."


clean:
	@echo "Cleaning up..."
	@rm -f checkpoints/*.pth
	@rm -f checkpoints/*.pt
	@echo "Checkpoints removed."

run_extractor:
	@echo "Running extractor"
	@$(PYTHON_MODULE) src.extractor.extractor


.PHONY: help
help:
	@echo "🛠️  Available commands:"
