"""Fetch every model weight ClipScribe's pipeline needs into ``backend/checkpoints/``.

This is the single source of truth for model prefetch, shared by the worker
container and local setup. It mirrors what the root ``Makefile`` targets do, but
in one Python process so a Docker image can run it as a build/prewarm step.

Why the weights land in different ways:

- **GroundingDINO / SAM2** — explicit ``.pth``/``.pt`` files loaded by path.
  Downloaded here (DINO reuses ``scripts/download_dino.py``; SAM2 tiny+small are
  fetched directly, the same URLs as ``scripts/download_sam_ckpts.sh``). Both
  skip if the file already exists, so in dev — where ``backend/checkpoints`` is
  volume-mounted and already holds them — this is a no-op.
- **DINOv2 / SBERT / NLTK / PaddleOCR** — auto-downloaders that pick their cache
  dir from an environment variable read at import time. We set those vars (below)
  to subdirectories of ``checkpoints/`` before importing the libraries. In the
  container the Dockerfile ``ENV`` block already sets them; ``setdefault`` here
  makes a standalone local run centralize too without overriding the container.
- **Whisper** — ignores those env vars and takes an explicit ``download_root``.
- **MTCNN (facenet-pytorch)** — weights ship inside the pip wheel; nothing to do.
- **spaCy ``en_core_web_sm``** — a pip wheel baked into the venv (Dockerfile /
  ``make spacy``), not a cache download; only verified here.

Usage::

    # container (env vars already set by the image):
    python scripts/prewarm.py
    # or via compose into the mounted checkpoints volume:
    docker compose run --rm prewarm python scripts/prewarm.py --force
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/  (this file is backend/scripts/prewarm.py)
BACKEND_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = BACKEND_ROOT / "checkpoints"

# Point every env-var-driven downloader at a checkpoints/ subdir BEFORE the
# libraries that read these vars are imported. setdefault: a value already in the
# environment (Dockerfile ENV, the user's shell) wins.
os.environ.setdefault("TORCH_HOME", str(CHECKPOINTS / "torch_hub"))
os.environ.setdefault("HF_HOME", str(CHECKPOINTS / "huggingface"))
os.environ.setdefault("NLTK_DATA", str(CHECKPOINTS / "nltk"))
os.environ.setdefault("PADDLE_OCR_BASE_DIR", str(CHECKPOINTS / "paddleocr"))

# SAM2 checkpoints actually used by the default configs (see clip_scribe.yaml
# sam2.size). Kept in sync with scripts/download_sam_ckpts.sh; we fetch only the
# sizes the pipeline loads rather than all four.
SAM2_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM2_CHECKPOINTS = ("sam2.1_hiera_tiny.pt", "sam2.1_hiera_small.pt")

# The SBERT model TaxonomyResolver uses (matches Makefile `sentence_transformers`).
SBERT_MODEL = "all-MiniLM-L6-v2"


def _download(url: str, dest: Path, min_size_mb: float = 1.0) -> None:
    """Stream ``url`` to ``dest``, skipping a plausibly-complete existing file."""
    import requests

    if dest.exists() and dest.stat().st_size / (1024 * 1024) > min_size_mb:
        print(f"  {dest.name} already present, skipping.")
        return

    print(f"  downloading {dest.name} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    size_mb = dest.stat().st_size / (1024 * 1024)
    if size_mb < min_size_mb:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name} downloaded too small ({size_mb:.1f} MB)")
    print(f"  saved {dest.name} ({size_mb:.0f} MB)")


def prewarm_dino() -> None:
    print("[1/6] GroundingDINO weights")
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "download_dino.py")],
        check=True,
    )


def prewarm_sam2() -> None:
    print("[2/6] SAM2 weights")
    for name in SAM2_CHECKPOINTS:
        _download(f"{SAM2_BASE_URL}/{name}", CHECKPOINTS / name, min_size_mb=30)


def prewarm_dinov2() -> None:
    print("[3/6] DINOv2 (torch.hub)")
    import torch

    torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")


def prewarm_sbert() -> None:
    print(f"[4/6] Sentence-Transformer ({SBERT_MODEL})")
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(SBERT_MODEL)


def prewarm_nltk() -> None:
    print("[5/6] NLTK WordNet")
    import nltk

    # Download explicitly into the checkpoints nltk dir and fail LOUDLY if it
    # doesn't land. nltk.download returns False (no exception) on failure, so a
    # silent miss would otherwise let the .prewarm_complete marker be written
    # with wordnet absent — and the worker then crashes in _labels_match
    # (wn.synsets) mid-run.
    nltk_dir = CHECKPOINTS / "nltk"
    nltk_dir.mkdir(parents=True, exist_ok=True)
    if not nltk.download("wordnet", download_dir=str(nltk_dir)):
        raise RuntimeError("NLTK 'wordnet' download failed")


def prewarm_whisper_and_ocr() -> None:
    print("[6/6] Whisper + PaddleOCR")
    import whisper

    whisper.load_model("base", download_root=str(CHECKPOINTS / "whisper"))

    from paddleocr import PaddleOCR

    # Mirror the constructor in backend/src/ocr/paddle_wrapper.py so the same
    # det/rec/cls models get pulled into PADDLE_OCR_BASE_DIR.
    PaddleOCR(
        use_textline_orientation=True,
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
    )


def verify_spacy() -> None:
    try:
        import spacy

        spacy.load("en_core_web_sm")
        print("spaCy en_core_web_sm OK")
    except Exception:  # noqa: BLE001 - advisory only
        print(
            "WARNING: spaCy model 'en_core_web_sm' is not installed. "
            "Install the wheel (Dockerfile bakes it; locally run `make spacy`)."
        )


# Written once all weights are in place; lets the container's one-shot prewarm
# service (docker-compose) short-circuit on every subsequent `up` instead of
# re-loading each model to verify. Delete it (or pass --force) to re-run.
MARKER = CHECKPOINTS / ".prewarm_complete"


def main(force: bool = False) -> None:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if MARKER.exists() and not force:
        print(f"Already prewarmed ({MARKER} exists); skipping. Pass --force to re-run.")
        return

    print(f"Prewarming model weights into {CHECKPOINTS}")
    prewarm_dino()
    prewarm_sam2()
    prewarm_dinov2()
    prewarm_sbert()
    prewarm_nltk()
    prewarm_whisper_and_ocr()
    verify_spacy()
    MARKER.touch()
    print("prewarm complete")


if __name__ == "__main__":
    import sys

    main(force="--force" in sys.argv[1:])
