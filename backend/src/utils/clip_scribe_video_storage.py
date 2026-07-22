"""Video storage seam: where source videos live before extraction.

This mirrors the :mod:`src.utils.clip_scribe_artifacts` upload seam, but for the
*input* side. The core never learns whether a video lives on a local disk or in
a cloud bucket — it always receives a plain local filesystem path from
:meth:`VideoStorage.materialize` and hands it to OpenCV / SAM / Whisper, which
all need a real seekable file.

The single abstraction has two halves:

* **Ingest** (API): a video is *staged* to a temp file (so its bytes can be
  hashed for dedup), then *committed* to permanent storage under an opaque key.
  Callers treat the key as opaque: the local backend uses a bare
  ``01J8XZ....mp4`` filename; the GCS backend uses a tenant-isolating
  ``videos/<user_id>/01J8XZ....mp4`` object name.
* **Materialize** (worker): given a key, produce a local path the extractor can
  open, and later *release* it. For the local backend the key already *is* a
  local file, so materialize is a lookup and release is a no-op — it must never
  delete the user's source. For a cloud backend materialize downloads to a
  scratch file and release deletes that scratch copy. That asymmetry (no-op vs
  delete) is the one sharp edge here, so ``release`` is only ever handed a path
  returned by ``materialize``.

Selected by ``CLIPSCRIBE_STORAGE_BACKEND`` (``local`` or ``gcs``), the single
selector that also governs artifact storage — the same way
``CLIPSCRIBE_DB_BACKEND`` selects the database.
"""

from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utils.ids import new_ulid

if TYPE_CHECKING:
    from google.cloud import storage

logger = logging.getLogger("clip_scribe")

# How long a signed GET URL stays valid. Kept short: the browser fetches a fresh
# one every time the run inspector mounts (the <video> points at our API route,
# not the signed URL), so the only stale case is a page left open past this
# window — a refresh reissues it. See src/utils/clip_scribe_artifacts.py.
SIGNED_URL_TTL = timedelta(minutes=15)


def _make_gcs_client() -> "storage.Client":  # pragma: no cover - thin SDK wrapper
    """Create a real GCS client. Isolated so tests can monkeypatch it.

    Credentials come from ``GOOGLE_APPLICATION_CREDENTIALS`` (the local
    ``service_account.json`` in dev) or the worker's attached identity in prod.
    """
    from google.cloud import storage

    return storage.Client()


class VideoStorage(ABC):
    """Stores source videos and materializes them locally for extraction."""

    @abstractmethod
    def staging_dir(self) -> Path:
        """A local directory the caller may write temp upload files into.

        Kept on the same filesystem as committed storage (local backend) so
        :meth:`commit` is a cheap rename rather than a copy.
        """

    @abstractmethod
    def commit(self, user_id: str, staged: Path, suffix: str) -> str:
        """Promote a staged temp file to permanent storage; return its key.

        ``staged`` is a path previously created under :meth:`staging_dir` and
        fully written by the caller. On return the staged file has been consumed
        (moved/uploaded); the caller must not use it again.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether ``key`` currently resolves to a stored object.

        Used by the input reconcile: a key whose object has been removed
        out-of-band (bucket cleanup, dev deletion) returns False.
        """

    @abstractmethod
    def materialize(self, key: str) -> Path:
        """Return a local filesystem path for ``key`` the extractor can open."""

    @abstractmethod
    def release(self, local: Path) -> None:
        """Release a path returned by :meth:`materialize`.

        No-op when the path is the permanent object itself (local backend);
        deletes the scratch copy for backends that download on materialize.
        """

    @abstractmethod
    def signed_url(self, key: str) -> str | None:
        """A time-limited URL a browser can GET the object from directly.

        ``None`` when the backend serves bytes locally — the caller streams a
        ``FileResponse`` instead. A cloud backend returns a signed URL so the
        API can 302-redirect and the browser streams straight from the bucket
        (native HTTP Range → video seeking), never proxying media through us.
        """


class LocalVideoStorage(VideoStorage):
    """Videos live as files under a single local root (``INPUT_DIR``).

    Keys are bare ``<ulid><suffix>`` filenames under ``root``. ``user_id`` is
    recorded in the registry table but not (yet) reflected in the on-disk
    layout — the local dev backend is single-tenant.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def staging_dir(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root

    def commit(self, user_id: str, staged: Path, suffix: str) -> str:
        key = f"{new_ulid()}{suffix}"
        dest = self.root / key
        # Same-filesystem rename: atomic and cheap (staged came from staging_dir).
        shutil.move(str(staged), str(dest))
        logger.info("Stored video %s (%d bytes)", key, dest.stat().st_size)
        return key

    def exists(self, key: str) -> bool:
        return (self.root / key).is_file()

    def materialize(self, key: str) -> Path:
        return self.root / key

    def release(self, local: Path) -> None:
        # The materialized path IS the stored file; never delete the source.
        return None

    def signed_url(self, key: str) -> str | None:
        # Bytes live on the API's own disk; the caller streams a FileResponse.
        return None


class GCSVideoStorage(VideoStorage):
    """Videos in a GCS bucket under the ``videos/`` prefix.

    Keys are the full object name ``videos/<user_id>/<ulid><suffix>`` (opaque to
    callers, tenant-isolated by ``user_id`` for per-user dedup). The API and
    worker need not share a filesystem: ``commit`` uploads, ``materialize``
    downloads a scratch copy the extractor opens, and ``release`` deletes it.

    ``local_root`` is reused as the on-disk scratch space for both staging
    (upload hashing) and materialization (download targets).
    """

    _PREFIX = "videos"

    def __init__(
        self, bucket: str, local_root: Path, *, client: "storage.Client | None" = None
    ) -> None:
        self._client = client or _make_gcs_client()
        self._bucket_name = bucket
        self._bucket: Any = self._client.bucket(bucket)
        self._staging_root = local_root
        # Downloads land here (kept out of the local backend's flat key space).
        self._scratch_root = local_root / ".gcs_scratch"

    def staging_dir(self) -> Path:
        self._staging_root.mkdir(parents=True, exist_ok=True)
        return self._staging_root

    def commit(self, user_id: str, staged: Path, suffix: str) -> str:
        key = f"{self._PREFIX}/{user_id}/{new_ulid()}{suffix}"
        self._bucket.blob(key).upload_from_filename(str(staged))
        staged.unlink(missing_ok=True)
        logger.info("Uploaded video to gs://%s/%s", self._bucket_name, key)
        return key

    def exists(self, key: str) -> bool:
        return bool(self._bucket.blob(key).exists())

    def materialize(self, key: str) -> Path:
        self._scratch_root.mkdir(parents=True, exist_ok=True)
        # Preserve the suffix so OpenCV/Whisper still sniff the format by name.
        local = self._scratch_root / f"{new_ulid()}{Path(key).suffix}"
        self._bucket.blob(key).download_to_filename(str(local))
        return local

    def release(self, local: Path) -> None:
        # Delete only the downloaded scratch copy, never the bucket object.
        local.unlink(missing_ok=True)

    def signed_url(self, key: str) -> str | None:
        return str(
            self._bucket.blob(key).generate_signed_url(
                version="v4", expiration=SIGNED_URL_TTL, method="GET"
            )
        )


def make_video_storage(
    backend: str, local_root: Path, bucket: str | None = None
) -> VideoStorage:
    """Build the configured video storage backend.

    ``backend`` is the validated ``CLIPSCRIBE_STORAGE_BACKEND`` value;
    ``local_root`` is where the local backend stores files (and where the GCS
    backend stages/materializes); ``bucket`` is required when ``backend`` is
    ``gcs``.
    """
    if backend == "local":
        return LocalVideoStorage(local_root)
    if backend == "gcs":
        if not bucket:
            raise ValueError(
                "gcs video storage requires a bucket (CLIPSCRIBE_GCS_BUCKET)"
            )
        return GCSVideoStorage(bucket, local_root)
    raise ValueError(f"unknown video storage backend: {backend!r}")
