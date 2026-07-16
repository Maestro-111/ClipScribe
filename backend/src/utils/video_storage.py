"""Video storage seam: where source videos live before extraction.

This mirrors the :mod:`src.utils.artifacts` upload seam, but for the *input*
side. The core never learns whether a video lives on a local disk or in a cloud
bucket — it always receives a plain local filesystem path from
:meth:`VideoStorage.materialize` and hands it to OpenCV / SAM / Whisper, which
all need a real seekable file.

The single abstraction has two halves:

* **Ingest** (API): a video is *staged* to a temp file (so its bytes can be
  hashed for dedup), then *committed* to permanent storage under an opaque key
  like ``01J8XZ....mp4``. Callers treat the key as opaque — the ``user_/``
  prefixing that isolates tenants is a backend detail, introduced when the GCS
  backend lands.
* **Materialize** (worker): given a key, produce a local path the extractor can
  open, and later *release* it. For the local backend the key already *is* a
  local file, so materialize is a lookup and release is a no-op — it must never
  delete the user's source. For a cloud backend materialize downloads to a
  scratch file and release deletes that scratch copy. That asymmetry (no-op vs
  delete) is the one sharp edge here, so ``release`` is only ever handed a path
  returned by ``materialize``.

Selected by ``CLIPSCRIBE_VIDEO_STORAGE`` (``local`` today; ``gcs`` reserved),
the same way ``CLIPSCRIBE_DB_BACKEND`` selects the database.
"""

from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from src.utils.ids import new_ulid

logger = logging.getLogger("clip_scribe")


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


class GCSVideoStorage(VideoStorage):
    """Reserved backend for videos in a GCS bucket.

    Intentionally unimplemented: constructing it fails fast at process startup
    so a misconfigured ``CLIPSCRIBE_VIDEO_STORAGE=gcs`` is caught immediately
    rather than at first upload. The method bodies below document the contract
    a real implementation must satisfy — it is a drop-in replacement, nothing
    else in the codebase changes:

    * ``staging_dir`` -> a local temp dir bytes are streamed into for hashing.
    * ``commit``      -> upload ``staged`` to ``gs://<bucket>/<user_id>/<ulid><suffix>``,
                         delete the staged file, return ``<user_id>/<ulid><suffix>``.
    * ``exists``      -> ``blob(key).exists()``.
    * ``materialize`` -> download the blob to a scratch file, return its path.
    * ``release``     -> delete that scratch file.
    """

    def __init__(self, bucket: str = "clipscribe-videos") -> None:
        raise NotImplementedError(
            "GCS video storage is not implemented yet. Set CLIPSCRIBE_VIDEO_STORAGE=local "
            "or implement GCSVideoStorage (see its docstring for the contract)."
        )

    def staging_dir(self) -> Path:  # pragma: no cover - unreachable stub
        raise NotImplementedError

    def commit(
        self, user_id: str, staged: Path, suffix: str
    ) -> str:  # pragma: no cover - unreachable stub
        raise NotImplementedError

    def exists(self, key: str) -> bool:  # pragma: no cover - unreachable stub
        raise NotImplementedError

    def materialize(self, key: str) -> Path:  # pragma: no cover - unreachable stub
        raise NotImplementedError

    def release(self, local: Path) -> None:  # pragma: no cover - unreachable stub
        raise NotImplementedError


def make_video_storage(backend: str, local_root: Path) -> VideoStorage:
    """Build the configured video storage backend.

    ``backend`` is the validated ``CLIPSCRIBE_VIDEO_STORAGE`` value; ``local_root``
    is where the local backend stores files (``settings.input_dir``).
    """
    if backend == "local":
        return LocalVideoStorage(local_root)
    if backend == "gcs":
        return GCSVideoStorage()
    raise ValueError(f"unknown video storage backend: {backend!r}")
