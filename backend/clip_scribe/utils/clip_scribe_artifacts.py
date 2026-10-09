"""Artifact directory convention and the remote-storage seam.

Per-run artifacts (the tracked mp4, the per-frame visualization PNGs, and
``extraction_summary.json``) are written under ``artifacts/<run_id>/`` — keyed
by run id rather than video name so two jobs over the same video never collide
(see docs/web-app-plan.md §9, §15).

Remote storage mirrors the video-storage seam: the core depends only on the
:class:`ArtifactUploader` abstraction. The backend is chosen by the single
``CLIPSCRIBE_STORAGE_BACKEND`` selector (the same one that selects video
storage): ``local`` -> a no-op uploader (artifacts stay on disk, served by
``FileResponse``); ``gcs`` -> :class:`GCSArtifactUploader`, which uploads the
run bundle at the end of the pipeline and mints signed URLs for the one file
the frontend fetches, ``tracked_output.mp4``.

Only the frontend's live dependency (``tracked_output.mp4``) is uploaded as a
loose, directly-servable object; the debug PNGs and prompt dumps — which nothing
consumes over HTTP — are bundled into a single ``artifacts.tar.gz`` for archival.
"""

from __future__ import annotations

import logging
import tarfile
import tempfile
from abc import ABC, abstractmethod
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from google.cloud import storage

logger = logging.getLogger("clip_scribe")

# The one artifact the frontend fetches; uploaded loose so it can be signed and
# streamed directly. Everything else is archived into the bundle below.
TRACKED_VIDEO_NAME = "tracked_output.mp4"
ARTIFACT_BUNDLE_NAME = "artifacts.tar.gz"

# Signed-URL lifetime, matching the video-storage seam (short by design; the
# frontend refetches on every inspector mount). See clip_scribe_video_storage.
SIGNED_URL_TTL = timedelta(minutes=15)


def run_artifact_dir(run_id: str) -> str:
    """The artifact directory for a run. Single source of truth for the path."""
    return f"artifacts/{run_id}"


def _make_gcs_client() -> "storage.Client":  # pragma: no cover - thin SDK wrapper
    """Create a real GCS client. Isolated so tests can monkeypatch it."""
    from google.cloud import storage

    return storage.Client()


class ArtifactUploader(ABC):
    """Stores a finished run's artifacts and serves the ones the UI needs."""

    @abstractmethod
    def upload_run_artifacts(self, run_id: str, artifact_dir: str) -> None:
        """Upload the run's artifacts. Best-effort — must not raise."""

    @abstractmethod
    def tracked_video_url(self, run_id: str) -> str | None:
        """A signed URL for the run's ``tracked_output.mp4``, or ``None``.

        ``None`` when the artifact is served from local disk (the caller streams
        a ``FileResponse`` instead); a signed URL for cloud backends so the API
        can 302-redirect and the browser streams straight from the bucket.
        """

    @abstractmethod
    def delete_run_artifacts(self, run_id: str) -> None:
        """Delete stored artifacts for a run. Best-effort — must not raise."""


class NullArtifactUploader(ArtifactUploader):
    """No-op uploader: artifacts stay local only. The default (local backend)."""

    def upload_run_artifacts(self, run_id: str, artifact_dir: str) -> None:
        return None

    def tracked_video_url(self, run_id: str) -> str | None:
        return None

    def delete_run_artifacts(self, run_id: str) -> None:
        return None


class GCSArtifactUploader(ArtifactUploader):
    """Uploads run artifacts to a GCS bucket under the ``artifacts/`` prefix.

    Layout per run (``run_id`` is a globally-unique ULID, so no user prefix is
    needed — ownership is resolved via the DB, not the object path):

    * ``artifacts/<run_id>/tracked_output.mp4`` — loose, signed + streamed to the UI.
    * ``artifacts/<run_id>/artifacts.tar.gz``   — the debug PNGs + prompt dumps.
    """

    _PREFIX = "artifacts"

    def __init__(self, bucket: str, *, client: "storage.Client | None" = None) -> None:
        self._client = client or _make_gcs_client()
        self._bucket_name = bucket
        self._bucket: Any = self._client.bucket(bucket)

    def _blob_name(self, run_id: str, filename: str) -> str:
        return f"{self._PREFIX}/{run_id}/{filename}"

    def upload_run_artifacts(self, run_id: str, artifact_dir: str) -> None:
        # Best-effort: a storage hiccup must not fail an otherwise-good run.
        try:
            base = Path(artifact_dir)
            if not base.is_dir():
                logger.warning("Artifact dir %s missing; nothing to upload", base)
                return

            # 1. The tracked video, loose and directly servable to the frontend.
            tracked = base / TRACKED_VIDEO_NAME
            if tracked.is_file():
                self._bucket.blob(
                    self._blob_name(run_id, TRACKED_VIDEO_NAME)
                ).upload_from_filename(str(tracked))

            # 2. Everything else, bundled — nothing fetches these over HTTP.
            self._upload_bundle(run_id, base)

            logger.info(
                "Uploaded run %s artifacts to gs://%s/%s/",
                run_id,
                self._bucket_name,
                self._blob_name(run_id, "").rstrip("/"),
            )
        except Exception:  # noqa: BLE001 - upload is best-effort, never fatal
            logger.warning(
                "Failed to upload artifacts for run %s", run_id, exc_info=True
            )

    def _upload_bundle(self, run_id: str, base: Path) -> None:
        """Tar every artifact except the tracked video and upload the bundle."""
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tar_path = Path(tmp.name)
        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                for path in sorted(base.iterdir()):
                    if path.is_file() and path.name != TRACKED_VIDEO_NAME:
                        tar.add(str(path), arcname=path.name)
            self._bucket.blob(
                self._blob_name(run_id, ARTIFACT_BUNDLE_NAME)
            ).upload_from_filename(str(tar_path))
        finally:
            tar_path.unlink(missing_ok=True)

    def tracked_video_url(self, run_id: str) -> str | None:
        blob = self._bucket.blob(self._blob_name(run_id, TRACKED_VIDEO_NAME))
        if not blob.exists():
            return None
        return str(
            blob.generate_signed_url(
                version="v4", expiration=SIGNED_URL_TTL, method="GET"
            )
        )

    def delete_run_artifacts(self, run_id: str) -> None:
        try:
            prefix = f"{self._PREFIX}/{run_id}/"
            for blob in self._client.list_blobs(self._bucket_name, prefix=prefix):
                blob.delete()
            logger.info(
                "Deleted run %s artifacts from gs://%s/%s",
                run_id,
                self._bucket_name,
                prefix,
            )
        except Exception:
            logger.warning(
                "Failed to delete artifacts for run %s", run_id, exc_info=True
            )


def make_artifact_uploader(backend: str, bucket: str | None = None) -> ArtifactUploader:
    """Build the artifact uploader for the single ``CLIPSCRIBE_STORAGE_BACKEND``.

    ``local`` (and any non-gcs) -> :class:`NullArtifactUploader`; ``gcs`` ->
    :class:`GCSArtifactUploader`, which requires ``bucket``.
    """
    if backend == "gcs":
        if not bucket:
            raise ValueError(
                "gcs artifact storage requires a bucket (CLIPSCRIBE_GCS_BUCKET)"
            )
        return GCSArtifactUploader(bucket)
    return NullArtifactUploader()
