"""Artifact directory convention and the remote-upload seam.

Per-run artifacts (the tracked mp4, the per-frame visualization PNGs, and
``extraction_summary.json``) are written under ``artifacts/<run_id>/`` — keyed
by run id rather than video name so two jobs over the same video never collide
(see docs/web-app-plan.md §9, §15).

Remote upload mirrors the ProgressReporter pattern: the core depends only on
the :class:`ArtifactUploader` abstraction. ``remote_artifact_write: false``
(the default) gives a no-op uploader; ``true`` gives a simulated GCS uploader
that *logs* the bundle it would push. The real GCS implementation is a
drop-in replacement for :class:`SimulatedGCSArtifactUploader` later — flip the
flag, swap the body, nothing else changes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger("clip_scribe")


def run_artifact_dir(run_id: str) -> str:
    """The artifact directory for a run. Single source of truth for the path."""
    return f"artifacts/{run_id}"


class ArtifactUploader(ABC):
    """Pushes a finished run's local artifact directory to remote storage."""

    @abstractmethod
    def upload_run_artifacts(self, run_id: str, artifact_dir: str) -> None:
        """Upload the run's artifacts as a single bundle. Must not raise."""


class NullArtifactUploader(ArtifactUploader):
    """No-op uploader: artifacts stay local only. The default."""

    def upload_run_artifacts(self, run_id: str, artifact_dir: str) -> None:
        return None


class SimulatedGCSArtifactUploader(ArtifactUploader):
    """Stand-in for the eventual GCS uploader.

    Logs the single bundle upload it *would* perform at the end of the
    pipeline, so the call site and timing can be exercised end-to-end before
    any real cloud code or ``google-cloud-storage`` dependency exists. Swap the
    body of :meth:`upload_run_artifacts` for a real tar + GCS upload later.
    """

    def __init__(self, bucket: str = "clipscribe-artifacts") -> None:
        self.bucket = bucket

    def upload_run_artifacts(self, run_id: str, artifact_dir: str) -> None:
        logger.info(
            "[remote_artifact_write] would bundle %r and upload to "
            "gs://%s/%s/artifacts.tar.gz",
            artifact_dir,
            self.bucket,
            run_id,
        )
