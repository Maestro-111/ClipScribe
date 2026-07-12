"""Progress reporting seam for the extraction / parse pipeline.

The pipeline emits structured progress events at fixed points (see
``docs/web-app-plan.md`` §5). The core — extractor, parser, engine — depends
only on the abstract :class:`ProgressReporter`, never on Redis or the web
layer. The web execution paths inject ``app.events.RedisProgressReporter`` to
publish into a per-job Redis Stream, while the CLI (``main.py``), tests, and
Redis fallback use :class:`NullProgressReporter`, which does nothing.

Keeping the interface here in ``src/utils`` (rather than in ``app/``) preserves
the dependency direction: the core never imports the web layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping


class ProgressEvent:
    """String identifiers for the progress event vocabulary (§5).

    Used as the ``event_type`` argument to :meth:`ProgressReporter.emit` so
    call sites stay free of stray string literals.
    """

    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"

    PHASE_STARTED = "phase.started"
    PHASE_COMPLETED = "phase.completed"

    AUDIO_SEGMENT = "audio.segment"

    SHOT_STARTED = "shot.started"
    SHOT_SCENE_DESCRIBED = "shot.scene_described"
    SHOT_TAXONOMY_RESOLVED = "shot.taxonomy_resolved"
    SHOT_FRAME_PROCESSED = "shot.frame_processed"
    SHOT_COMPLETED = "shot.completed"

    IDENTITY_MERGED = "identity.merged"

    PARSER_CRITERION_STARTED = "parser.criterion_started"
    PARSER_CRITERION_COMPLETED = "parser.criterion_completed"


class Phase:
    """Canonical phase names, the value of the ``phase`` key on phase events."""

    SCENE_DETECTION = "scene_detection"
    AUDIO = "audio"
    SHOT_PROCESSING = "shot_processing"
    FINALIZE = "finalize"
    PARSE = "parse"


class ProgressReporter(ABC):
    """Sink for structured pipeline progress events.

    Implementations must be cheap and must never raise — a failure to report
    progress must not break the pipeline. The default phase helpers are
    concrete and route through :meth:`emit`.
    """

    @abstractmethod
    def emit(self, event_type: str, data: Mapping[str, Any] | None = None) -> None:
        """Emit a single progress event.

        Args:
            event_type: One of the :class:`ProgressEvent` identifiers.
            data: Optional event payload (JSON-serializable values).
        """

    def phase_started(self, phase: str, data: Mapping[str, Any] | None = None) -> None:
        self._phase_event(ProgressEvent.PHASE_STARTED, phase, data)

    def phase_completed(
        self, phase: str, data: Mapping[str, Any] | None = None
    ) -> None:
        self._phase_event(ProgressEvent.PHASE_COMPLETED, phase, data)

    def _phase_event(
        self, event_type: str, phase: str, data: Mapping[str, Any] | None
    ) -> None:
        payload: dict[str, Any] = {"phase": phase}
        if data:
            payload.update(data)
        self.emit(event_type, payload)


class NullProgressReporter(ProgressReporter):
    """No-op reporter used by the CLI (just main.py) and tests; emits nothing."""

    def emit(self, event_type: str, data: Mapping[str, Any] | None = None) -> None:
        return None
