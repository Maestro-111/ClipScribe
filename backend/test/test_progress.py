"""Tests for the ProgressReporter seam (docs/web-app-plan.md §5, step 2).

The pipeline itself needs models to run end-to-end, so we don't exercise it
here. Instead we test the abstraction's contract: the Null reporter is a true
no-op, the phase helpers route through ``emit`` with the right payload, and a
recording subclass observes events in the order they are emitted.
"""

from typing import Any, Mapping

import pytest

from src.utils.progress import (
    NullProgressReporter,
    Phase,
    ProgressEvent,
    ProgressReporter,
)


class RecordingProgressReporter(ProgressReporter):
    """Captures every emitted (event_type, data) pair in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, event_type: str, data: Mapping[str, Any] | None = None) -> None:
        self.events.append((event_type, dict(data or {})))


def test_null_reporter_is_a_noop() -> None:
    reporter = NullProgressReporter()
    assert reporter.emit(ProgressEvent.JOB_STARTED) is None
    assert reporter.emit(ProgressEvent.SHOT_STARTED, {"shot_idx": 0}) is None
    # Phase helpers must also be safe no-ops.
    assert reporter.phase_started(Phase.AUDIO) is None
    assert reporter.phase_completed(Phase.AUDIO, {"segments_kept": 3}) is None


def test_abstract_reporter_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        ProgressReporter()  # type: ignore[abstract]


def test_phase_helpers_build_expected_payload() -> None:
    reporter = RecordingProgressReporter()

    reporter.phase_started(Phase.SHOT_PROCESSING, {"total_shots": 12})
    reporter.phase_completed(Phase.SHOT_PROCESSING)

    assert reporter.events == [
        (ProgressEvent.PHASE_STARTED, {"phase": "shot_processing", "total_shots": 12}),
        (ProgressEvent.PHASE_COMPLETED, {"phase": "shot_processing"}),
    ]


def test_events_recorded_in_emit_order() -> None:
    reporter = RecordingProgressReporter()

    reporter.emit(ProgressEvent.JOB_STARTED, {"mode": "full"})
    reporter.phase_started(Phase.SCENE_DETECTION)
    reporter.phase_completed(Phase.SCENE_DETECTION, {"total_shots": 2})
    reporter.emit(ProgressEvent.SHOT_STARTED, {"shot_idx": 0})
    reporter.emit(ProgressEvent.SHOT_COMPLETED, {"shot_idx": 0})
    reporter.emit(ProgressEvent.JOB_COMPLETED, {"run_id": "abc"})

    ordered_types = [event_type for event_type, _ in reporter.events]
    assert ordered_types == [
        ProgressEvent.JOB_STARTED,
        ProgressEvent.PHASE_STARTED,
        ProgressEvent.PHASE_COMPLETED,
        ProgressEvent.SHOT_STARTED,
        ProgressEvent.SHOT_COMPLETED,
        ProgressEvent.JOB_COMPLETED,
    ]
