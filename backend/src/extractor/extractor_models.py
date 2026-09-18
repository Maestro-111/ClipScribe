"""Structured types describing extractor inputs, registries, and output payloads.

These shapes are the contract between the extractor and its consumers (the
engine, the DB writer, and the persisted ``extraction_summary.json``). Keeping
them here lets callers depend on the schema without importing the extractor's
heavy model stack.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    NotRequired,
    Protocol,
    Sequence,
    TypeAlias,
    TypedDict,
)

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch

Box: TypeAlias = Sequence[float]
Embedding: TypeAlias = npt.NDArray[np.floating[Any]]


class ReIDModel(Protocol):
    """Backbone used to embed crops for cross-shot identity resolution."""

    def forward_features(self, x: torch.Tensor) -> Mapping[str, torch.Tensor]:
        ...


class TrackerData(TypedDict):
    box: Box
    label: str


class OCRDetection(TypedDict):
    box: Box
    text: str
    confidence: float
    label: NotRequired[str]


class DetectionResult(TypedDict):
    box: Box
    label: str
    score: float


class ObjectRegistryEntry(TypedDict):
    label: str
    shot_id: int
    embedding_sum: Embedding
    embedding_count: int
    boxes: list[Box]
    timestamps: list[float]
    last_embedding_frame: float


class VisualObjectOccurrence(TypedDict):
    shot_index: int
    lifespan: list[float]
    screen_coverage: float
    velocity_px_sec: float
    growth_factor: float
    direction: str
    centrality_score: float
    screen_time_ratio: float
    quadrant: str


class VisualObjectSummary(TypedDict):
    global_id: int
    label: str
    occurrences: list[VisualObjectOccurrence]


class TextEvent(TypedDict):
    second: int
    text: list[str]


class AudioSegment(TypedDict):
    start: float
    end: float
    text: str
    confidence: float


class SceneDescriptionRecord(TypedDict):
    shot_index: int
    start_time: float
    end_time: float
    description: str


class ShotData(TypedDict):
    index: int
    start: float
    end: float
    duration: float


class RapidFireInterval(TypedDict):
    start_time: float
    end_time: float
    shot_count: int
    duration: float
    shot_indices: list[int]


class FrameDetection(TypedDict):
    shot_index: int
    frame_idx: int
    timestamp_sec: float
    source: str  # dino | ocr | mtcnn | sam_mask
    label: str | None
    text: str | None
    box_x1: float
    box_y1: float
    box_x2: float
    box_y2: float
    confidence: float | None
    object_id: int | None


class ExtractionSummary(TypedDict):
    global_stats: dict[str, object]
    visual_objects: list[VisualObjectSummary]
    text_events: list[TextEvent]
    audio_segments: list[AudioSegment]
    scene_descriptions: list[SceneDescriptionRecord]
    shot_boundaries: list[ShotData]
    frame_detections: list[FrameDetection]
