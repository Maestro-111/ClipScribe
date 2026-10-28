"""Deterministic exploratory inputs: run metrics, OCR text cards, shot context.

The OCR fixture is taken from a real run (a Jeep ad) to pin down how noisy
per-frame reads collapse into distinct text cards.
"""

from __future__ import annotations

from typing import Any

import pytest

from clip_scribe.parser.explore.metrics import (
    build_shot_context,
    build_text_cards,
    compute_run_metrics,
    run_duration,
)


def _summary(**overrides: Any) -> Any:
    summary: dict[str, Any] = {
        "global_stats": {"total_shots": 3, "video_duration": 12.0},
        "shot_boundaries": [
            {"index": 0, "start": 0.0, "end": 2.0, "duration": 2.0},
            {"index": 1, "start": 2.0, "end": 7.0, "duration": 5.0},
            {"index": 2, "start": 7.0, "end": 12.0, "duration": 5.0},
        ],
        "scene_descriptions": [
            {
                "shot_index": 0,
                "start_time": 0.0,
                "end_time": 2.0,
                "description": "logo",
            },
            {
                "shot_index": 2,
                "start_time": 7.0,
                "end_time": 12.0,
                "description": "car",
            },
        ],
        "audio_segments": [
            # Straddles the 2.0s cut but overlaps shot 1 more.
            {"start": 1.5, "end": 4.5, "text": "one two three", "confidence": 0.9},
            {"start": 8.0, "end": 9.5, "text": "four five six", "confidence": 0.9},
        ],
        "text_events": [
            {"second": 0, "text": ["Jeep.", "DRIVE ITHOME"]},
            {"second": 1, "text": ["Jeep", "DRIVE IT HOME", "DRIVE IT HOME", "x"]},
            {"second": 10, "text": ["Jeep", "HLIDAYEVENT"]},
            {"second": 11, "text": ["HOLIDAY EVENT", "HOLIDAY EVENT"]},
        ],
        "visual_objects": [],
        "frame_detections": [],
    }
    summary.update(overrides)
    return summary


def test_text_cards_merge_ocr_variants_and_split_spans():
    cards = build_text_cards(_summary()["text_events"], 12.0)

    assert [c.text for c in cards] == ["Jeep", "DRIVE IT HOME", "HOLIDAY EVENT"]
    jeep = cards[0]
    # Seen at 0-1 and again at 10: two spans, not one 0-11 block.
    assert [(s.start_sec, s.end_sec) for s in jeep.spans] == [(0, 2), (10, 11)]
    assert jeep.total_sec == 3.0
    # "HLIDAYEVENT" merges into the same card as "HOLIDAY EVENT".
    assert [(s.start_sec, s.end_sec) for s in cards[2].spans] == [(10, 12)]


def test_text_cards_keep_short_distinct_words_apart():
    cards = build_text_cards(
        [{"second": 0, "text": ["Jeep"]}, {"second": 1, "text": ["Deep"]}], 5.0
    )
    assert [c.text for c in cards] == ["Jeep", "Deep"]


def test_text_card_spans_clip_to_duration():
    cards = build_text_cards([{"second": 14, "text": ["END"]}], 14.5)
    assert cards[0].spans[0].end_sec == 14.5


def test_metrics_without_object_tracking():
    metrics = compute_run_metrics(_summary(), objects_tracked=False)

    assert metrics.duration_sec == 12.0
    assert metrics.shot_count == 3
    assert metrics.avg_shot_sec == 4.0
    assert metrics.cuts_per_minute == 10.0
    assert metrics.speech_coverage == pytest.approx(4.5 / 12, abs=1e-3)
    # 6 words over 4.5s of speech.
    assert metrics.words_per_minute == 80.0
    assert metrics.text_coverage == pytest.approx(4 / 12, abs=1e-3)
    # Faces/objects weren't detected at all: unknown, not zero.
    assert metrics.face_coverage is None
    assert metrics.objects == []


def test_metrics_with_objects_and_faces():
    summary = _summary(
        visual_objects=[
            {
                "global_id": 1,
                "label": "car",
                "occurrences": [
                    _occ(shot_index=1, lifespan=[2.0, 6.0], screen_coverage=0.4),
                    _occ(shot_index=2, lifespan=[7.0, 12.0], screen_coverage=0.6),
                ],
            },
            {
                "global_id": 2,
                "label": "car",
                # Overlaps car #1, so the union doesn't double-count it.
                "occurrences": [
                    _occ(shot_index=2, lifespan=[8.0, 10.0], screen_coverage=0.2)
                ],
            },
            {
                "global_id": 3,
                "label": "logo",
                "occurrences": [
                    _occ(shot_index=0, lifespan=[0.0, 2.0], screen_coverage=0.1)
                ],
            },
        ],
        frame_detections=[
            _face(0.5),
            _face(0.9),
            _face(3.2),
            {**_face(5.0), "source": "dino"},
        ],
    )
    metrics = compute_run_metrics(summary, objects_tracked=True)

    car, logo = metrics.objects
    assert (car.label, car.instance_count, car.screen_time_sec) == ("car", 2, 9.0)
    assert car.first_seen_sec == 2.0
    assert car.avg_screen_coverage == 0.4
    assert logo.label == "logo"
    # Faces in seconds 0 and 3 only.
    assert metrics.face_coverage == pytest.approx(2 / 12, abs=1e-3)


def test_metrics_without_speech():
    metrics = compute_run_metrics(_summary(audio_segments=[]), objects_tracked=False)
    assert metrics.speech_coverage == 0.0
    assert metrics.words_per_minute is None


def test_run_duration_falls_back_to_last_shot():
    assert run_duration(_summary(global_stats={})) == 12.0


def test_shot_context_aligns_modalities_per_shot():
    summary = _summary(
        visual_objects=[
            {
                "global_id": 1,
                "label": "car",
                "occurrences": [_occ(shot_index=2, lifespan=[7.0, 12.0])],
            }
        ]
    )
    cards = build_text_cards(summary["text_events"], 12.0)
    shots = build_shot_context(summary, cards)

    assert [s.shot_index for s in shots] == [0, 1, 2]
    assert shots[0].description == "logo"
    assert shots[1].description is None
    assert shots[0].on_screen_text == ["Jeep", "DRIVE IT HOME"]
    assert shots[2].on_screen_text == ["Jeep", "HOLIDAY EVENT"]
    # The straddling segment is quoted once, in the shot it overlaps most.
    assert shots[0].speech == ""
    assert shots[1].speech == "one two three"
    assert shots[2].speech == "four five six"
    assert shots[2].objects == ["car"]


def _occ(
    *, shot_index: int, lifespan: list[float], screen_coverage: float = 0.5
) -> dict[str, Any]:
    return {
        "shot_index": shot_index,
        "lifespan": lifespan,
        "screen_coverage": screen_coverage,
        "velocity_px_sec": 0.0,
        "growth_factor": 1.0,
        "direction": "static",
        "centrality_score": 0.5,
        "screen_time_ratio": 0.5,
        "quadrant": "center",
    }


def _face(timestamp_sec: float) -> dict[str, Any]:
    return {
        "shot_index": 0,
        "frame_idx": int(timestamp_sec * 30),
        "timestamp_sec": timestamp_sec,
        "source": "mtcnn",
        "label": "human face",
        "text": None,
        "box_x1": 0.0,
        "box_y1": 0.0,
        "box_x2": 1.0,
        "box_y2": 1.0,
        "confidence": 0.99,
        "object_id": None,
    }
