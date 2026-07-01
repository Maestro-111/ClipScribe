"""Tests for the no-mistakes(review) fixes:

1. ``main.py`` typo: branded-product category strings were implicitly
   concatenated (``"A" "B"`` -> ``"AB"``). They must be separate entries.
2. SAM2 size config key: builder must read ``sam2.size`` (not ``sam2_size``)
   so the configured model size actually reaches ``build_sam2_video_predictor``.
3. Heavy model load-once: constructing the builder loads the heavy extractor
   models exactly once (the worker load-once contract, web-app-plan §3);
   ``build_clip_scribe`` never reloads them per job, for any mode.
"""

import ast
from pathlib import Path
from unittest import mock

import pytest

from src.clip_scribe.engine import ClipScribeEngine
from src.db.engine import ensure_sqlite_parent_directory
from src.extractor.extractor_core import VideoInformationExtractor
from src.utils.artifacts import NullArtifactUploader
from src.utils.progress import ProgressEvent

BACKEND = Path(__file__).resolve().parents[1]
MAIN_PY = BACKEND / "main.py"


def _extract_list_literal(source_path: Path, dict_key: str, list_key: str) -> list[str]:
    """Pull a string-list literal out of a module without executing it."""
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == dict_key for t in node.targets
        ):
            assert isinstance(node.value, ast.Dict)
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant) and k.value == list_key:
                    return [ast.literal_eval(e) for e in v.elts]  # type: ignore[attr-defined]
    raise AssertionError(f"{dict_key}[{list_key!r}] not found in {source_path}")


class RecordingProgressReporter:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event_type, data=None):
        self.events.append((event_type, dict(data or {})))


class FailingExtractor:
    def __init__(self) -> None:
        self.cleaned = False

    def extract(self, **kwargs):
        raise RuntimeError("extract failed")

    def cleanup(self) -> None:
        self.cleaned = True


def test_main_branded_categories_are_not_concatenated():
    categories = _extract_list_literal(
        MAIN_PY, "platform_params", "youtube_branded_products_categories"
    )
    print("\nbranded_products_categories:")
    for c in categories:
        print(f"  - {c}")

    # "RAM HD Rebel Car" must survive as its own entry, not be glued onto a
    # neighbouring "...Truck" string by Python's implicit concatenation.
    assert "RAM HD Rebel Car" in categories
    assert "Jeep Wrangler Truck" in categories
    assert "Jeep Adventure Days Truck" in categories

    # No entry should be the tell-tale concatenated artifact.
    for c in categories:
        assert "TruckRAM" not in c, f"implicit-concat artifact survived: {c!r}"


def test_engine_reports_failed_when_extraction_fails():
    reporter = RecordingProgressReporter()
    extractor = FailingExtractor()
    engine = ClipScribeEngine(
        mode="extract",
        video_name="v.mp4",
        video_path="input/v.mp4",
        video_type="car ad",
        extractor=extractor,
        parser=None,
        reader_db=mock.MagicMock(),
        writer_db=mock.MagicMock(),
        progress_reporter=reporter,
        artifact_uploader=NullArtifactUploader(),
    )

    with pytest.raises(RuntimeError, match="extract failed"):
        engine.run()

    event_types = [event_type for event_type, _ in reporter.events]
    assert ProgressEvent.JOB_FAILED in event_types
    assert ProgressEvent.JOB_COMPLETED not in event_types
    assert extractor.cleaned is True


def test_frame_detections_use_final_global_object_ids():
    extractor = VideoInformationExtractor.__new__(VideoInformationExtractor)
    extractor.frame_detections = [
        {
            "shot_index": 0,
            "frame_idx": 1,
            "timestamp_sec": 0.1,
            "source": "sam_mask",
            "label": "car",
            "text": None,
            "box_x1": 1.0,
            "box_y1": 2.0,
            "box_x2": 3.0,
            "box_y2": 4.0,
            "confidence": None,
            "object_id": 7,
        },
        {
            "shot_index": 1,
            "frame_idx": 5,
            "timestamp_sec": 0.5,
            "source": "sam_mask",
            "label": "car",
            "text": None,
            "box_x1": 5.0,
            "box_y1": 6.0,
            "box_x2": 7.0,
            "box_y2": 8.0,
            "confidence": None,
            "object_id": 12,
        },
        {
            "shot_index": 1,
            "frame_idx": 5,
            "timestamp_sec": 0.5,
            "source": "dino",
            "label": "car",
            "text": None,
            "box_x1": 5.0,
            "box_y1": 6.0,
            "box_x2": 7.0,
            "box_y2": 8.0,
            "confidence": 0.9,
            "object_id": None,
        },
    ]

    detections = extractor._frame_detections_with_global_object_ids({7: 0, 12: 0})

    assert [d["object_id"] for d in detections] == [0, 0, None]
    assert [d["object_id"] for d in extractor.frame_detections] == [7, 12, None]


def test_sqlite_parent_directory_is_created(tmp_path):
    db_path = tmp_path / "nested" / "clip_scribe.db"

    ensure_sqlite_parent_directory(f"sqlite:///{db_path}")

    assert db_path.parent.is_dir()
    assert not db_path.exists()


@pytest.fixture
def patched_heavy(monkeypatch, tmp_path):
    """Stub every heavy dependency so the builder can be imported/constructed
    without loading torch hub / whisper / SAM2 / DINO weights, and route the
    sqlite DB to a throwaway file under pytest's tmp_path (absolute URL so the
    builder's relative-path rewrite leaves it alone)."""
    monkeypatch.setenv("SQLITE_URL", f"sqlite:////{tmp_path / 'test.db'}")
    import src.clip_scribe.build_clip_scribe as bcs

    for name in (
        "DinoDetector",
        "OCRSystem",
        "build_sam2_video_predictor",
        "MTCNN",
        "ProfilesPile",
        "TaxonomyResolver",
        "TaxonomyGenerator",
    ):
        monkeypatch.setattr(bcs, name, mock.MagicMock(), raising=True)
    monkeypatch.setattr(bcs.torch.hub, "load", mock.MagicMock())
    monkeypatch.setattr(bcs.whisper, "load_model", mock.MagicMock())
    return bcs


def test_sam2_size_key_flows_from_config(patched_heavy):
    bcs = patched_heavy
    builder = bcs.ClipScribeBuilder()

    # Simulate a non-default configured size to prove the *key* is honoured.
    builder.sam2_params = {"size": "large"}
    builder._assemble_heavy_extractor_utils()

    first_arg = bcs.build_sam2_video_predictor.call_args.args[0]
    print(f"\nbuild_sam2_video_predictor received size={first_arg!r}")
    assert first_arg == "large", "configured sam2.size did not reach the predictor"

    # And with the key absent it falls back to the documented default.
    bcs.build_sam2_video_predictor.reset_mock()
    builder.sam2_params = {}
    builder._assemble_heavy_extractor_utils()
    assert bcs.build_sam2_video_predictor.call_args.args[0] == "tiny"


def test_construction_loads_heavy_models_once(patched_heavy):
    """Heavy models load exactly once, at construction — the load-once
    contract a long-lived worker relies on (web-app-plan §3). __init__ pays
    the 30-60s model-load cost a single time so every later job amortizes it.
    """
    bcs = patched_heavy
    with mock.patch.object(
        bcs.ClipScribeBuilder, "_assemble_heavy_extractor_utils"
    ) as heavy:
        bcs.ClipScribeBuilder()
        print(f"\nheavy load calls after __init__: {heavy.call_count}")
        assert heavy.call_count == 1


@pytest.mark.parametrize("mode", ["parse", "extract", "full"])
def test_build_clip_scribe_does_not_reload_heavy_models(patched_heavy, mode):
    """Per-job builds never re-trigger the heavy load, for any mode.

    The models are loaded once at construction and shared by reference into
    each freshly-built extractor, so ``build_clip_scribe`` must add zero
    further heavy loads regardless of mode.
    """
    bcs = patched_heavy
    with mock.patch.object(
        bcs.ClipScribeBuilder, "_assemble_heavy_extractor_utils"
    ) as heavy, mock.patch.object(
        bcs.ClipScribeBuilder, "build_extractor", return_value=mock.MagicMock()
    ), mock.patch.object(
        bcs.ClipScribeBuilder, "build_parser", return_value=mock.MagicMock()
    ), mock.patch.object(bcs, "ClipScribeEngine", mock.MagicMock()):
        builder = bcs.ClipScribeBuilder()
        assert heavy.call_count == 1, "heavy models should load once at construction"

        heavy.reset_mock()
        builder.build_clip_scribe(
            video_name="v.mp4",
            video_path="input/v.mp4",
            video_type="car ad",
            clib_scribe_mode=mode,
            clib_scribe_device="cpu",
            clib_scribe_platform_name="youtube",
            clib_scribe_platform_conf=mock.MagicMock(),
        )
        print(f"\nmode={mode!r} -> per-job heavy load calls={heavy.call_count}")
        assert heavy.call_count == 0, "build_clip_scribe must not reload heavy models"
