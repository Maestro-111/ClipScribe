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

from src.db.engine import ensure_sqlite_parent_directory

BACKEND = Path(__file__).resolve().parents[1]


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
            clib_scribe_platform_name="youtube",
            clib_scribe_platform_conf=mock.MagicMock(),
        )
        print(f"\nmode={mode!r} -> per-job heavy load calls={heavy.call_count}")
        assert heavy.call_count == 0, "build_clip_scribe must not reload heavy models"


def test_parser_report_path_stays_under_output_dir(tmp_path):
    from src.parser.parser_core import VideoInformationParser
    from src.utils.progress import NullProgressReporter

    class DummyEvaluator:
        def evaluate_all(self, run_id, video_name):
            return []

    class DummyReportWriter:
        def __init__(self, report_output_path, scores_output_path):
            self.report_output_path = report_output_path
            self.scores_output_path = scores_output_path

        def write_results(self, results):
            self.report_output_path.parent.mkdir(parents=True, exist_ok=True)
            self.report_output_path.write_text("")
            self.scores_output_path.write_text("")

    class DummyParser(VideoInformationParser):
        def __init__(self, output_dir):
            self.platform_name = "youtube"
            self.platform_config = mock.MagicMock()
            self.output_dir = str(output_dir)
            self.max_parallel_agents = 1
            self.recursion_limit = 1
            self.model = mock.MagicMock()
            self.progress = NullProgressReporter()

        def create_report_name(self):
            return "abcd"

        def create_evaluator(self, reader_db):
            return DummyEvaluator()

        def create_report_writer(self, output_path, scores_output_path):
            return DummyReportWriter(output_path, scores_output_path)

    output_dir = tmp_path / "parser_artifacts"
    parser = DummyParser(output_dir)
    writer_db = mock.MagicMock()

    report_path = Path(
        parser.parse(
            run_id="../outside-run",
            video_name="../outside-video",
            reader_db=mock.MagicMock(),
            writer_db=writer_db,
        )
    )

    report_path.resolve().relative_to(output_dir.resolve())
    assert report_path.parent.name == "outside-run"
    assert not (tmp_path / "outside-video").exists()
    writer_db.save_parser_results.assert_called_once()


def test_alembic_baseline_accepts_existing_runs_table(tmp_path, monkeypatch):
    import sqlite3

    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    db_path = tmp_path / "adopted.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE runs (
                run_id TEXT NOT NULL,
                video_name TEXT,
                video_path TEXT,
                video_type TEXT,
                created_at TEXT DEFAULT (CURRENT_TIMESTAMP),
                PRIMARY KEY (run_id)
            )
            """
        )

    monkeypatch.setenv("SQLITE_URL", f"sqlite:////{db_path}")
    monkeypatch.chdir(BACKEND)
    config = Config(str(BACKEND / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND / "alembic"))

    command.upgrade(config, "head")

    engine = create_engine(f"sqlite:///{db_path}")
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    assert {"runs", "jobs", "frame_detections", "alembic_version"} <= tables
