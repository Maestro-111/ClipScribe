"""Tests for the no-mistakes(review) fixes:

1. ``main.py`` typo: branded-product category strings were implicitly
   concatenated (``"A" "B"`` -> ``"AB"``). They must be separate entries.
2. SAM2 size config key: builder must read ``sam2.size`` (not ``sam2_size``)
   so the configured model size actually reaches ``build_sam2_video_predictor``.
3. Heavy model load gating: constructing the builder and running a parser-only
   build must NOT load the heavy extractor models; only ``extract``/``full``
   modes load them.
"""

import ast
from pathlib import Path
from unittest import mock

import pytest

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


def test_construction_does_not_load_heavy_models(patched_heavy):
    bcs = patched_heavy
    with mock.patch.object(
        bcs.ClipScribeBuilder, "_assemble_heavy_extractor_utils"
    ) as heavy:
        bcs.ClipScribeBuilder()
        print(f"\nheavy load calls after __init__: {heavy.call_count}")
        assert heavy.call_count == 0


@pytest.mark.parametrize(
    "mode,expect_heavy",
    [("parse", 0), ("extract", 1), ("full", 1)],
)
def test_heavy_load_gated_by_mode(patched_heavy, mode, expect_heavy):
    bcs = patched_heavy
    with mock.patch.object(
        bcs.ClipScribeBuilder, "_assemble_heavy_extractor_utils"
    ) as heavy, mock.patch.object(
        bcs.ClipScribeBuilder, "build_extractor", return_value=mock.MagicMock()
    ), mock.patch.object(
        bcs.ClipScribeBuilder, "build_parser", return_value=mock.MagicMock()
    ), mock.patch.object(bcs, "ClipScribeEngine", mock.MagicMock()):
        builder = bcs.ClipScribeBuilder()
        builder.build_clip_scribe(
            video_name="v.mp4",
            video_path="input/v.mp4",
            video_type="car ad",
            clib_scribe_mode=mode,
            clib_scribe_device="cpu",
            clib_scribe_platform_name="youtube",
            clib_scribe_platform_conf=mock.MagicMock(),
        )
        print(f"\nmode={mode!r} -> heavy load calls={heavy.call_count}")
        assert heavy.call_count == expect_heavy
