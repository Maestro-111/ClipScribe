"""Focused tests for SAM2 checkpoint identity reconciliation."""

import os
from types import SimpleNamespace
from unittest import mock

import torch

from clip_scribe.extractor.extractor_core import VideoInformationExtractor


def _extractor_with_sam() -> tuple[VideoInformationExtractor, mock.MagicMock]:
    extractor = VideoInformationExtractor.__new__(VideoInformationExtractor)
    sam_model = mock.MagicMock()
    extractor.sam_model = sam_model
    extractor.sam_inference_state = {"num_frames": 20}
    extractor.sam_frame_offset = 10
    extractor.active_trackers = {}
    extractor._sam_registered_tracker_ids = set()
    extractor.id_to_label = {}
    extractor.iou_threshold = 0.5
    return extractor, sam_model


def test_checkpoint_projection_skips_sam_when_shot_has_no_registered_ids():
    extractor, sam_model = _extractor_with_sam()

    assert extractor._predict_registered_objects_at(12) == {}
    sam_model.propagate_in_video.assert_not_called()


def test_checkpoint_projection_restores_only_ids_with_non_empty_masks():
    extractor, sam_model = _extractor_with_sam()
    extractor._sam_registered_tracker_ids = {7, 8}
    extractor.id_to_label = {7: "car", 8: "person"}

    masks = torch.full((2, 1, 8, 8), -1.0)
    masks[0, 0, 1:5, 2:6] = 1.0
    sam_model.propagate_in_video.return_value = iter([(2, [7, 8], masks)])

    visible = extractor._predict_registered_objects_at(12)

    assert visible == {7: {"box": [2, 1, 5, 4], "label": "car"}}
    sam_model.propagate_in_video.assert_called_once_with(
        extractor.sam_inference_state,
        start_frame_idx=2,
        max_frame_num_to_track=0,
    )

    extractor.active_trackers = visible
    assert not extractor._is_new_object([2, 1, 5, 4], "car")
    assert extractor._is_new_object([0, 0, 1, 1], "car")


def test_new_tracker_is_mirrored_as_registered_only_after_sam_accepts_it():
    extractor, sam_model = _extractor_with_sam()
    extractor.obj_id_counter = 1
    extractor.current_frame = 12

    extractor._add_new_tracker([2, 1, 5, 4], "car")

    assert extractor._sam_registered_tracker_ids == {1}
    assert extractor.active_trackers == {1: {"box": [2, 1, 5, 4], "label": "car"}}
    sam_model.add_new_points_or_box.assert_called_once_with(
        inference_state=extractor.sam_inference_state,
        frame_idx=2,
        obj_id=1,
        box=[2, 1, 5, 4],
    )


def test_shot_state_uses_temporary_video_and_records_frame_offset():
    extractor, sam_model = _extractor_with_sam()
    extractor.sam_inference_state = None
    extractor.sam_frame_offset = None
    extractor.device = "cpu"
    sam_model.init_state.return_value = {"num_frames": 10}

    with mock.patch.object(extractor, "_write_video_chunk") as write_video_chunk:
        extractor._init_sam_inference_state_for_shot("source.mp4", 30, 40)

    assert extractor.sam_inference_state == {"num_frames": 10}
    assert extractor.sam_frame_offset == 30
    assert extractor._to_sam_frame(34) == 4
    assert extractor._to_video_frame(4) == 34

    output_path = write_video_chunk.call_args.args[3]
    assert os.path.basename(output_path) == "shot.mp4"
    write_video_chunk.assert_called_once_with("source.mp4", 30, 40, output_path)
    sam_model.init_state.assert_called_once_with(
        video_path=output_path,
        async_loading_frames=False,
    )


def test_dynamic_profile_uses_all_cached_shot_descriptions():
    extractor = VideoInformationExtractor.__new__(VideoInformationExtractor)
    profiles = mock.MagicMock()
    profiles.has_profile.return_value = False
    profiles.resolve_or_create.return_value = "travel vlog"
    extractor.taxonomy_generator = SimpleNamespace(profiles=profiles)
    extractor.taxonomy_user_hints = ["host"]
    extractor.shot_boundaries = [(0, 10), (10, 20)]
    extractor._cancel = mock.MagicMock()
    extractor._describe_shot = mock.MagicMock(
        side_effect=[
            ("person . street .", "A host walks through a city."),
            ("person . food .", "The host visits a restaurant."),
        ]
    )

    cache = extractor._prepare_video_profile(None)

    assert cache == {
        0: ("person . street .", "A host walks through a city."),
        1: ("person . food .", "The host visits a restaurant."),
    }
    assert extractor.resolved_video_type == "travel vlog"
    call = profiles.resolve_or_create.call_args
    assert "Shot 0: A host walks through a city." in call.kwargs["scene_context"]
    assert "Shot 1: The host visits a restaurant." in call.kwargs["scene_context"]


def test_known_profile_still_caches_shot_descriptions():
    extractor = VideoInformationExtractor.__new__(VideoInformationExtractor)
    profiles = mock.MagicMock()
    profiles.resolve_or_create.return_value = "car commercial"
    extractor.taxonomy_generator = SimpleNamespace(profiles=profiles)
    extractor.taxonomy_user_hints = None
    extractor.shot_boundaries = [(0, 10)]
    extractor._cancel = mock.MagicMock()
    extractor._describe_shot = mock.MagicMock(
        return_value=("car . road .", "A car drives on a road.")
    )

    assert extractor._prepare_video_profile("car commercial") == {
        0: ("car . road .", "A car drives on a road.")
    }
    assert extractor.resolved_video_type == "car commercial"
    extractor._describe_shot.assert_called_once_with(0, 10)
    profiles.resolve_or_create.assert_called_once()
