"""Focused tests for SAM2 checkpoint identity reconciliation."""

from unittest import mock

import torch

from src.extractor.extractor_core import VideoInformationExtractor


def _extractor_with_sam() -> tuple[VideoInformationExtractor, mock.MagicMock]:
    extractor = VideoInformationExtractor.__new__(VideoInformationExtractor)
    sam_model = mock.MagicMock()
    extractor.sam_model = sam_model
    extractor.inference_state = object()
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
    sam_model.propagate_in_video.return_value = iter([(12, [7, 8], masks)])

    visible = extractor._predict_registered_objects_at(12)

    assert visible == {7: {"box": [2, 1, 5, 4], "label": "car"}}
    sam_model.propagate_in_video.assert_called_once_with(
        extractor.inference_state,
        start_frame_idx=12,
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
        inference_state=extractor.inference_state,
        frame_idx=12,
        obj_id=1,
        box=[2, 1, 5, 4],
    )
