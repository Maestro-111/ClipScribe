from __future__ import annotations

import cv2
import gc
import os
import tempfile
import logging
import numpy as np
import numpy.typing as npt
import torch

from typing import (
    TYPE_CHECKING,
    Any,
    Sequence,
)

logger = logging.getLogger("clip_scribe")

if TYPE_CHECKING:
    from clip_scribe.sam2.sam.sam2_video_processor import SAM2VideoPredictor


class ExtractorSam2Mixin:
    device: str
    sam2_dynamic_refresh_rate: bool
    sam2_default_refresh_rate: int
    active_detection_interval: int
    cap: Any
    video_writer: Any
    id_to_label: dict[int, str]
    video_writer_dims: tuple[int, int]
    sam_model: SAM2VideoPredictor
    sam_inference_state: Any
    # Frame 0 in a per-shot SAM state corresponds to this frame in the
    # original video. All persisted frame indices remain video-global.
    sam_frame_offset: int | None

    if TYPE_CHECKING:

        def _write_video_chunk(
            self,
            video_path: str,
            start_frame: int,
            end_frame: int,
            output_path: str,
        ) -> None:
            ...

    def visualize_sam_tracking(
        self,
        frame_idx: int,
        obj_ids: Sequence[int],
        masks: torch.Tensor | npt.NDArray[Any],
    ) -> None:
        """Overlay colored SAM masks and ID labels onto the frame and write it to the output video."""
        if not self.video_writer.isOpened():
            logger.error("ClipScribe Extractor: Error: Video Writer is NOT open.")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            logger.error(
                f"ClipScribe Extractor: Error: Could not read frame {frame_idx}."
            )
            return

        vis_frame = frame.copy()

        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        for i, obj_id in enumerate(obj_ids):
            mask = masks[i]
            if len(mask.shape) == 3:
                mask = mask[0]
            mask_binary = (mask > 0.0).astype(np.uint8)

            if mask_binary.sum() > 0:
                np.random.seed(int(obj_id))
                color = np.random.randint(0, 255, (3,), dtype=int).tolist()

                # Draw Mask
                colored_mask = np.zeros_like(vis_frame)
                colored_mask[mask_binary == 1] = color
                mask_indices = mask_binary == 1
                vis_frame[mask_indices] = cv2.addWeighted(
                    vis_frame[mask_indices], 0.6, colored_mask[mask_indices], 0.4, 0
                ).reshape(-1, 3)

                # Draw ID and Label
                x, y, w, h = cv2.boundingRect(mask_binary)
                label = self.id_to_label.get(obj_id, "")

                text = f"ID {obj_id}: {label}"
                cv2.putText(
                    vis_frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        target_w, target_h = self.video_writer_dims

        if vis_frame.shape[1] != target_w or vis_frame.shape[0] != target_h:
            vis_frame = cv2.resize(vis_frame, (target_w, target_h))

        self.video_writer.write(vis_frame)

        if frame_idx % 30 == 0:
            logger.info(f"ClipScribe Extractor: Wrote frame {frame_idx} to video.")

    def _to_sam_frame(self, video_frame_idx: int) -> int:
        """Translate a video-global frame index to the current SAM shot index."""
        if self.sam_inference_state is None or self.sam_frame_offset is None:
            raise RuntimeError("SAM2 shot state is not initialized")

        sam_frame_idx = video_frame_idx - self.sam_frame_offset
        num_frames = int(self.sam_inference_state["num_frames"])
        if not 0 <= sam_frame_idx < num_frames:
            raise IndexError(
                f"Video frame {video_frame_idx} maps to SAM2 frame {sam_frame_idx}, "
                f"outside [0, {num_frames})"
            )
        return sam_frame_idx

    def _to_video_frame(self, sam_frame_idx: int) -> int:
        """Translate a current-shot SAM frame index to the original video."""
        if self.sam_frame_offset is None:
            raise RuntimeError("SAM2 shot state is not initialized")
        return self.sam_frame_offset + sam_frame_idx

    def _init_sam_inference_state_for_shot(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
    ) -> None:
        """Create a SAM2 state containing only one shot's decoded frames."""
        self._release_sam_inference_state()

        with tempfile.TemporaryDirectory(prefix="clipscribe-sam-shot-") as temp_dir:
            shot_video_path = os.path.join(temp_dir, "shot.mp4")
            self._write_video_chunk(
                video_path,
                start_frame,
                end_frame,
                shot_video_path,
            )
            sam_inference_state = self.sam_model.init_state(
                video_path=shot_video_path,
                async_loading_frames=False,
            )

        expected_frames = end_frame - start_frame
        actual_frames = int(sam_inference_state["num_frames"])
        if actual_frames != expected_frames:
            try:
                self.sam_model.reset_state(sam_inference_state)
            finally:
                del sam_inference_state
            raise RuntimeError(
                f"SAM2 initialized {actual_frames} shot frames; "
                f"expected {expected_frames}"
            )

        self.sam_inference_state = sam_inference_state
        self.sam_frame_offset = start_frame
        logger.info(
            "ClipScribe Extractor: Initialized SAM2 shot state: video frames "
            "[%s, %s) -> local [0, %s)",
            start_frame,
            end_frame,
            actual_frames,
        )

    def _resolve_sam2_refresh_rate(self, active_object_count: int) -> int:
        default_rate = self.sam2_default_refresh_rate

        if not self.sam2_dynamic_refresh_rate:
            return default_rate

        if active_object_count <= 2:
            factor = 1.0
        elif active_object_count <= 5:
            factor = 0.8
        elif active_object_count <= 10:
            factor = 0.6
        else:
            factor = 0.4

        minimum_rate = min(
            default_rate,
            max(self.active_detection_interval, 20),
        )
        return max(minimum_rate, round(default_rate * factor))

    def _release_sam_inference_state(self) -> None:
        """Release the current shot-local SAM2 state and accelerator cache."""
        sam_inference_state = self.sam_inference_state
        self.sam_inference_state = None
        self.sam_frame_offset = None

        if sam_inference_state is not None:
            try:
                self.sam_model.reset_state(sam_inference_state)
            except Exception:  # noqa: BLE001 - cleanup must not mask task failures
                logger.warning(
                    "ClipScribe Extractor: Failed to reset SAM2 inference state",
                    exc_info=True,
                )
        del sam_inference_state

        # Drop cyclic references held by model outputs before returning memory
        # to the accelerator allocator. Model weights remain resident.
        gc.collect()

        if self.device == "mps" and torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except Exception:  # noqa: BLE001 - best-effort accelerator cleanup
                logger.warning(
                    "ClipScribe Extractor: Failed to release MPS cached memory",
                    exc_info=True,
                )

        if self.device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - best-effort accelerator cleanup
                logger.warning(
                    "ClipScribe Extractor: Failed to release CUDA cached memory",
                    exc_info=True,
                )
