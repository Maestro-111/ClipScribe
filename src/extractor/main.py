import cv2
import os
import numpy as np
from dingo.dingo_wrapper import DingoDetector
from ocr.paddle_wrapper import OCRSystem
from sam2.sam.build_sam import build_sam2_video_predictor


class InformationExtractor:
    """
    Process objects, text, moving history and record in json/csv
    """

    def __init__(
        self,
        sam_model,
        dingo_model,
        dingo_prompt,
        ocr_engine,
        video_path,
        detection_interval=10,
    ):
        self.sam_model = sam_model
        self.video_path = video_path
        self.detection_interval = detection_interval

        self.ocr_engine = ocr_engine

        self.dingo_prompt = dingo_prompt
        self.dingo_model = dingo_model

        self.current_frame = 0
        self.obj_id_counter = 1  # Start IDs at 1

        # Stores { obj_id: [x1, y1, x2, y2] } of the most recent mask
        self.active_trackers = {}
        self.metadata_log = []

        self.artifact_path = f"extractor_artifacts/{video_path}/"

        self.state_init()

    def state_init(self):
        if not os.path.exists(self.artifact_path):
            os.makedirs(self.artifact_path)

        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.video_writer_dims = (width, height)

        if fps <= 0:
            fps = 30.0

        self.fps = fps

        output_filename = os.path.join(self.artifact_path, "tracked_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")

        self.video_writer = cv2.VideoWriter(
            output_filename, fourcc, fps, (width, height)
        )
        print(f"Recording video to: {output_filename}")

        self.inference_state = self.sam_model.init_state(video_path=self.video_path)
        self.total_frames = self.inference_state["num_frames"]

        print(f"Video opened: {self.video_path}")
        print(f"Video FPS: {fps}; Total Frames: {self.total_frames}")

    def cleanup(self):
        """Release resources"""
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()

        # --- NEW: Release Writer ---
        if hasattr(self, "video_writer") and self.video_writer is not None:
            self.video_writer.release()
            print("Video writer released. Output saved.")

    @staticmethod
    def _calculate_iou(boxA, boxB):
        # Determine intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute area of intersection
        interWidth = max(0, xB - xA)
        interHeight = max(0, yB - yA)
        interArea = interWidth * interHeight

        # Compute area of both rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Compute IoU
        epsilon = 1e-5
        iou = interArea / float(boxAArea + boxBArea - interArea + epsilon)
        return iou

    @staticmethod
    def _mask_to_box(mask):
        """Converts SAM mask to bounding box [x1, y1, x2, y2]"""
        # Mask shape is [1, H, W] or [H, W]
        if len(mask.shape) == 3:
            mask = mask[0]
        y, x = np.where(mask > 0)
        if len(x) == 0:
            return None
        return [np.min(x), np.min(y), np.max(x), np.max(y)]

    # 2. Implement is_new_object logic
    def is_new_object(self, new_box):
        """Returns True if the box does NOT overlap with active trackers."""
        for obj_id, active_box in self.active_trackers.items():
            if self._calculate_iou(new_box, active_box) > 0.5:  # param?
                return False
        return True

    def get_next_obj_id(self):
        current_id = self.obj_id_counter
        self.obj_id_counter += 1
        return current_id

    def save_metadata(self, frame_idx, obj_ids, masks, frame_text):
        timestamp = frame_idx / self.fps

        masks_np = masks.cpu().numpy()

        frame_data = {
            "frame": frame_idx,
            "timestamp": timestamp,
            "objects": [],
            "text_data": [],
            "text_coords": [],
        }

        for i, obj_id in enumerate(obj_ids):
            mask_binary = masks_np[i] > 0.0
            current_box = self._mask_to_box(mask_binary)

            if current_box:
                self.active_trackers[obj_id] = current_box

                frame_data["objects"].append(
                    {
                        "id": obj_id,
                        "box": current_box,
                    }
                )
            else:
                self.active_trackers.pop(obj_id, None)

        for cur in frame_text:
            text_confidence = cur["confidence"]

            if text_confidence > 0.5:
                frame_data["text_data"].append((cur["text"], text_confidence))
                frame_data["text_coords"].append(cur["box"])

        self.metadata_log.append(frame_data)

    def visualize_sam_tracking(self, frame_idx, obj_ids, masks):
        if not self.video_writer.isOpened():
            print("Error: Video Writer is NOT open. File path issue or Codec missing?")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            print(f"Error: Could not read frame {frame_idx} from source video.")
            return

        vis_frame = frame.copy()

        if hasattr(masks, "cpu"):
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

                # Draw ID
                x, y, w, h = cv2.boundingRect(mask_binary)
                cv2.putText(
                    vis_frame,
                    f"ID {obj_id}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        target_w, target_h = self.video_writer_dims

        if vis_frame.shape[1] != target_w or vis_frame.shape[0] != target_h:
            print(
                f"Resizing frame {frame_idx}: {vis_frame.shape} -> {self.video_writer_dims}"
            )
            vis_frame = cv2.resize(vis_frame, (target_w, target_h))

        self.video_writer.write(vis_frame)

        if frame_idx % 30 == 0:
            print(f"Wrote frame {frame_idx} to video.")

    def extract(self):
        while self.current_frame < self.total_frames:
            print(f"--- Processing Chunk starting at Frame {self.current_frame} ---")

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

            ret, frame_bgr = self.cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            raw_image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            detected_objects_data = self.dingo_model.detect(
                raw_image_rgb,
                text_prompt=self.dingo_prompt,
                box_threshold=0.35,
                text_threshold=0.30,
            )
            detected_text_data = self.ocr_engine.detect(raw_image_rgb)

            text_viz_path = os.path.join(
                self.artifact_path, f"ocr_result_{self.current_frame}.jpg"
            )
            object_viz_path = os.path.join(
                self.artifact_path, f"dingo_result_{self.current_frame}.png"
            )

            self.ocr_engine.save_visualization(
                raw_image_rgb, detected_text_data, text_viz_path
            )
            self.dingo_model.map_results(
                raw_image_rgb, detected_objects_data, object_viz_path
            )

            for box_info in detected_objects_data:
                box = box_info["box"]
                label = box_info["label"]

                if label in ["text", "dashboard"]:  # make as param
                    print(f"ignoring box label {label}")
                    continue

                # Deduplication Check
                if self.is_new_object(box):
                    new_id = self.get_next_obj_id()
                    print(
                        f"Injecting new object {new_id} ({label}) at frame {self.current_frame}"
                    )

                    self.sam_model.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=self.current_frame,
                        obj_id=new_id,
                        box=box,
                    )
                    # Add to trackers immediately to prevent double-adding in same frame
                    self.active_trackers[new_id] = box

            remaining_frames = self.total_frames - self.current_frame
            frames_to_track = min(self.detection_interval, remaining_frames)

            if frames_to_track <= 0:
                break

            if not self.active_trackers:
                print(
                    f"No active objects. Skipping tracking for {frames_to_track} frames."
                )
                self.current_frame += frames_to_track
                continue

            chunk_generator = self.sam_model.propagate_in_video(
                self.inference_state,
                start_frame_idx=self.current_frame,
                max_frame_num_to_track=frames_to_track,
            )

            for frame_idx, obj_ids, video_res_masks in chunk_generator:
                print(f"Frame {frame_idx} processed")
                self.save_metadata(
                    frame_idx, obj_ids, video_res_masks, detected_text_data
                )
                self.visualize_sam_tracking(frame_idx, obj_ids, video_res_masks)

            self.current_frame += frames_to_track

        print("Video Processing Complete.")
        return self.metadata_log


dingo = DingoDetector()
dingo_prompt = "car . person . text . human face . dashboard . "
ocr = OCRSystem()
sam2 = build_sam2_video_predictor(
    "sam2_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt", "cpu"
)


info = InformationExtractor(
    sam2,
    dingo,
    dingo_prompt,
    ocr,
    "DODGE_lA2DSd8Ik3Y - Sept 2023 Dodge Hornet.mp4",
    detection_interval=5,
)

try:
    metadata = info.extract()
    print("Extraction finished successfully.")
except KeyboardInterrupt:
    print("\n!!! Interrupted by User. Saving video... !!!")
except Exception as e:
    print(f"\n!!! Error: {e} !!!")
finally:
    info.cleanup()
    print("Done.")
