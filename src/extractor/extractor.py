import cv2
import os
import numpy as np
from dino.dino_wrapper import DingoDetector
from dino.dino_prompt import DynamicPrompter
from ocr.paddle_wrapper import OCRSystem
from sam2.sam.build_sam import build_sam2_video_predictor
from utils.clip_scribe_logging import logger
from scenedetect import detect, ContentDetector
import json
from collections import defaultdict


class NumpyEncoder(json.JSONEncoder):
    """
    helper to encode information to json
    """

    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class InformationExtractor:
    """
    Process objects, text, moving history and record in json/csv
    """

    def __init__(
        self,
        sam_model,
        dingo_model,
        dingo_prompter,
        ocr_engine,
        video_path,
        logger,
        detection_interval=10,
    ):
        self.sam_model = sam_model
        self.video_path = video_path
        self.detection_interval = detection_interval

        self.ocr_engine = ocr_engine
        self.dingo_prompter = dingo_prompter
        self.dingo_model = dingo_model

        self.current_frame = 0
        self.obj_id_counter = 1  # Start IDs at 1

        # Stores { obj_id: {'box': [x1, y1, x2, y2], 'label': 'car'} }
        self.active_trackers = {}

        # Persistent map of ID -> Label
        self.id_to_label = {}

        self.metadata_log = defaultdict(dict)
        self.logger = logger

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
        self.logger.info(f"Recording video to: {output_filename}")

        self.inference_state = self.sam_model.init_state(video_path=self.video_path)
        self.total_frames = self.inference_state["num_frames"]

        self.logger.info(f"Video opened: {self.video_path}")
        self.logger.info(f"Video FPS: {fps}; Total Frames: {self.total_frames}")

    def cleanup(self):
        """Release resources"""
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()

        if hasattr(self, "video_writer") and self.video_writer is not None:
            self.video_writer.release()
            self.logger.info("Video writer released. Output saved.")

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
        if len(mask.shape) == 3:
            mask = mask[0]
        y, x = np.where(mask > 0)
        if len(x) == 0:
            return None
        return [np.min(x), np.min(y), np.max(x), np.max(y)]

    @staticmethod
    def _labels_match(label_a, label_b):
        """
        Fuzzy match for labels to handle 'car' vs 'red car'.
        Returns True if labels seem to describe the same category.
        """
        la = label_a.lower()
        lb = label_b.lower()
        return la in lb or lb in la

    def is_new_object(self, new_box, new_label):
        """
        Returns True if the box does NOT overlap significantly with
        an active tracker OF THE SAME CLASS.
        """
        for obj_id, tracker_data in self.active_trackers.items():
            active_box = tracker_data["box"]
            active_label = tracker_data["label"]

            iou = self._calculate_iou(new_box, active_box)

            self.logger.info(f"Object {obj_id} has {iou} iou")

            if iou > 0.5:
                # If labels match (e.g. 'car' overlaps 'car'), it's a duplicate -> Return False
                if self._labels_match(new_label, active_label):
                    return False
                    # If labels differ (e.g. 'man' overlaps 'car'), it's a new object -> Continue loop

        return True

    def get_next_obj_id(self):
        current_id = self.obj_id_counter
        self.obj_id_counter += 1
        return current_id

    def save_metadata(self, frame_idx, obj_ids, masks, frame_text):
        timestamp = frame_idx / self.fps
        masks_np = masks.cpu().numpy()

        frame_data = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "objects": [],
            "text_data": [],
        }

        for i, obj_id in enumerate(obj_ids):
            mask_binary = masks_np[i] > 0.0
            current_box = self._mask_to_box(mask_binary)

            if current_box:
                # Retrieve label from persistent storage
                label = self.id_to_label.get(obj_id, "unknown")

                # Update tracker with BOTH box and label
                self.active_trackers[obj_id] = {"box": current_box, "label": label}

                frame_data["objects"].append(
                    {
                        "id": obj_id,
                        "box": current_box,
                        "label": label,  # Added label to output metadata
                    }
                )
            else:
                self.active_trackers.pop(obj_id, None)

        for cur in frame_text:
            text_confidence = cur["confidence"]
            if text_confidence > 0.5:
                frame_data["text_data"].append((cur["text"], text_confidence))

        self.metadata_log[timestamp] = frame_data

    def visualize_sam_tracking(self, frame_idx, obj_ids, masks):
        if not self.video_writer.isOpened():
            self.logger.error("Error: Video Writer is NOT open.")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            self.logger.error(f"Error: Could not read frame {frame_idx}.")
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
            self.logger.info(f"Wrote frame {frame_idx} to video.")

    def _analyze_global_features(self):
        self.logger.info("--- Step 1: Analyzing Shots & Audio ---")
        self.logger.info("Detecting scenes...")

        scene_list = detect(self.video_path, ContentDetector(threshold=27.0))
        self.shot_boundaries = [
            (s[0].get_frames(), s[1].get_frames()) for s in scene_list
        ]

        if not self.shot_boundaries:
            self.shot_boundaries = [(0, self.total_frames)]

        self.logger.info(f"Found {len(self.shot_boundaries)} scenes.")
        self.logger.info(" > Calculating pacing metrics...")

        # First Shot Duration
        first_shot_frames = self.shot_boundaries[0][1] - self.shot_boundaries[0][0]
        first_shot_seconds = first_shot_frames / self.fps

        # Quick Pacing Analysis
        max_shots_in_5s = 0
        rapid_fire_intervals = []

        for i in range(len(self.shot_boundaries) - 4):
            start_time = self.shot_boundaries[i][0] / self.fps
            end_time = self.shot_boundaries[i + 4][1] / self.fps
            duration = end_time - start_time

            if duration <= 5.0:
                max_shots_in_5s = max(max_shots_in_5s, 5)
                rapid_fire_intervals.append(
                    {
                        "start_time": round(start_time, 2),
                        "end_time": round(end_time, 2),
                        "shot_count": 5,
                        "duration": round(duration, 2),
                    }
                )

        self.global_stats = {
            "total_shots": len(self.shot_boundaries),
            "avg_shot_duration": (self.total_frames / self.fps)
            / max(1, len(self.shot_boundaries)),
            "first_shot": {
                "duration_seconds": round(first_shot_seconds, 2),
                "end_frame": self.shot_boundaries[0][1],
            },
            "pacing": {
                "rapid_fire_segments": rapid_fire_intervals,
                "has_rapid_fire": len(rapid_fire_intervals) > 0,
            },
        }

        self.logger.info(
            f" > Analysis Complete. Found {len(self.shot_boundaries)} shots."
        )

    def _save_results_to_json(self, information):
        output_file = os.path.join(self.artifact_path, "extraction_summary.json")
        try:
            with open(output_file, "w") as f:
                json.dump(information, f, cls=NumpyEncoder, indent=4)
            self.logger.info(f"Extraction results successfully saved to: {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving JSON: {e}")

    def extract(self):
        self._analyze_global_features()

        for shot_idx, (start_f, end_f) in enumerate(self.shot_boundaries):
            self.active_trackers = {}
            self.current_frame = start_f

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            ret, frame = self.cap.read()

            self.logger.info(f"Processing Shot {shot_idx}: Frames {start_f} to {end_f}")

            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            scene_prompt = self.dingo_prompter.generate_prompt_from_frame(frame_rgb)
            current_shot_prompt = scene_prompt
            self.logger.info(f"Shot Prompt: {current_shot_prompt}")

            while self.current_frame < end_f:
                # --- DETECTION PHASE (Periodic) ---
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame_bgr = self.cap.read()

                if not ret:
                    break

                raw_image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                detected_objects_data = self.dingo_model.detect(
                    raw_image_rgb,
                    text_prompt=current_shot_prompt,
                    box_threshold=0.4,
                    text_threshold=0.30,
                )

                detected_text_data = self.ocr_engine.detect(raw_image_rgb)

                # Visualizations
                text_viz_path = os.path.join(
                    self.artifact_path,
                    f"ocr_result_{shot_idx}_{self.current_frame}.jpg",
                )
                object_viz_path = os.path.join(
                    self.artifact_path,
                    f"dingo_result_{shot_idx}_{self.current_frame}.png",
                )
                self.ocr_engine.save_visualization(
                    raw_image_rgb, detected_text_data, text_viz_path
                )
                self.dingo_model.map_results(
                    raw_image_rgb, detected_objects_data, object_viz_path
                )

                # Process new detections
                for box_info in detected_objects_data:
                    box = box_info["box"]
                    label = box_info["label"]

                    # --- CHANGED: Pass label to robustness check ---
                    if self.is_new_object(box, label):
                        new_id = self.get_next_obj_id()
                        self.logger.info(f"  + New object {new_id} ({label})")

                        # Register the ID -> Label mapping
                        self.id_to_label[new_id] = label

                        # Initialize tracker with box AND label
                        self.active_trackers[new_id] = {"box": box, "label": label}

                        self.sam_model.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=self.current_frame,
                            obj_id=new_id,
                            box=box,
                        )

                # --- TRACKING PHASE (Propagation) ---
                frames_left_in_shot = end_f - self.current_frame
                frames_to_track = min(self.detection_interval, frames_left_in_shot)

                if frames_to_track <= 0:
                    break

                if not self.active_trackers:
                    self.current_frame += frames_to_track
                    continue

                chunk_generator = self.sam_model.propagate_in_video(
                    self.inference_state,
                    start_frame_idx=self.current_frame,
                    max_frame_num_to_track=frames_to_track,
                )

                for frame_idx, obj_ids, video_res_masks in chunk_generator:
                    self.save_metadata(
                        frame_idx, obj_ids, video_res_masks, detected_text_data
                    )
                    self.visualize_sam_tracking(frame_idx, obj_ids, video_res_masks)

                self.current_frame += frames_to_track

        self.logger.info("Video Processing Complete.")

        information = {
            "global_analysis": self.global_stats,
            "frame_by_frame_log": self.metadata_log,
        }

        self._save_results_to_json(information)
        return information


if __name__ == "__main__":
    dingo = DingoDetector(logger)
    dingo_prompter = DynamicPrompter(logger)

    ocr = OCRSystem(logger)
    sam2 = build_sam2_video_predictor(
        "sam2_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt", "cpu"
    )

    info = InformationExtractor(
        sam2,
        dingo,
        dingo_prompter,
        ocr,
        "DODGE_qC2tIXJGDTQ - DODGE Hornet - Miroir - Tarifs - Avril.mp4",
        logger=logger,
        detection_interval=10,
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
