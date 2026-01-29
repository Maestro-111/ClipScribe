import cv2
import os

import numpy as np
import math

from dino.dino_wrapper import DinoDetector
from dino.dino_prompt import DynamicPrompter

from ocr.paddle_wrapper import OCRSystem

from sam2.sam.build_sam import build_sam2_video_predictor
from utils.clip_scribe_logging import logger

import json
import torch

from scenedetect import detect, ContentDetector
from collections import defaultdict

from .taxonomy_core import TaxonomyGenerator, TaxonomyResolver
from .taxonomy_config import ProfilesPile

from torchvision import transforms

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


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
        video_type: str,
        sam_model,
        dino_model,
        dino_prompter,
        ocr_engine,
        taxonomy_resolver,
        taxonomy_layers: dict,
        video_path,
        device: str,
        bert_threshold: float,
        logger,
        mandatory_targets: list,
        detection_interval=10,
        active_depth: int = 2,
    ):
        self.sam_model = sam_model
        self.video_path = video_path

        self.detection_interval = detection_interval
        self.active_depth = active_depth

        self.ocr_engine = ocr_engine
        self.dingo_prompter = dino_prompter
        self.dingo_model = dino_model

        self.device = device
        self.logger = logger
        self.logger.info(
            f"Loading DINOv2 (ViT-S/14) for Object Re-Identification on {self.device}..."
        )

        self.reid_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(
            self.device
        )
        self.reid_model.eval()

        self.current_frame = 0
        self.obj_id_counter = 1

        self.active_trackers: dict[int, dict] = {}
        self.id_to_label: dict[int, str] = {}

        self.text_registry: dict[int, set] = defaultdict(set)
        self.object_registry: dict[int, dict] = {}

        self.taxonomy_resolver = taxonomy_resolver
        self.video_type = video_type

        self.bert_threshold = bert_threshold

        # --- STORE THE MANDATORY TARGETS ---
        self.mandatory_targets = mandatory_targets if mandatory_targets else []
        # -----------------------------------

        self.artifact_path = f"extractor_artifacts/{video_path}/"

        self.embedding_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.taxonomy_search_set_up(taxonomy_layers)
        self.state_init()

    def taxonomy_search_set_up(self, taxonomy_layers):
        self.search_terms = []

        for level in range(self.active_depth):
            if level in taxonomy_layers:
                layer_items = taxonomy_layers[level]
                self.logger.info(
                    f"Loading Level {level} ({len(layer_items)} items) into search pool."
                )
                self.search_terms.extend(layer_items)
            else:
                self.logger.warning(
                    f"Requested level {level} not found in taxonomy layers."
                )

        # --- ADD THIS BLOCK ---
        # Add mandatory targets to search terms so they are valid candidates
        if self.mandatory_targets:
            self.logger.info(
                f"Adding mandatory targets to search pool: {self.mandatory_targets}"
            )
            self.search_terms.extend(self.mandatory_targets)
        # ----------------------

        self.search_terms = list(set([t.lower().strip() for t in self.search_terms]))

        self.logger.info(f"Final Semantic Search Pool Size: {len(self.search_terms)}")
        self.logger.info(f"Pool Sample: {self.search_terms[:100]}")

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

            self.logger.info(
                f"Object {obj_id} label {active_label} has {iou} iou with new box label {new_label}"
            )

            if iou > 0.5:
                if self._labels_match(new_label, active_label):
                    return False

        return True

    def get_next_obj_id(self):
        current_id = self.obj_id_counter
        self.obj_id_counter += 1
        return current_id

    def _extract_embedding(self, frame_bgr, box):
        """
        Crops the object and returns a DINOv2 embedding vector.
        Uses self.reid_model (DINOv2) instead of self.dingo_model (GroundingDINO).
        """
        x1, y1, x2, y2 = map(int, box)
        h, w, _ = frame_bgr.shape

        # Safe crop
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        img_tensor = self.embedding_transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.reid_model.forward_features(img_tensor)
            embedding = features["x_norm_clstoken"]

        return embedding.cpu().numpy().flatten()

    def save_metadata(
        self, frame_idx, obj_ids, masks, frame_text, shot_idx, current_frame_img
    ):
        timestamp = frame_idx / self.fps
        second_key = int(timestamp)
        masks_np = masks.cpu().numpy()

        h, w, _ = current_frame_img.shape
        bottom_threshold = h * 0.85

        # 1. Text (Grouped by Second)
        for cur in frame_text:
            box_y_min = cur["box"][1]
            if cur["confidence"] > 0.6 and box_y_min < bottom_threshold:
                self.text_registry[second_key].add(cur["text"])

        # 2. Objects
        for i, obj_id in enumerate(obj_ids):
            mask_binary = masks_np[i] > 0.0
            current_box = self._mask_to_box(mask_binary)

            if current_box:
                label = self.id_to_label.get(obj_id, "unknown")

                # If this is the VERY FIRST time we see this specific ID, capture embedding
                if obj_id not in self.object_registry:
                    self.object_registry[obj_id] = {
                        "label": label,
                        "shot_id": shot_idx,
                        "embedding_sum": np.zeros(384),  # Can be None if crop failed
                        "embedding_count": 0,
                        "boxes": [],
                        "timestamps": [],
                    }

                    new_emb = self._extract_embedding(current_frame_img, current_box)
                    if new_emb is not None:
                        # Accumulate raw vector
                        current_sum = self.object_registry[obj_id]["embedding_sum"]

                        if new_emb.shape == current_sum.shape:
                            self.object_registry[obj_id]["embedding_sum"] += new_emb
                            self.object_registry[obj_id]["embedding_count"] += 1
                    # ---------------------------------------------

                self.object_registry[obj_id]["boxes"].append(current_box)
                self.object_registry[obj_id]["timestamps"].append(timestamp)

                self.active_trackers[obj_id] = {"box": current_box, "label": label}
            else:
                self.active_trackers.pop(obj_id, None)

    def _calculate_metrics(self, boxes, timestamps):
        if len(boxes) < 2:
            # Return defaults for all 4 values if not enough data
            return 0.0, 0.0, 0.0, "unknown"

        # --- 1. Existing Velocity/Growth Logic ---
        centroids = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        dist = 0
        for k in range(1, len(centroids)):
            d = math.sqrt(
                (centroids[k][0] - centroids[k - 1][0]) ** 2
                + (centroids[k][1] - centroids[k - 1][1]) ** 2
            )
            dist += d
        duration = timestamps[-1] - timestamps[0]
        velocity = dist / duration if duration > 0 else 0

        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        growth = areas[-1] / (areas[0] + 1e-6)

        # --- 2. NEW: Screen Coverage & Shot Scale ---
        # Get frame dimensions from class state
        width, height = self.video_writer_dims
        frame_area = width * height

        # Find the maximum size this object reached in the shot
        max_box_area = max(areas)

        # Calculate percentage of screen occupied (0.0 to 1.0)
        screen_coverage = max_box_area / (frame_area + 1e-6)

        # Determine Shot Scale based on coverage
        shot_scale = "Long Shot"
        if screen_coverage > 0.15:
            shot_scale = "Medium Shot"
        if screen_coverage > 0.40:
            shot_scale = "Close Up"

        return (
            round(velocity, 2),
            round(growth, 2),
            round(screen_coverage, 3),
            shot_scale,
        )

    def _resolve_identities(self):
        self.logger.info("Resolving identities across shots...")
        id_map = {}
        next_global_id = 0

        object_ids = sorted(
            self.object_registry.keys(),
            key=lambda k: self.object_registry[k]["timestamps"][0],
        )

        for i, id_a in enumerate(object_ids):
            if id_a in id_map:
                continue

            current_global_id = next_global_id
            next_global_id += 1
            id_map[id_a] = current_global_id

            obj_a = self.object_registry[id_a]

            if obj_a["embedding_count"] == 0:
                continue
            emb_a = obj_a["embedding_sum"] / obj_a["embedding_count"]
            norm_a = np.linalg.norm(emb_a)

            start_a = obj_a["timestamps"][0]
            end_a = obj_a["timestamps"][-1]

            for j in range(i + 1, len(object_ids)):
                id_b = object_ids[j]
                if id_b in id_map:
                    continue

                obj_b = self.object_registry[id_b]

                if not self._labels_match(obj_a["label"], obj_b["label"]):
                    continue

                # 2. TEMPORAL OVERLAP CHECK
                start_b = obj_b["timestamps"][0]
                end_b = obj_b["timestamps"][-1]

                is_overlapping = max(start_a, start_b) < min(end_a, end_b)

                if is_overlapping:
                    continue

                # 3. Visual Check (Sequential)
                if obj_b["embedding_count"] == 0:
                    continue
                emb_b = obj_b["embedding_sum"] / obj_b["embedding_count"]
                norm_b = np.linalg.norm(emb_b)

                if norm_a == 0 or norm_b == 0:
                    continue
                similarity = np.dot(emb_a, emb_b) / (norm_a * norm_b)

                threshold = 0.65

                if similarity > threshold:
                    id_map[id_b] = current_global_id
                    self.logger.info(
                        f"Merged {id_a} (Shot {obj_a['shot_id']}) & {id_b} (Shot {obj_b['shot_id']}) - Sim: {similarity:.2f}"
                    )

                    end_a = max(end_a, end_b)

        return id_map

    def _finalize_data(self):
        global_id_map = self._resolve_identities()
        final_objects = {}

        for local_id, data in self.object_registry.items():
            g_id = global_id_map.get(local_id, -1)

            velocity, growth, coverage, scale = self._calculate_metrics(
                data["boxes"], data["timestamps"]
            )

            # Determine mechanics
            move_type = "static"
            if velocity > 50:
                move_type = "slow"
            if velocity > 200:
                move_type = "fast"

            cam_type = "stable"
            if growth > 1.2:
                cam_type = "approaching"
            elif growth < 0.8:
                cam_type = "retreating"

            occurence_data = {
                "shot_index": data["shot_id"],
                "lifespan": [
                    round(data["timestamps"][0], 2),
                    round(data["timestamps"][-1], 2),
                ],
                "mechanics": f"{move_type}, {cam_type}",
                # --- NEW FIELDS ---
                "scale": scale,  # e.g. "Close Up"
                "screen_coverage": coverage,  # e.g. 0.45
                "velocity_px_sec": velocity,  # Added
                "growth_factor": growth,  # Added
            }

            if g_id not in final_objects:
                final_objects[g_id] = {
                    "global_id": g_id,
                    "label": data["label"],
                    "occurrences": [occurence_data],
                }
            else:
                final_objects[g_id]["occurrences"].append(occurence_data)

        # 3. Format Text
        final_text = [
            {"second": sec, "text": list(txt_set)}
            for sec, txt_set in sorted(self.text_registry.items())
        ]

        return {
            "global_stats": self.global_stats if hasattr(self, "global_stats") else {},
            "visual_objects": list(final_objects.values()),
            "text_events": final_text,
        }

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

        # 1. Run Scene Detection
        # threshold=27.0 compares adjacent frames' content (HSV).
        # >27 diff = Cut.
        scene_list = detect(self.video_path, ContentDetector(threshold=27.0))

        # Format: [(start_frame, end_frame), ...]
        self.shot_boundaries = [
            (s[0].get_frames(), s[1].get_frames()) for s in scene_list
        ]

        # Handle case with 0 detections (entire video is 1 shot)
        if not self.shot_boundaries:
            self.shot_boundaries = [(0, self.total_frames)]

        self.logger.info(f"Found {len(self.shot_boundaries)} scenes.")

        # 2. Convert to Seconds for precise calculation
        # Data format: [{"index": i, "start": 0.0, "end": 2.4, "duration": 2.4}, ...]
        shot_data = []
        for i, (start_f, end_f) in enumerate(self.shot_boundaries):
            shot_data.append(
                {
                    "index": i,
                    "start": round(start_f / self.fps, 3),
                    "end": round(end_f / self.fps, 3),
                    "duration": round((end_f - start_f) / self.fps, 3),
                }
            )

        # --- METRIC 1: Dynamic Start ---
        # "The first shot in the video changes in less than 3 seconds."
        first_shot = shot_data[0]
        has_dynamic_start = first_shot["duration"] < 3.0

        # --- METRIC 2: Quick Pacing (First 5 Seconds) ---
        # "There are at least 5 shot changes... in the first 5 seconds"
        # We count shots that START before the 5.0s mark.
        shots_in_first_5s = [s for s in shot_data if s["start"] < 5.0]
        has_quick_pacing_start = len(shots_in_first_5s) >= 5

        # --- METRIC 3: Quick Pacing (Any 5 Consecutive Seconds) ---
        # "Within ANY 5 consecutive seconds there are 5 or more shots"
        # Logic: Look at shot i and shot i+4. If (end of i+4) - (start of i) <= 5.0s,
        # then we have 5 shots happening within a 5s window.
        has_quick_pacing_any = False
        rapid_fire_intervals = []

        if len(shot_data) >= 5:
            for i in range(len(shot_data) - 4):
                current_shot = shot_data[i]
                fifth_shot = shot_data[i + 4]
                cluster_duration = fifth_shot["end"] - current_shot["start"]

                if cluster_duration <= 5.0:
                    has_quick_pacing_any = True
                    rapid_fire_intervals.append(
                        {
                            "start_time": current_shot["start"],
                            "end_time": fifth_shot["end"],
                            "shot_count": 5,  # Minimum 5, could be tighter
                            "duration": round(cluster_duration, 2),
                            "shot_indices": [s["index"] for s in shot_data[i : i + 5]],
                        }
                    )

        # 3. Compile Final Stats
        self.global_stats = {
            "total_shots": len(self.shot_boundaries),
            "video_duration": round(self.total_frames / self.fps, 2),
            "avg_shot_duration": round(
                sum(s["duration"] for s in shot_data) / len(shot_data), 2
            ),
            # Feature: Dynamic Start
            "dynamic_start": {
                "detected": has_dynamic_start,
                "first_shot_duration": first_shot["duration"],
                "criteria": "First shot < 3.0s",
            },
            # Feature: Quick Pacing (First 5s)
            "quick_pacing_intro": {
                "detected": has_quick_pacing_start,
                "shot_count": len(shots_in_first_5s),
                "shots": [s["index"] for s in shots_in_first_5s],
                "criteria": ">= 5 shots starting within t=0s to t=5s",
            },
            # Feature: Quick Pacing (Anywhere)
            "quick_pacing_general": {
                "detected": has_quick_pacing_any,
                "rapid_fire_segments": rapid_fire_intervals,
                "criteria": ">= 5 shots within any 5s window",
            },
        }

        self.logger.info(f" > Analysis Complete. Dynamic Start: {has_dynamic_start}")

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
            self.sam_model.reset_state(self.inference_state)

            self.active_trackers = {}
            self.current_frame = start_f

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            ret, frame = self.cap.read()

            self.logger.info(f"Processing Shot {shot_idx}: Frames {start_f} to {end_f}")

            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            scene_prompt = self.dingo_prompter.generate_prompt_from_frame(
                frame_rgb, self.search_terms
            )

            if self.mandatory_targets:
                scene_prompt += " . ".join(self.mandatory_targets)

            current_shot_prompt = scene_prompt

            self.logger.info(f"Final Dino Shot Prompt: {current_shot_prompt}")

            while self.current_frame < end_f:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame_bgr = self.cap.read()

                if not ret:
                    break

                raw_image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                detected_objects_data = self.dingo_model.detect(
                    raw_image_rgb,
                    text_prompt=current_shot_prompt,
                    box_threshold=0.3,
                    text_threshold=0.30,
                )

                self.logger.info(
                    f"Detected objects #{len(detected_objects_data)} in the current frame"
                )

                detected_text_data = self.ocr_engine.detect(raw_image_rgb)

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

                for box_info in detected_objects_data:
                    box = box_info["box"]
                    label = box_info["label"]

                    best_semantic = self.taxonomy_resolver.resolve(
                        self.search_terms, label, threshold=self.bert_threshold
                    )

                    if best_semantic is None:
                        self.logger.warning(
                            f"No proper semantic found for label: {label}. skipping..."
                        )
                        continue

                    if self.is_new_object(box, best_semantic):
                        new_id = self.get_next_obj_id()
                        self.logger.info(f"  + New object {new_id} ({best_semantic})")

                        self.id_to_label[new_id] = best_semantic
                        self.active_trackers[new_id] = {
                            "box": box,
                            "label": best_semantic,
                        }

                        self.sam_model.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=self.current_frame,
                            obj_id=new_id,
                            box=box,
                        )

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
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    _, curr_frame = self.cap.read()

                    self.save_metadata(
                        frame_idx,
                        obj_ids,
                        video_res_masks,
                        detected_text_data,
                        shot_idx,
                        curr_frame,
                    )
                    self.visualize_sam_tracking(frame_idx, obj_ids, video_res_masks)

                self.current_frame += frames_to_track

        self.logger.info("Video Processing Complete. Finalizing data...")

        information = self._finalize_data()
        self._save_results_to_json(information)
        return information


if __name__ == "__main__":
    video_type = "car ad"
    bert_threshold = 0.8

    logger.info(f"bert_threshold: {bert_threshold}")

    profiles = ProfilesPile()
    taxonomy_resolver = TaxonomyResolver(logger)

    gen = TaxonomyGenerator(75, profiles, logger)
    taxonomy_layers = gen.build_taxonomy(video_type, levels=3)

    # --- FIX 1: Extract Brand Keywords ---
    car_profile = profiles.get_video_profile(video_type)
    brand_terms = car_profile.brand_keywords if car_profile else []
    logger.info(f"Mandatory Brand Terms: {brand_terms}")
    # -------------------------------------

    dino = DinoDetector(logger)
    dino_prompter = DynamicPrompter(logger, taxonomy_resolver, bert_threshold)

    sam2_device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    dino_reid_device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    logger.info(f"sam2 Using device: {sam2_device}")

    ocr = OCRSystem(logger)
    sam2 = build_sam2_video_predictor(
        "sam2_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt", sam2_device.type
    )

    info = InformationExtractor(
        video_type,
        sam2,
        dino,
        dino_prompter,
        ocr,
        taxonomy_resolver,
        taxonomy_layers,
        "CHRYSLER_jtuJbB1QXd8 - Nov 2024 Pacifica.mp4",
        dino_reid_device.type,
        bert_threshold,
        logger=logger,
        mandatory_targets=brand_terms,
        detection_interval=5,
        active_depth=2,
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
