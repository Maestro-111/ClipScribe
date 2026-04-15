import cv2
import os

import re
import logging
import numpy as np
import math

import json
import torch

from typing import TYPE_CHECKING

from scenedetect import detect, ContentDetector
from collections import defaultdict

from dotenv import load_dotenv, find_dotenv

from nltk.corpus import wordnet as wn

if TYPE_CHECKING:
    import whisper
    from torchvision import transforms
    from facenet_pytorch import MTCNN
    from src.dino.dino_wrapper import DinoDetector
    from src.dino.dino_prompt import DynamicPrompter
    from src.ocr.paddle_wrapper import OCRSystem
    from src.extractor.taxonomy_core import TaxonomyGenerator, TaxonomyResolver

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

    @staticmethod
    def merge_dino_prompts(prompts):
        """
        Deduplicates noun phrases across multiple DINO-formatted prompts.
        Input:  ["car . road . tree .", "car . building ."]
        Output: "car . road . tree . building ."
        """
        seen = set()
        unique = []
        for prompt in prompts:
            parts = [p.strip() for p in prompt.split(".") if p.strip()]
            for part in parts:
                if part.lower() not in seen:
                    seen.add(part.lower())
                    unique.append(part)
        return " . ".join(unique) + " ." if unique else "object ."

    @staticmethod
    def merge_raw_prompts(prompts):
        """
        Joins raw BLIP captions into a combined context for the LLM.
        Input:  ["a car on a road with trees", "a car near a building"]
        Output: "a car on a road with trees, a car near a building"
        """
        seen = set()
        unique = []
        for p in prompts:
            normalized = p.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique.append(p.strip())
        return ", ".join(unique) if unique else ""

    @staticmethod
    def _calculate_iou(boxA, boxB):
        """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
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
        Fuzzy match for labels using string inclusion and WordNet semantics.
        Checks for synonyms and direct hypernym/hyponym relationships.
        """
        la = label_a.lower().strip()
        lb = label_b.lower().strip()

        # Fallback: Original string inclusion (handles "car" vs "red car")
        if la in lb or lb in la:
            return True

        # Format for WordNet (multi-word labels use underscores, e.g., "human_face")
        wn_la = la.replace(" ", "_")
        wn_lb = lb.replace(" ", "_")

        # Fetch synsets (meaning groupings) for both words
        synsets_a = wn.synsets(wn_la)
        synsets_b = wn.synsets(wn_lb)

        # If either word isn't in WordNet's dictionary, we can't compare semantically
        if not synsets_a or not synsets_b:
            return False

        # Cross-reference all meanings of Label A against all meanings of Label B
        for syn_a in synsets_a:
            for syn_b in synsets_b:
                # Check A: Exact Synonym (Same Synset)
                # Example: "automobile" and "car" share a synset.
                if syn_a == syn_b:
                    return True

                # Check B: Direct Hypernym (Parent Category) or Hyponym (Child Category)
                # Example: "vehicle" is a direct hypernym of "car".

                # Is B a direct parent of A?
                if syn_b in syn_a.hypernyms():
                    return True

                # Is A a direct parent of B?
                if syn_a in syn_b.hypernyms():
                    return True

                # Optional Check C: Wu-Palmer Semantic Similarity
                # If they aren't DIRECT parents/children, but are very closely related
                # up the tree (e.g., "truck" and "car" sharing the parent "motor_vehicle")
                similarity = syn_a.wup_similarity(syn_b)
                if similarity is not None and similarity > 0.85:
                    return True

        return False

    @staticmethod
    def _is_valid_text(text_data, frame_height, max_text_height):
        """
        Robust filter for OCR noise and irrelevant fine print.
        Returns True only if the text is significant.
        """

        text = text_data["text"]
        confidence = text_data["confidence"]

        x1, y1, x2, y2 = text_data["box"]

        box_height = y2 - y1

        relative_height = box_height / frame_height

        if relative_height < 0.02:
            if confidence < 0.90:
                return False

        elif relative_height < 0.04:
            if confidence < 0.80:
                return False

        else:
            if confidence < 0.60:
                return False

        if max_text_height > 0:
            if box_height < (max_text_height * 0.20):
                return False

        clean_chars = re.sub(r"[^a-zA-Z0-9]", "", text)

        if len(clean_chars) < 2:
            return False

        alpha_ratio = len(clean_chars) / len(text)

        if alpha_ratio < 0.5:
            return False

        ignore_terms = [  # param?
            "msrp",
            "copyright",
            "rights reserved",
            "fca us llc",
            "visit",
            "www.",
            ".com",
            "license",
            "simulation",
        ]

        text_lower = text.lower()
        if any(term in text_lower for term in ignore_terms):
            return False

        return True

    def __init__(
        self,
        video_type: str,
        video_path: str,
        video_name: str,
        sam_model: torch.nn.Module,
        dino_model: "DinoDetector",
        dino_prompter: "DynamicPrompter",
        ocr_engine: "OCRSystem",
        taxonomy_resolver: "TaxonomyResolver",
        taxonomy_generator: "TaxonomyGenerator",
        reid_model: torch.nn.Module,
        audio_model: "whisper.Whisper",
        embedding_transform: "transforms.Compose",
        face_detection: "MTCNN",
        device: str,
        dino_reid_device: str,
        word_similarity_threshold: float,
        dino_text_conf: float,
        dino_box_conf: float,
        torch_face_cong: float,
        audio_confidence: float,
        label_match_merge_threshold: float,
        label_no_match_merge_threshold: float,
        logger: logging.Logger,
        detection_interval: int = 10,
        reid_model_frame_check_freq: int = 20,
    ):
        # helper models

        self.sam_model = sam_model
        self.ocr_engine = ocr_engine
        self.dingo_prompter = dino_prompter

        self.dingo_model = dino_model
        self.audio_model = audio_model

        self.embedding_transform = embedding_transform
        self.reid_model = reid_model

        self.face_detection = face_detection

        # video params

        self.video_path = video_path
        self.video_name = video_name
        self.video_type = video_type

        self.detection_interval = detection_interval

        self.current_frame = 0
        self.obj_id_counter = 1

        self.active_trackers: dict[int, dict] = {}
        self.id_to_label: dict[int, str] = {}

        self.text_registry: dict[int, set] = defaultdict(set)
        self.object_registry: dict[int, dict] = {}

        self.taxonomy_resolver = taxonomy_resolver
        self.taxonomy_generator = taxonomy_generator

        self.word_similarity_threshold = word_similarity_threshold

        self.dino_text_conf = dino_text_conf
        self.dino_box_conf = dino_box_conf

        self.torch_face_cong = torch_face_cong

        self.artifact_path = f"extractor_artifacts/{self.video_name}/"

        self.label_match_merge_threshold = label_match_merge_threshold
        self.label_no_match_merge_threshold = label_no_match_merge_threshold

        self.device = device
        self.dino_reid_device = dino_reid_device
        self.logger = logger

        self.audio_registry: list[dict] = []
        self.audio_confidence = audio_confidence

        self.reid_model_frame_check_freq = reid_model_frame_check_freq

        self.state_init()

    def __repr__(self) -> str:
        return f"InformationExtractor: device: {self.device}"

    def state_init(self):
        """Open the video capture, initialize the video writer, and set up SAM inference state."""
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
        """Return the next available object ID and increment the counter."""
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

        img_tensor = (
            self.embedding_transform(crop_rgb).unsqueeze(0).to(self.dino_reid_device)
        )

        with torch.no_grad():
            features = self.reid_model.forward_features(img_tensor)
            embedding = features["x_norm_clstoken"]

        return embedding.cpu().numpy().flatten()

    def save_metadata(
        self, frame_idx, obj_ids, masks, frame_text, shot_idx, current_frame_img
    ):
        """
        Record per-frame tracking data: filter and store OCR text, convert SAM masks
        to bounding boxes, and periodically accumulate DINOv2 embeddings for re-ID.
        """
        timestamp = frame_idx / self.fps
        second_key = int(timestamp)
        masks_np = masks.cpu().numpy()

        h, w, _ = current_frame_img.shape

        max_text_height = 0
        if frame_text:
            max_text_height = max((t["box"][3] - t["box"][1]) for t in frame_text)

        for cur in frame_text:
            if self._is_valid_text(cur, h, max_text_height):
                self.text_registry[second_key].add(cur["text"])

        # 2. Objects
        for i, obj_id in enumerate(obj_ids):
            mask_binary = masks_np[i] > 0.0
            current_box = self._mask_to_box(mask_binary)

            if current_box:
                label = self.id_to_label.get(obj_id, "unknown")

                if obj_id not in self.object_registry:
                    self.object_registry[obj_id] = {
                        "label": label,
                        "shot_id": shot_idx,
                        "embedding_sum": np.zeros(384),
                        "embedding_count": 0,
                        "boxes": [],
                        "timestamps": [],
                        "last_embedding_frame": -float("inf"),
                    }

                # Accumulate embeddings periodically for robust cross-shot re-identification

                frames_since_last = (
                    frame_idx - self.object_registry[obj_id]["last_embedding_frame"]
                )

                if frames_since_last >= self.reid_model_frame_check_freq:
                    new_emb = self._extract_embedding(current_frame_img, current_box)

                    if (
                        new_emb is not None
                        and new_emb.shape
                        == self.object_registry[obj_id]["embedding_sum"].shape
                    ):
                        if self.object_registry[obj_id]["embedding_count"] == 0:
                            self.object_registry[obj_id]["embedding_sum"] += new_emb
                            self.object_registry[obj_id]["embedding_count"] += 1
                            self.object_registry[obj_id][
                                "last_embedding_frame"
                            ] = frame_idx

                        else:
                            current_mean = (
                                self.object_registry[obj_id]["embedding_sum"]
                                / self.object_registry[obj_id]["embedding_count"]
                            )

                            norm_new = np.linalg.norm(new_emb)
                            norm_mean = np.linalg.norm(current_mean)

                            if norm_new > 0 and norm_mean > 0:
                                cos_sim = np.dot(new_emb, current_mean) / (
                                    norm_new * norm_mean
                                )

                                if cos_sim < 0.85:
                                    self.object_registry[obj_id][
                                        "embedding_sum"
                                    ] += new_emb
                                    self.object_registry[obj_id]["embedding_count"] += 1

                                    self.logger.info(
                                        f"New viewpoint for ID {obj_id} captured. Sim: {cos_sim:.2f}"
                                    )
                        self.object_registry[obj_id]["last_embedding_frame"] = frame_idx

                self.object_registry[obj_id]["boxes"].append(current_box)
                self.object_registry[obj_id]["timestamps"].append(timestamp)

                self.active_trackers[obj_id] = {"box": current_box, "label": label}
            else:
                self.active_trackers.pop(obj_id, None)

    def _calculate_metrics(self, boxes, timestamps):
        """
        Derive motion and spatial metrics from a sequence of bounding boxes:
        velocity, growth factor, screen coverage, direction, centrality, screen time, and quadrant.
        """
        if len(boxes) < 2:
            return 0.0, 0.0, 0.0, "unknown", 0.0, 0.0, "center"

        centroids = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        dist = 0
        for k in range(1, len(centroids)):
            d = math.sqrt(
                (centroids[k][0] - centroids[k - 1][0]) ** 2
                + (centroids[k][1] - centroids[k - 1][1]) ** 2
            )
            dist += d

        duration = timestamps[-1] - timestamps[0]
        velocity = dist / duration if duration > 0 else 0  # pixel per second

        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
        growth = areas[-1] / (
            areas[0] + 1e-6
        )  # how much the ratio changed between start and end?

        width, height = self.video_writer_dims
        frame_area = width * height

        max_box_area = max(areas)
        screen_coverage = max_box_area / (frame_area + 1e-6)

        start_c = centroids[0]
        end_c = centroids[-1]
        dx = end_c[0] - start_c[0]
        dy = end_c[1] - start_c[1]

        direction = "static"
        min_displacement = 5  # pixels — ignore sub-pixel jitter
        if abs(dx) > min_displacement or abs(dy) > min_displacement:
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"  # Y grows downwards in OpenCV

        # Centrality (0.0 = perfectly centered, 1.0 = at the very corner)
        video_center = (width / 2, height / 2)
        avg_dist_from_center = 0
        max_possible_dist = math.sqrt(video_center[0] ** 2 + video_center[1] ** 2)

        for c in centroids:
            d_center = math.sqrt(
                (c[0] - video_center[0]) ** 2 + (c[1] - video_center[1]) ** 2
            )
            avg_dist_from_center += d_center

        avg_dist_from_center /= len(centroids)
        centrality_score = avg_dist_from_center / max_possible_dist

        # Screen time ratio (fraction of total video this object is visible)
        video_duration = self.total_frames / self.fps
        screen_time_ratio = duration / video_duration if video_duration > 0 else 0.0

        # Positional quadrant (dominant region based on average centroid)
        avg_cx = sum(c[0] for c in centroids) / len(centroids)
        avg_cy = sum(c[1] for c in centroids) / len(centroids)

        col = (
            "left"
            if avg_cx < width / 3
            else ("right" if avg_cx > 2 * width / 3 else "center")
        )
        row = (
            "top"
            if avg_cy < height / 3
            else ("bottom" if avg_cy > 2 * height / 3 else "center")
        )

        if row == "center" and col == "center":
            quadrant = "center"
        elif row == "center":
            quadrant = col
        elif col == "center":
            quadrant = row
        else:
            quadrant = f"{row}-{col}"

        return (
            round(velocity, 2),
            round(growth, 2),
            round(screen_coverage, 3),
            direction,
            round(centrality_score, 2),
            round(screen_time_ratio, 3),
            quadrant,
        )

    def _resolve_identities(self):
        """
        Merge local object IDs into global identities across shots using DINOv2
        cosine similarity and semantic label matching. Returns a local-to-global ID map.
        """
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

                start_b = obj_b["timestamps"][0]
                end_b = obj_b["timestamps"][-1]

                is_overlapping = max(start_a, start_b) < min(end_a, end_b)

                if is_overlapping:
                    continue

                if obj_b["embedding_count"] == 0:
                    continue

                emb_b = obj_b["embedding_sum"] / obj_b["embedding_count"]
                norm_b = np.linalg.norm(emb_b)

                if norm_a == 0 or norm_b == 0:
                    continue

                visual_sim = np.dot(emb_a, emb_b) / (norm_a * norm_b)
                labels_match = self._labels_match(obj_a["label"], obj_b["label"])

                should_merge = False

                if labels_match and visual_sim > self.label_match_merge_threshold:
                    self.logger.info(
                        f"Merge (Standard): {obj_a['label']} matches. Sim: {visual_sim:.2f}"
                    )
                    should_merge = True

                elif visual_sim > self.label_no_match_merge_threshold:
                    self.logger.info(
                        f"Merge (Visual Override): Labels '{obj_a['label']}'/'{obj_b['label']}' differ, but visual sim is high ({visual_sim:.2f})"
                    )
                    should_merge = True

                if should_merge:
                    id_map[id_b] = current_global_id
                    end_a = max(end_a, end_b)

        return id_map

    def _finalize_data(self):
        """Resolve cross-shot identities, compute per-object metrics, and assemble the final output dict."""
        global_id_map = self._resolve_identities()
        final_objects = {}

        for local_id, data in self.object_registry.items():
            g_id = global_id_map.get(local_id, -1)

            (
                velocity,
                growth,
                coverage,
                direction,
                centrality_score,
                screen_time_ratio,
                quadrant,
            ) = self._calculate_metrics(data["boxes"], data["timestamps"])

            occurence_data = {
                "shot_index": data["shot_id"],
                "lifespan": [
                    round(data["timestamps"][0], 2),
                    round(data["timestamps"][-1], 2),
                ],
                "screen_coverage": coverage,
                "velocity_px_sec": velocity,
                "growth_factor": growth,
                "direction": direction,
                "centrality_score": centrality_score,
                "screen_time_ratio": screen_time_ratio,
                "quadrant": quadrant,
            }

            if g_id not in final_objects:
                final_objects[g_id] = {
                    "global_id": g_id,
                    "label": data["label"],
                    "occurrences": [occurence_data],
                }
            else:
                final_objects[g_id]["occurrences"].append(occurence_data)

        final_text = [
            {"second": sec, "text": list(txt_set)}
            for sec, txt_set in sorted(self.text_registry.items())
        ]

        return {
            "global_stats": self.global_stats if hasattr(self, "global_stats") else {},
            "visual_objects": list(final_objects.values()),
            "text_events": final_text,
            "audio_segments": self.audio_registry,  # Add this line
        }

    def visualize_sam_tracking(self, frame_idx, obj_ids, masks):
        """Overlay colored SAM masks and ID labels onto the frame and write it to the output video."""
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

    def _digest_video(self):
        """Run scene detection, compute shot boundaries, and derive global pacing statistics."""
        self.logger.info("--- Step 1: Analyzing Shots ---")
        self.logger.info("Detecting scenes...")

        # 1. Run Scene Detection
        # threshold=27.0 compares adjacent frames' content (HSV).
        # >27 diff = Cut.
        scene_list = detect(self.video_path, ContentDetector(threshold=27.0))

        # Format: [(start_frame, end_frame), ...]
        self.shot_boundaries = [
            (s[0].get_frames(), s[1].get_frames()) for s in scene_list
        ]

        if not self.shot_boundaries:
            self.shot_boundaries = [(0, self.total_frames)]

        self.logger.info(f"Found {len(self.shot_boundaries)} scenes.")

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

        first_shot = shot_data[0]
        has_dynamic_start = first_shot["duration"] < 3.0

        shots_in_first_5s = [s for s in shot_data if s["start"] < 5.0]
        has_quick_pacing_start = len(shots_in_first_5s) >= 5

        rapid_fire_intervals = []

        for start_idx in range(len(shot_data)):
            window_start = shot_data[start_idx]["start"]
            window_end = window_start + 5.0  # Exactly 5 seconds from this start point

            shot_count = 0
            shot_indices = []

            for idx in range(start_idx, len(shot_data)):
                if shot_data[idx]["start"] < window_end:
                    shot_count += 1
                    shot_indices.append(idx)
                else:
                    break

            if shot_count >= 5:
                rapid_fire_intervals.append(
                    {
                        "start_time": window_start,
                        "end_time": window_end,
                        "shot_count": shot_count,
                        "duration": 5.0,
                        "shot_indices": shot_indices,
                    }
                )

        has_quick_pacing_any = len(rapid_fire_intervals) > 0

        self.global_stats = {
            "total_shots": len(self.shot_boundaries),
            "video_duration": round(self.total_frames / self.fps, 2),
            "avg_shot_duration": round(
                sum(s["duration"] for s in shot_data) / len(shot_data), 2
            ),
            "dynamic_start": {
                "detected": has_dynamic_start,
                "first_shot_duration": first_shot["duration"],
                "criteria": "First shot < 3.0s",
            },
            "quick_pacing_intro": {
                "detected": has_quick_pacing_start,
                "shot_count": len(shots_in_first_5s),
                "shots": [s["index"] for s in shots_in_first_5s],
                "criteria": ">= 5 shots starting within t=0s to t=5s",
            },
            "quick_pacing_general": {
                "detected": has_quick_pacing_any,
                "rapid_fire_segments": rapid_fire_intervals,
                "criteria": ">= 5 shots within any 5s window",
            },
        }

        self.logger.info(f" > Analysis Complete. Dynamic Start: {has_dynamic_start}")

    def _save_results_to_json(self, information):
        """Serialize the extraction results dict to extraction_summary.json."""
        output_file = os.path.join(self.artifact_path, "extraction_summary.json")
        try:
            with open(output_file, "w") as f:
                json.dump(information, f, cls=NumpyEncoder, indent=4)
            self.logger.info(f"Extraction results successfully saved to: {output_file}")
        except Exception as e:
            self.logger.error(f"Error saving JSON: {e}")

    def _add_new_tracker(self, box, label):
        """Helper to register the new object with SAM and internal state"""

        new_id = self.get_next_obj_id()

        self.logger.info(f"  + New object {new_id} ({label})")
        self.id_to_label[new_id] = label

        self.active_trackers[new_id] = {
            "box": box,
            "label": label,
        }

        self.sam_model.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=self.current_frame,
            obj_id=new_id,
            box=box,
        )

    def _analyze_audio(self):
        """Transcribe the video audio with Whisper and filter segments below the confidence threshold."""
        self.logger.info("--- Step 2: Transcribing Audio with Whisper ---")

        result = self.audio_model.transcribe(
            audio=self.video_path,
            verbose=False,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )

        self.audio_registry = []

        for segment in result["segments"]:
            # Convert Log Probability to a 0-1 Confidence Score
            # avg_logprob is usually negative (e.g., -0.21). exp(-0.21) ≈ 0.81 (81%)
            confidence = math.exp(segment["avg_logprob"])

            if confidence < self.audio_confidence:
                self.logger.info(
                    f"Skipping audio segment '{segment['text']}' (Conf: {confidence:.2f})"
                )
                continue

            self.audio_registry.append(
                {
                    "start": round(segment["start"], 2),
                    "end": round(segment["end"], 2),
                    "text": segment["text"].strip(),
                    "confidence": round(confidence, 2),  # Useful to save this metric
                }
            )

        self.logger.info(
            f"Audio transcription complete. Kept {len(self.audio_registry)} segments."
        )

    def extract(self):
        """
        Main entry point. Runs the full pipeline: scene analysis, audio transcription,
        per-shot DINO detection + SAM tracking + OCR, identity resolution, and JSON export.
        """
        self._digest_video()
        self._analyze_audio()

        self.logger.info("--- Step 3: Tracking/OCR ---")

        for shot_idx, (start_f, end_f) in enumerate(self.shot_boundaries):
            self.sam_model.reset_state(self.inference_state)

            self.active_trackers = {}
            self.current_frame = start_f

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            ret, frame = self.cap.read()

            self.logger.info(f"Processing Shot {shot_idx}: Frames {start_f} to {end_f}")

            if not ret:
                break

            shot_duration = (end_f - start_f) / self.fps
            num_samples = max(1, min(int(shot_duration), 5))
            shot_length = end_f - start_f

            if num_samples == 1:
                sample_frames = [start_f]
            else:
                sample_frames = [
                    start_f + i * shot_length // (num_samples - 1)
                    for i in range(num_samples)
                ]

            blip_raw_prompts = []
            blip_final_prompts = []

            for sample_f in sample_frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, sample_f)
                ret, sample_frame = self.cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
                    (
                        final_prompt,
                        raw_prompt,
                    ) = self.dingo_prompter.generate_prompt_from_frame(frame_rgb)

                    blip_raw_prompts.append(raw_prompt)
                    blip_final_prompts.append(final_prompt)

            combined_raw_context = self.merge_raw_prompts(blip_raw_prompts)
            combined_final_context = self.merge_dino_prompts(blip_final_prompts)

            dynamic_taxonomy = self.taxonomy_generator.generate_targets(
                self.video_type, scene_context=combined_raw_context
            )
            self.taxonomy_resolver.set_active_targets(dynamic_taxonomy)

            self.logger.info(f"Dino Shot Prompt: {combined_final_context}")

            while self.current_frame < end_f:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame_bgr = self.cap.read()

                if not ret:
                    break

                raw_image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                detected_objects_data = self.dingo_model.detect(
                    raw_image_rgb,
                    text_prompt=combined_final_context,
                    box_threshold=self.dino_box_conf,
                    text_threshold=self.dino_text_conf,
                )

                detected_text_data = self.ocr_engine.detect(raw_image_rgb)

                self.logger.info(
                    f"Detected #{len(detected_objects_data)} general objects in the current frame"
                )

                text_viz_path = os.path.join(
                    self.artifact_path,
                    f"ocr_result_{shot_idx}_{self.current_frame}.jpg",
                )

                face_viz_path = os.path.join(
                    self.artifact_path,
                    f"torch_face_result_{shot_idx}_{self.current_frame}.png",
                )

                object_viz_path = os.path.join(
                    self.artifact_path,
                    f"dino_general_result_{shot_idx}_{self.current_frame}.png",
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

                    self.logger.info(
                        f"trying to match {label} with {self.video_type} taxonomy targets"
                    )

                    best_semantic = self.taxonomy_resolver.resolve(
                        label, threshold=self.word_similarity_threshold
                    )

                    if best_semantic is None:
                        self.logger.warning(
                            f"No proper semantic found for label: {label}. skipping..."
                        )
                        continue

                    if self.is_new_object(box, best_semantic):
                        self._add_new_tracker(box, best_semantic)

                try:
                    boxes, probs = self.face_detection.detect(raw_image_rgb)
                except Exception as e:
                    self.logger.warning(f"MTCNN Error: {e}")
                    boxes, probs = None, None

                if boxes is not None:
                    mapping = []

                    for box, prob in zip(boxes, probs):
                        if prob < self.torch_face_cong:
                            continue

                        x1, y1, x2, y2 = map(int, box)
                        tracker_box = [x1, y1, x2, y2]

                        forced_label = "human face"

                        mapping.append(
                            {
                                "box": tracker_box,
                                "label": forced_label,
                                "score": float(prob),
                            }
                        )

                        if self.is_new_object(tracker_box, forced_label):
                            self._add_new_tracker(tracker_box, forced_label)

                    self.logger.info(
                        f"Detected #{len(mapping)} faces in the current frame"
                    )
                    if mapping:
                        self.dingo_model.map_results(
                            raw_image_rgb, mapping, face_viz_path
                        )

                frames_left_in_shot = end_f - self.current_frame
                frames_to_track = min(self.detection_interval, frames_left_in_shot)

                if frames_to_track <= 0:
                    break

                if len(self.active_trackers) == 0:
                    self.current_frame += frames_to_track
                    continue

                chunk_generator = self.sam_model.propagate_in_video(
                    self.inference_state,
                    start_frame_idx=self.current_frame,
                    max_frame_num_to_track=frames_to_track,
                )

                last_propagated_frame = self.current_frame
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
                    last_propagated_frame = frame_idx

                self.current_frame = last_propagated_frame + 1

        self.logger.info("Video Processing Complete. Finalizing data...")

        information = self._finalize_data()
        self._save_results_to_json(information)

        return information
