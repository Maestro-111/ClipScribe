from __future__ import annotations

import cv2
import gc
import os

import re
import logging
import numpy as np
import numpy.typing as npt
import math

import json
import torch

from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    NotRequired,
    Protocol,
    Sequence,
    TypedDict,
    TypeAlias,
)

from scenedetect import detect, ContentDetector
from collections import defaultdict
from nltk.corpus import wordnet as wn

from src.utils.cancel import CancellationToken, NullCancellationToken
from src.utils.progress import (
    NullProgressReporter,
    Phase,
    ProgressEvent,
    ProgressReporter,
)
from src.utils.artifacts import run_artifact_dir

logger = logging.getLogger("clip_scribe")

if TYPE_CHECKING:
    import whisper
    from torchvision import transforms
    from facenet_pytorch import MTCNN
    from src.dino.dino_wrapper import DinoDetector
    from src.extractor.scene_describer import GPTSceneDescriber
    from src.ocr.paddle_wrapper import OCRSystem
    from src.extractor.taxonomy_core import TaxonomyGenerator, TaxonomyResolver


Box: TypeAlias = Sequence[float]
Embedding: TypeAlias = npt.NDArray[np.floating[Any]]


class TrackerData(TypedDict):
    box: Box
    label: str


class OCRDetection(TypedDict):
    box: Box
    text: str
    confidence: float
    label: NotRequired[str]


class DetectionResult(TypedDict):
    box: Box
    label: str
    score: float


class ObjectRegistryEntry(TypedDict):
    label: str
    shot_id: int
    embedding_sum: Embedding
    embedding_count: int
    boxes: list[Box]
    timestamps: list[float]
    last_embedding_frame: float


class VisualObjectOccurrence(TypedDict):
    shot_index: int
    lifespan: list[float]
    screen_coverage: float
    velocity_px_sec: float
    growth_factor: float
    direction: str
    centrality_score: float
    screen_time_ratio: float
    quadrant: str


class VisualObjectSummary(TypedDict):
    global_id: int
    label: str
    occurrences: list[VisualObjectOccurrence]


class TextEvent(TypedDict):
    second: int
    text: list[str]


class AudioSegment(TypedDict):
    start: float
    end: float
    text: str
    confidence: float


class SceneDescriptionRecord(TypedDict):
    shot_index: int
    start_time: float
    end_time: float
    description: str


class ShotData(TypedDict):
    index: int
    start: float
    end: float
    duration: float


class RapidFireInterval(TypedDict):
    start_time: float
    end_time: float
    shot_count: int
    duration: float
    shot_indices: list[int]


class FrameDetection(TypedDict):
    shot_index: int
    frame_idx: int
    timestamp_sec: float
    source: str  # dino | ocr | mtcnn | sam_mask
    label: str | None
    text: str | None
    box_x1: float
    box_y1: float
    box_x2: float
    box_y2: float
    confidence: float | None
    object_id: int | None


class ExtractionSummary(TypedDict):
    global_stats: dict[str, object]
    visual_objects: list[VisualObjectSummary]
    text_events: list[TextEvent]
    audio_segments: list[AudioSegment]
    scene_descriptions: list[SceneDescriptionRecord]
    shot_boundaries: list[ShotData]
    frame_detections: list[FrameDetection]


class ReIDModel(Protocol):
    def forward_features(self, x: torch.Tensor) -> Mapping[str, torch.Tensor]:
        ...


class NumpyEncoder(json.JSONEncoder):
    """
    helper to encode information to json
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class VideoInformationExtractor:

    """
    Process objects, text, moving history and record in json/csv
    """

    @staticmethod
    def get_schema_descriptions() -> dict[str, dict[str, str]]:
        """
        Returns a flat dictionary mapping DB table names which are going to be used by LangChain later to their column descriptions.
        Structure: {table_name: {column_name: description_string}}
        This is persisted to the field_descriptions table for self-documenting DB schemas.
        """
        return {
            "runs": {
                "run_id": "Unique identifier for this extraction run.",
                "video_name": "Name of the video file processed.",
                "video_path": "Full path to the video file.",
                "video_type": "Category of video (e.g., 'car ad').",
                "created_at": "Timestamp when this run was created.",
            },
            "global_stats": {
                "total_shots": "Total number of distinct shots (scene cuts) detected in the video.",
                "video_duration": "Total video length in seconds.",
                "avg_shot_duration": "Mean duration of a single shot in seconds. Lower values indicate faster editing pace.",
                "dynamic_start_detected": "Boolean. True if the first shot is shorter than the criteria threshold.",
                "dynamic_start_first_shot_dur": "Duration of the very first shot in seconds.",
                "dynamic_start_criteria": "Human-readable rule used for detection (e.g., 'First shot < 3.0s').",
                "qp_intro_detected": "Boolean. True if the number of shots starting within t=0s to t=5s meets the threshold.",
                "qp_intro_shot_count": "Number of shots that start within the first 5 seconds.",
                "qp_intro_shots": "JSON list of shot indices that fall within the first 5 seconds.",
                "qp_intro_criteria": "Human-readable rule used for detection.",
                "qp_general_detected": "Boolean. True if at least one 5-second window meets the threshold.",
                "qp_general_rapid_fire_segments": "JSON list of 5-second windows that qualify as rapid-fire. Each contains start_time, end_time, shot_count, duration, shot_indices.",
                "qp_general_criteria": "Human-readable rule used for detection.",
            },
            "visual_object_occurrences": {
                "global_id": "Unique integer ID assigned after cross-shot identity resolution.",
                "label": "Semantic label for this object (e.g., 'car', 'human face'). Resolved via SBERT taxonomy matching.",
                "shot_index": "Zero-based index of the shot this occurrence belongs to.",
                "lifespan_start": "Timestamp (seconds) when the object was first tracked within this shot.",
                "lifespan_end": "Timestamp (seconds) when the object was last tracked within this shot.",
                "screen_coverage": "Fraction of total frame area occupied by the object's largest bounding box (0.0 to 1.0).",
                "velocity_px_sec": "Average speed of the object's centroid in pixels per second.",
                "growth_factor": "Ratio of bounding-box area at end vs. start of tracking. >1.0 means it approached the camera.",
                "direction": "Dominant movement direction: 'left', 'right', 'up', 'down', or 'static'.",
                "centrality_score": "Distance from frame center (0.0 = centered, 1.0 = corner). Lower = more focal.",
                "screen_time_ratio": "Fraction of total video duration this object was visible in this shot.",
                "quadrant": "Dominant screen region based on average centroid in a 3x3 grid.",
            },
            "text_events": {
                "second": "The integer second of the video (e.g., 0 = first second).",
                "line_index": "Zero-based index for multiple text lines detected in the same second.",
                "text": "Distinct text string detected on screen.",
            },
            "audio_segments": {
                "start_time": "Start time of the speech segment in seconds.",
                "end_time": "End time of the speech segment in seconds.",
                "text": "The transcribed speech content.",
                "confidence": "Confidence score (0.0 to 1.0) derived from Whisper's average log-probability.",
            },
            "scene_descriptions": {
                "shot_index": "Zero-based index of the shot this description belongs to.",
                "start_time": "Start time of the shot in seconds.",
                "end_time": "End time of the shot in seconds.",
                "description": "Rich narrative scene description generated by GPT vision analysis.",
            },
        }

    @staticmethod
    def _calculate_iou(boxA: Box, boxB: Box) -> float:
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
    def _mask_to_box(mask: npt.NDArray[Any]) -> list[int] | None:
        """Converts SAM mask to bounding box [x1, y1, x2, y2]"""
        if len(mask.shape) == 3:
            mask = mask[0]
        y, x = np.where(mask > 0)
        if len(x) == 0:
            return None
        return [int(np.min(x)), int(np.min(y)), int(np.max(x)), int(np.max(y))]

    @staticmethod
    def _labels_match(label_a: str, label_b: str) -> bool:
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
    def _is_valid_text(
        text_data: OCRDetection, frame_height: int, max_text_height: float
    ) -> bool:
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
        sam_model: Any,
        dino_model: "DinoDetector",
        scene_describer: "GPTSceneDescriber",
        ocr_engine: "OCRSystem",
        taxonomy_resolver: "TaxonomyResolver",
        taxonomy_generator: "TaxonomyGenerator",
        taxonomy_user_hints: list[str] | None,
        reid_model: ReIDModel,
        audio_model: "whisper.Whisper",
        embedding_transform: "transforms.Compose",
        face_detection: "MTCNN",
        device: str,
        word_similarity_threshold: float,
        dino_text_conf: float,
        dino_box_conf: float,
        torch_face_cong: float,
        audio_confidence: float,
        label_match_merge_threshold: float,
        label_no_match_merge_threshold: float,
        detection_interval: int = 10,
        reid_model_frame_check_freq: int = 20,
        reid_similarity_difference: float = 0.8,
        min_samples: int = 1,
        max_samples: int = 12,
        sampling_rate: float = 2.0,
        progress_reporter: "ProgressReporter | None" = None,
        cancel_token: "CancellationToken | None" = None,
        max_artifact_files: int | None = None,
    ) -> None:
        # helper models

        self.sam_model = sam_model
        self.ocr_engine = ocr_engine
        self.scene_describer = scene_describer

        self.dingo_model = dino_model
        self.audio_model = audio_model

        self.embedding_transform = embedding_transform
        self.reid_model = reid_model

        self.face_detection = face_detection
        self.detection_interval = detection_interval

        # OpenCV and SAM2 create these per input video.  Their types are
        # runtime-library objects, so keep the boundary untyped while ensuring
        # cleanup can safely run after a partially initialized extraction.
        self.cap: Any = None
        self.video_writer: Any = None
        self.inference_state: Any = None

        self.current_frame = 0
        self.obj_id_counter = 1

        self.active_trackers: dict[int, TrackerData] = {}
        self.id_to_label: dict[int, str] = {}

        self.text_registry: dict[int, set[str]] = defaultdict(set)
        self.object_registry: dict[int, ObjectRegistryEntry] = {}

        self.taxonomy_resolver = taxonomy_resolver
        self.taxonomy_generator = taxonomy_generator
        self.taxonomy_user_hints = taxonomy_user_hints

        self.word_similarity_threshold = word_similarity_threshold

        self.dino_text_conf = dino_text_conf
        self.dino_box_conf = dino_box_conf

        self.torch_face_cong = torch_face_cong

        self.label_match_merge_threshold = label_match_merge_threshold
        self.label_no_match_merge_threshold = label_no_match_merge_threshold

        self.device = device

        self.audio_registry: list[AudioSegment] = []
        self.scene_description_registry: list[SceneDescriptionRecord] = []
        self.global_stats: dict[str, object] = {}
        self.audio_confidence = audio_confidence

        self.reid_model_frame_check_freq = reid_model_frame_check_freq
        self.reid_similarity_difference = reid_similarity_difference

        # Adaptive frame sampling parameters
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.sampling_rate = sampling_rate

        self.progress = progress_reporter or NullProgressReporter()
        # Cooperative-cancel token, polled at checkpoints in extract(). Null
        # (never canceled) for CLI/tests; Redis-backed in the web paths.
        self._cancel = cancel_token or NullCancellationToken()

        # Raw per-(frame, box) detections for the UI overlay; persisted to the
        # frame_detections table by the writer. Per-shot boundaries (seconds)
        # for the timeline view, captured in _digest_video.
        self.frame_detections: list[FrameDetection] = []
        self.shot_data: list[ShotData] = []
        self.shot_boundaries: list[tuple[int, int]] = []

        # Cap on per-frame visualization PNGs written to the artifact dir
        # (None = unlimited). The tracked mp4 and extraction_summary.json are
        # always kept. Counts PNGs written so far this run.
        self.max_artifact_files = max_artifact_files
        self._viz_files_written = 0

    def __repr__(self) -> str:
        return f"InformationExtractor: device: {self.device}"

    def _record_detection(
        self,
        *,
        shot_index: int,
        frame_idx: int,
        source: str,
        box: Sequence[float],
        label: str | None = None,
        text: str | None = None,
        confidence: float | None = None,
        object_id: int | None = None,
    ) -> None:
        """Append one raw detection (for the frame_detections table / overlay)."""
        self.frame_detections.append(
            {
                "shot_index": shot_index,
                "frame_idx": frame_idx,
                "timestamp_sec": round(frame_idx / self.fps, 3),
                "source": source,
                "label": label,
                "text": text,
                "box_x1": float(box[0]),
                "box_y1": float(box[1]),
                "box_x2": float(box[2]),
                "box_y2": float(box[3]),
                "confidence": float(confidence) if confidence is not None else None,
                "object_id": object_id,
            }
        )

    def _take_viz_slot(self) -> bool:
        """Reserve one per-frame viz-PNG write under the max_artifact_files cap.

        Returns True (and consumes a slot) while under the cap, else False so
        the caller skips the write. None means unlimited.
        """
        if (
            self.max_artifact_files is not None
            and self._viz_files_written >= self.max_artifact_files
        ):
            return False
        self._viz_files_written += 1
        return True

    def _state_init(self, artifact_path: str, video_path: str) -> None:
        """Open the video capture, initialize the video writer, and set up SAM inference state."""
        if not os.path.exists(artifact_path):
            os.makedirs(artifact_path)

        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.video_writer_dims = (width, height)

        if fps <= 0:
            fps = 30.0

        self.fps = fps

        output_filename = os.path.join(artifact_path, "tracked_output.mp4")

        # Prefer H.264 (avc1) for a compact, widely-playable file. In some
        # environments the avc1 writer can't open — notably Linux containers,
        # where OpenCV's ffmpeg resolves H.264 to the hardware encoder
        # h264_v4l2m2m, which needs a /dev/video* device that isn't present.
        # Fall back to mp4v (MPEG-4 Part 2, pure software, always available) so
        # tracked_output.mp4 is still written.
        self.video_writer = cv2.VideoWriter(
            output_filename, cv2.VideoWriter.fourcc(*"avc1"), fps, (width, height)
        )
        if not self.video_writer.isOpened():
            logger.warning(
                "avc1/H.264 VideoWriter unavailable (no hardware encoder in this "
                "environment); falling back to mp4v for tracked_output.mp4"
            )
            self.video_writer.release()
            self.video_writer = cv2.VideoWriter(
                output_filename, cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
            )
        logger.info(f"Recording video to: {output_filename}")

        self.inference_state = self.sam_model.init_state(video_path=video_path)
        self.total_frames = self.inference_state["num_frames"]

        logger.info(f"Video opened: {video_path}")
        logger.info(f"Video FPS: {fps}; Total Frames: {self.total_frames}")

    def cleanup(self) -> None:
        """Release resources owned by this extraction run.

        The detector, SAM2 predictor, Whisper model, and other model objects are
        owned by the long-lived ``ClipScribeBuilder`` and intentionally remain
        loaded between Celery tasks.  The SAM2 *inference state*, however, is
        created for one video and can retain decoded frames and accelerator
        tensors.  Drop that state explicitly so a worker can process a second
        video without carrying the previous video's memory forward.
        """

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            logger.info("Video writer released. Output saved.")

        inference_state = getattr(self, "inference_state", None)
        if inference_state is not None:
            try:
                self.sam_model.reset_state(inference_state)
            except Exception:  # noqa: BLE001 - cleanup must not mask task failures
                logger.warning("Failed to reset SAM2 inference state", exc_info=True)
            finally:
                self.inference_state = None

        # These are all owned by this per-job extractor.  Rebind instead of
        # clearing in place: the completed extraction summary may still hold
        # references to the old lists/dicts while the engine persists it.
        self.active_trackers = {}
        self.id_to_label = {}
        self.text_registry = defaultdict(set)
        self.object_registry = {}
        self.audio_registry = []
        self.scene_description_registry = []
        self.global_stats = {}
        self.frame_detections = []
        self.shot_data = []
        self.shot_boundaries = []
        self.current_frame = 0
        self.obj_id_counter = 1

        gc.collect()

        if self.device == "mps" and torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
                torch.mps.empty_cache()
            except Exception:  # noqa: BLE001 - best-effort accelerator cleanup
                logger.warning("Failed to release MPS cached memory", exc_info=True)

        if self.device == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - best-effort accelerator cleanup
                logger.warning("Failed to release CUDA cached memory", exc_info=True)

    def _is_new_object(self, new_box: Box, new_label: str) -> bool:
        """
        Returns True if the box does NOT overlap significantly with
        an active tracker OF THE SAME CLASS.
        """
        for obj_id, tracker_data in self.active_trackers.items():
            active_box = tracker_data["box"]
            active_label = tracker_data["label"]

            iou = self._calculate_iou(new_box, active_box)

            logger.info(
                f"Object {obj_id} label {active_label} has {iou} iou with new box label {new_label}"
            )

            if iou > 0.5:
                if self._labels_match(new_label, active_label):
                    return False

        return True

    def _get_next_obj_id(self) -> int:
        """Return the next available object ID and increment the counter."""
        current_id = self.obj_id_counter
        self.obj_id_counter += 1
        return current_id

    def _extract_embedding(
        self, frame_bgr: npt.NDArray[Any], box: Box
    ) -> Embedding | None:
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

    def _save_metadata(
        self,
        frame_idx: int,
        obj_ids: Sequence[int],
        masks: torch.Tensor,
        frame_text: Sequence[OCRDetection],
        shot_idx: int,
        current_frame_img: npt.NDArray[Any],
    ) -> None:
        """
        Record per-frame tracking data: filter and store OCR text, convert SAM masks
        to bounding boxes, and periodically accumulate DINOv2 embeddings for re-ID.
        """
        timestamp = frame_idx / self.fps
        second_key = int(timestamp)
        masks_np = masks.cpu().numpy()

        h, w, _ = current_frame_img.shape

        max_text_height = 0.0
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

                                if cos_sim < self.reid_similarity_difference:
                                    self.object_registry[obj_id][
                                        "embedding_sum"
                                    ] += new_emb
                                    self.object_registry[obj_id]["embedding_count"] += 1

                                    logger.info(
                                        f"New viewpoint for ID {obj_id} captured. Sim: {cos_sim:.2f}"
                                    )
                        self.object_registry[obj_id]["last_embedding_frame"] = frame_idx

                self.object_registry[obj_id]["boxes"].append(current_box)
                self.object_registry[obj_id]["timestamps"].append(timestamp)

                self._record_detection(
                    shot_index=shot_idx,
                    frame_idx=frame_idx,
                    source="sam_mask",
                    box=current_box,
                    label=label,
                    object_id=obj_id,
                )

                self.active_trackers[obj_id] = {"box": current_box, "label": label}
            else:
                self.active_trackers.pop(obj_id, None)

    def _calculate_metrics(
        self, boxes: Sequence[Box], timestamps: Sequence[float]
    ) -> tuple[float, float, float, str, float, float, str]:
        """
        Derive motion and spatial metrics from a sequence of bounding boxes:
        velocity, growth factor, screen coverage, direction, centrality, screen time, and quadrant.
        """
        if len(boxes) < 2:
            return 0.0, 0.0, 0.0, "unknown", 0.0, 0.0, "center"

        centroids = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
        dist = 0.0
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
        avg_dist_from_center = 0.0
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

    def _resolve_identities(self) -> dict[int, int]:
        """
        Merge local object IDs into global identities across shots using DINOv2
        cosine similarity and semantic label matching. Returns a local-to-global ID map.
        """
        logger.info("Resolving identities across shots...")

        id_map: dict[int, int] = {}
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
                    logger.info(
                        f"Merge (Standard): {obj_a['label']} matches. Sim: {visual_sim:.2f}"
                    )
                    should_merge = True

                elif visual_sim > self.label_no_match_merge_threshold:
                    logger.info(
                        f"Merge (Visual Override): Labels '{obj_a['label']}'/'{obj_b['label']}' differ, but visual sim is high ({visual_sim:.2f})"
                    )
                    should_merge = True

                if should_merge:
                    id_map[id_b] = current_global_id
                    end_a = max(end_a, end_b)
                    self.progress.emit(
                        ProgressEvent.IDENTITY_MERGED,
                        {
                            "from_ids": [id_a, id_b],
                            "to_global_id": current_global_id,
                            "similarity": round(float(visual_sim), 2),
                        },
                    )

        return id_map

    def _frame_detections_with_global_object_ids(
        self, global_id_map: Mapping[int, int]
    ) -> list[FrameDetection]:
        final_detections: list[FrameDetection] = []

        for detection in self.frame_detections:
            final_detection = detection.copy()
            local_object_id = detection["object_id"]
            final_detection["object_id"] = (
                global_id_map.get(local_object_id)
                if local_object_id is not None
                else None
            )
            final_detections.append(final_detection)

        return final_detections

    def _finalize_data(self) -> ExtractionSummary:
        """Resolve cross-shot identities, compute per-object metrics, and assemble the final output dict."""
        global_id_map = self._resolve_identities()
        final_objects: dict[int, VisualObjectSummary] = {}

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

            occurence_data: VisualObjectOccurrence = {
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

        final_text: list[TextEvent] = [
            {"second": sec, "text": list(txt_set)}
            for sec, txt_set in sorted(self.text_registry.items())
        ]

        return {
            "global_stats": self.global_stats,
            "visual_objects": list(final_objects.values()),
            "text_events": final_text,
            "audio_segments": self.audio_registry,
            "scene_descriptions": self.scene_description_registry,
            "shot_boundaries": self.shot_data,
            "frame_detections": self._frame_detections_with_global_object_ids(
                global_id_map
            ),
        }

    def visualize_sam_tracking(
        self,
        frame_idx: int,
        obj_ids: Sequence[int],
        masks: torch.Tensor | npt.NDArray[Any],
    ) -> None:
        """Overlay colored SAM masks and ID labels onto the frame and write it to the output video."""
        if not self.video_writer.isOpened():
            logger.error("Error: Video Writer is NOT open.")
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            logger.error(f"Error: Could not read frame {frame_idx}.")
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
            logger.info(f"Wrote frame {frame_idx} to video.")

    def _digest_video(self, video_path: str) -> None:
        """Run scene detection, compute shot boundaries, and derive global pacing statistics."""
        logger.info("--- Step 1: Analyzing Shots ---")
        logger.info("Detecting scenes...")

        # 1. Run Scene Detection
        # threshold=27.0 compares adjacent frames' content (HSV).
        # >27 diff = Cut.
        scene_list = detect(video_path, ContentDetector(threshold=27.0))

        # Format: [(start_frame, end_frame), ...]
        self.shot_boundaries = [
            (s[0].get_frames(), s[1].get_frames()) for s in scene_list
        ]

        if not self.shot_boundaries:
            self.shot_boundaries = [(0, self.total_frames)]

        logger.info(f"Found {len(self.shot_boundaries)} scenes.")

        shot_data: list[ShotData] = []

        for i, (start_f, end_f) in enumerate(self.shot_boundaries):
            shot_data.append(
                {
                    "index": i,
                    "start": round(start_f / self.fps, 3),
                    "end": round(end_f / self.fps, 3),
                    "duration": round((end_f - start_f) / self.fps, 3),
                }
            )

        # Persisted to shot_boundaries (timeline view); also feeds global_stats.
        self.shot_data = shot_data

        first_shot = shot_data[0]
        has_dynamic_start = first_shot["duration"] < 3.0

        shots_in_first_5s = [s for s in shot_data if s["start"] < 5.0]
        has_quick_pacing_start = len(shots_in_first_5s) >= 5

        rapid_fire_intervals: list[RapidFireInterval] = []

        for start_idx in range(len(shot_data)):
            window_start = shot_data[start_idx]["start"]
            window_end = window_start + 5.0  # Exactly 5 seconds from this start point

            shot_count = 0
            shot_indices: list[int] = []

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

        logger.info(f" > Analysis Complete. Dynamic Start: {has_dynamic_start}")

    def _save_results_to_json(
        self, information: ExtractionSummary, artifact_path: str
    ) -> None:
        """Serialize the extraction results dict to extraction_summary.json."""
        output_file = os.path.join(artifact_path, "extraction_summary.json")
        try:
            with open(output_file, "w") as f:
                json.dump(information, f, cls=NumpyEncoder, indent=4)
            logger.info(f"Extraction results successfully saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving JSON: {e}")

    def _add_new_tracker(self, box: Box, label: str) -> None:
        """Helper to register the new object with SAM and internal state"""

        new_id = self._get_next_obj_id()

        logger.info(f"New object {new_id} ({label})")
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

    def _analyze_audio(self, video_path: str) -> None:
        """Transcribe the video audio with Whisper and filter segments below the confidence threshold."""
        logger.info("--- Step 2: Transcribing Audio with Whisper ---")

        result = self.audio_model.transcribe(
            audio=video_path,
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
                logger.info(
                    f"Skipping audio segment '{segment['text']}' (Conf: {confidence:.2f})"
                )
                continue

            kept_segment: AudioSegment = {
                "start": round(segment["start"], 2),
                "end": round(segment["end"], 2),
                "text": segment["text"].strip(),
                "confidence": round(confidence, 2),  # Useful to save this metric
            }
            self.audio_registry.append(kept_segment)

            self.progress.emit(ProgressEvent.AUDIO_SEGMENT, dict(kept_segment))

        logger.info(
            f"Audio transcription complete. Kept {len(self.audio_registry)} segments."
        )

    def extract(
        self,
        video_type: str | None,
        video_path: str,
        video_name: str,
        run_id: str,
    ) -> ExtractionSummary:
        """
        Main entry point. Runs the full pipeline: scene analysis, audio transcription,
        per-shot DINO detection + SAM tracking + OCR, identity resolution, and JSON export.
        """

        # Keyed by run_id (not video_name) so repeated jobs over the same video
        # never collide; this dir is what the remote uploader bundles.
        artifact_path = run_artifact_dir(run_id)
        self._state_init(artifact_path, video_path)

        self._cancel.check()
        self.progress.phase_started(Phase.SCENE_DETECTION)
        self._digest_video(video_path)
        self.progress.phase_completed(
            Phase.SCENE_DETECTION,
            {
                "total_shots": len(self.shot_boundaries),
                "video_duration": self.global_stats.get("video_duration"),
            },
        )

        # Whisper transcription is a single, minutes-long, uninterruptible call;
        # check right before it so a cancel during scene detection skips it.
        self._cancel.check()
        self.progress.phase_started(Phase.AUDIO)
        self._analyze_audio(video_path)
        self.progress.phase_completed(
            Phase.AUDIO, {"segments_kept": len(self.audio_registry)}
        )

        logger.info("--- Step 3: Tracking/OCR ---")

        self.progress.phase_started(
            Phase.SHOT_PROCESSING, {"total_shots": len(self.shot_boundaries)}
        )

        for shot_idx, (start_f, end_f) in enumerate(self.shot_boundaries):
            self._cancel.check()
            self.progress.emit(
                ProgressEvent.SHOT_STARTED,
                {
                    "shot_idx": shot_idx,
                    "start": round(start_f / self.fps, 2),
                    "end": round(end_f / self.fps, 2),
                },
            )
            self.sam_model.reset_state(self.inference_state)

            self.active_trackers = {}
            self.current_frame = start_f

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            ret, frame = self.cap.read()

            logger.info(f"Processing Shot {shot_idx}: Frames {start_f} to {end_f}")

            if not ret:
                break

            # Adaptive frame sampling based on shot duration
            shot_duration = (end_f - start_f) / self.fps
            num_samples = max(
                self.min_samples,
                min(
                    self.max_samples,
                    int(math.ceil(self.sampling_rate * math.sqrt(shot_duration))),
                ),
            )
            shot_length = end_f - start_f

            logger.info(
                f"Shot duration: {shot_duration:.2f}s -> sampling {num_samples} frames"
            )

            if num_samples == 1:
                sample_frames = [start_f]
            else:
                sample_frames = [
                    start_f + i * shot_length // (num_samples - 1)
                    for i in range(num_samples)
                ]

            # Collect sampled RGB frames for GPT vision analysis
            sampled_rgb_frames = []
            for sample_f in sample_frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, sample_f)
                ret, sample_frame = self.cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
                    sampled_rgb_frames.append(frame_rgb)

            # Single GPT call processes all sampled frames
            (
                combined_final_context,
                combined_raw_context,
            ) = self.scene_describer.describe_scene(sampled_rgb_frames)

            self.scene_description_registry.append(
                {
                    "shot_index": shot_idx,
                    "start_time": round(start_f / self.fps, 2),
                    "end_time": round(end_f / self.fps, 2),
                    "description": combined_raw_context,
                }
            )

            self.progress.emit(
                ProgressEvent.SHOT_SCENE_DESCRIBED,
                {
                    "shot_idx": shot_idx,
                    "description": combined_raw_context,
                    "dino_prompt": combined_final_context,
                },
            )

            taxonomy_prompt = self.taxonomy_generator.generate_taxonomy_prompt(
                video_type,
                scene_context=combined_raw_context,
                dino_prompt=combined_final_context,
                user_hints=self.taxonomy_user_hints,
            )

            taxonomy_prompt_path = os.path.join(
                artifact_path,
                f"taxonomy_prompt_{shot_idx}.txt",
            )

            with open(taxonomy_prompt_path, "w", encoding="utf-8") as file:
                file.write(taxonomy_prompt)
                logger.info(f"Taxonomy Prompt saved to: {taxonomy_prompt_path}")

            dynamic_taxonomy = self.taxonomy_generator.generate_taxonomy_targets(
                taxonomy_prompt
            )

            self.taxonomy_resolver.set_active_targets(dynamic_taxonomy)

            self.progress.emit(
                ProgressEvent.SHOT_TAXONOMY_RESOLVED,
                {
                    "shot_idx": shot_idx,
                    "targets": [item.anchor for item in dynamic_taxonomy],
                },
            )

            logger.info(f"Dino Shot Prompt: {combined_final_context}")

            while self.current_frame < end_f:
                # Finest checkpoint: each iteration is a detection batch (DINO +
                # OCR + MTCNN + SAM) worth seconds of GPU work, so a per-batch
                # Redis check is negligible overhead and bounds cancel latency.
                self._cancel.check()
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

                logger.info(
                    f"Detected #{len(detected_objects_data)} general objects in the current frame"
                )

                text_viz_path = os.path.join(
                    artifact_path,
                    f"ocr_result_{shot_idx}_{self.current_frame}.png",
                )

                face_viz_path = os.path.join(
                    artifact_path,
                    f"torch_face_result_{shot_idx}_{self.current_frame}.png",
                )

                object_viz_path = os.path.join(
                    artifact_path,
                    f"dino_general_result_{shot_idx}_{self.current_frame}.png",
                )

                for text_box in detected_text_data:
                    self._record_detection(
                        shot_index=shot_idx,
                        frame_idx=self.current_frame,
                        source="ocr",
                        box=text_box["box"],
                        text=text_box.get("text"),
                        confidence=text_box.get("confidence"),
                    )

                if self._take_viz_slot():
                    self.ocr_engine.save_visualization(
                        raw_image_rgb, detected_text_data, text_viz_path
                    )

                if self._take_viz_slot():
                    self.dingo_model.map_results(
                        raw_image_rgb, detected_objects_data, object_viz_path
                    )

                for box_info in detected_objects_data:
                    box = box_info["box"]
                    label = box_info["label"]

                    logger.info(
                        f"trying to match dino label {label} with {video_type} taxonomy targets"
                    )

                    best_semantic = self.taxonomy_resolver.resolve(
                        label, threshold=self.word_similarity_threshold
                    )

                    if best_semantic is None:
                        logger.warning(
                            f"No proper semantic found for dino label: {label}. skipping..."
                        )
                        continue

                    self._record_detection(
                        shot_index=shot_idx,
                        frame_idx=self.current_frame,
                        source="dino",
                        box=box,
                        label=best_semantic,
                        confidence=box_info.get("score"),
                    )

                    if self._is_new_object(box, best_semantic):
                        self._add_new_tracker(box, best_semantic)

                num_faces = 0

                try:
                    boxes, probs = self.face_detection.detect(raw_image_rgb)
                except Exception as e:
                    logger.warning(f"MTCNN Error: {e}")
                    boxes, probs = None, None

                if boxes is not None:
                    mapping: list[DetectionResult] = []

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

                        self._record_detection(
                            shot_index=shot_idx,
                            frame_idx=self.current_frame,
                            source="mtcnn",
                            box=tracker_box,
                            label=forced_label,
                            confidence=float(prob),
                        )

                        if self._is_new_object(tracker_box, forced_label):
                            self._add_new_tracker(tracker_box, forced_label)

                    num_faces = len(mapping)
                    logger.info(f"Detected #{num_faces} faces in the current frame")
                    if mapping and self._take_viz_slot():
                        self.dingo_model.map_results(
                            raw_image_rgb, mapping, face_viz_path
                        )

                self.progress.emit(
                    ProgressEvent.SHOT_FRAME_PROCESSED,
                    {
                        "shot_idx": shot_idx,
                        "frame_idx": self.current_frame,
                        "detections": len(detected_objects_data),
                        "ocr_lines": len(detected_text_data),
                        "faces": num_faces,
                    },
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

                    self._save_metadata(
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

            self.progress.emit(
                ProgressEvent.SHOT_COMPLETED,
                {"shot_idx": shot_idx, "objects_tracked": len(self.active_trackers)},
            )

        self.progress.phase_completed(Phase.SHOT_PROCESSING)

        logger.info("Video Processing Complete. Finalizing data...")

        self.progress.phase_started(Phase.FINALIZE)
        information = self._finalize_data()
        self.progress.phase_completed(Phase.FINALIZE)
        self._save_results_to_json(information, artifact_path)

        return information
