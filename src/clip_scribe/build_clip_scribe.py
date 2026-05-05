from src.extractor.taxonomy_core import (
    TaxonomyGenerator,
    TaxonomyResolver,
    generate_hints_from_video_name,
)
from src.extractor.taxonomy_config import ProfilesPile
from src.extractor.extractor_core import InformationExtractor

from src.parser.parser_core import VideoInformationParser

from src.dino.dino_wrapper import DinoDetector
from src.extractor.scene_describer import GPTSceneDescriber

from torchvision import transforms
from facenet_pytorch import MTCNN
import torch
import whisper

from src.ocr.paddle_wrapper import OCRSystem

from src.sam2.sam.build_sam import build_sam2_video_predictor
from src.utils.clip_scribe_logging import logger
from src.utils.clib_scribe_db import ClipScribeWriterDB, ClipScribeReaderDB

from .engine import ClipScribeEngine

from pathlib import Path
import yaml  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = Path(__file__).resolve().parent


def build_clip_scribe(
    video_name: str,
    video_path: str,
    video_type: str | None,
    clib_scribe_device: str,
    user_hints: list[str] | None = None,
) -> ClipScribeEngine:
    try:
        with open(LOCAL_DIR / "configs" / "clip_scribe.yaml") as f:
            _cfg = yaml.safe_load(f)

        clib_scribe_paths = {
            name: PROJECT_ROOT / rel_path for name, rel_path in _cfg["paths"].items()
        }

        models_weights_dir = clib_scribe_paths["checkpoints"]

        dino_params = _cfg["dino"]
        clib_scribe_general_params = _cfg["clib_scribe"]
        face_detection_params = _cfg["face_detection"]
        taxonomy_params = _cfg["taxonomy"]
        audio_params = _cfg["audio"]
        sam2_params = _cfg["sam2"]
        scene_analysis_params = _cfg["scene_analysis"]

        db_params = _cfg.get("database", {})
        db_path = PROJECT_ROOT / db_params.get("path", "data/clip_scribe.db")

        sam2_size: str = sam2_params.get("size", "tiny")

        taxonommy_objects_num: int = taxonomy_params.get("taxonomy_objects_num", 100)
        audio_confidence: float = audio_params.get("audio_confidence", 0.4)

        dino_text_conf: float = dino_params.get("dino_text_conf", 0.4)
        dino_box_conf: float = dino_params.get("dino_box_conf", 0.4)
        dino_size: str = dino_params.get("dino_size", "base")

        torch_face_cong: float = face_detection_params.get("torch_face_cong", 0.9)

        label_match_merge_threshold: float = clib_scribe_general_params.get(
            "label_match_merge_threshold", 0.6
        )
        label_no_match_merge_threshold: float = clib_scribe_general_params.get(
            "label_no_match_merge_threshold", 0.8
        )
        word_similarity_threshold: float = clib_scribe_general_params.get(
            "word_similarity_threshold", 0.4
        )
        detection_interval: int = clib_scribe_general_params.get(
            "detection_interval", 10
        )

        reid_model_frame_check_freq: int = clib_scribe_general_params.get(
            "reid_model_frame_check_freq", 20
        )

        logger.info(f"word_similarity_threshold: {word_similarity_threshold}")

        logger.info(f"dino_text_conf: {dino_text_conf}")
        logger.info(f"dino_box_conf: {dino_box_conf}")

        profiles = ProfilesPile()

        if not user_hints:
            user_hints = generate_hints_from_video_name(video_name, logger)

        taxonomy_resolver = TaxonomyResolver(logger)
        taxonomy_generator = TaxonomyGenerator(
            taxonommy_objects_num, profiles, logger, user_hints=user_hints
        )

        # Scene analysis configuration
        scene_model = scene_analysis_params.get("model", "gpt-4o-mini")
        min_samples = scene_analysis_params.get("min_samples", 1)
        max_samples = scene_analysis_params.get("max_samples", 12)
        sampling_rate = scene_analysis_params.get("sampling_rate", 2.0)
        max_frame_dim = scene_analysis_params.get("max_frame_dim", 512)
        image_detail = scene_analysis_params.get("image_detail", "low")

        dino = DinoDetector(logger, dino_type=dino_size, weights_dir=models_weights_dir)
        scene_describer = GPTSceneDescriber(
            logger,
            model=scene_model,
            max_frame_dim=max_frame_dim,
            image_detail=image_detail,
        )

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

        whisper_device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

        ocr = OCRSystem(logger)
        sam2 = build_sam2_video_predictor(
            sam2_size, "src.sam2.configs", models_weights_dir, logger, sam2_device.type
        )

        logger.info(
            f"Loading DINOv2 (ViT-S/14) for Object Re-Identification on {dino_reid_device.type}..."
        )

        reid_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(
            dino_reid_device.type
        )

        reid_model.eval()

        logger.info(f"loading whisper to {whisper_device.type}")

        audio_model = whisper.load_model("base", device=whisper_device.type)

        embedding_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        face_detection = MTCNN(keep_all=True, device="cpu")  # force cpu

        info_extractor = InformationExtractor(
            video_type,
            video_path,
            video_name,
            sam2,
            dino,
            scene_describer,
            ocr,
            taxonomy_resolver,
            taxonomy_generator,
            reid_model,
            audio_model,
            embedding_transform,
            face_detection,
            clib_scribe_device,
            dino_reid_device.type,
            word_similarity_threshold,
            dino_text_conf,
            dino_box_conf,
            torch_face_cong,
            audio_confidence,
            label_match_merge_threshold,
            label_no_match_merge_threshold,
            logger,
            detection_interval,
            reid_model_frame_check_freq,
            min_samples,
            max_samples,
            sampling_rate,
        )

        info_parser = VideoInformationParser()

        writer_db = ClipScribeWriterDB(db_path=db_path, logger=logger)
        reader_db = ClipScribeReaderDB(db_path=db_path, logger=logger)

        clib_scribe = ClipScribeEngine(
            extractor=info_extractor,
            parser=info_parser,
            logger=logger,
            reader_db=reader_db,
            writer_db=writer_db,
        )

        return clib_scribe

    except Exception as e:
        logger.error(e)
        raise e
