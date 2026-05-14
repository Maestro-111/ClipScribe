from src.extractor.taxonomy_core import (
    TaxonomyGenerator,
    TaxonomyResolver,
    generate_hints_from_video_name,
)
from src.extractor.taxonomy_config import ProfilesPile
from src.extractor.extractor_core import VideoInformationExtractor

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
from src.db import ClipScribeWriterDB, ClipScribeReaderDB, create_db_engine

from .engine import ClipScribeEngine
from .platform_configs import BasePlatformConf

from pathlib import Path
import os
import yaml  # type: ignore
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = Path(__file__).resolve().parent


def build_parser(
    clib_scribe_platform_name: str,
    clib_scribe_platform_conf: BasePlatformConf,
    clib_scribe_parser_params: dict,
    parser_agent_params: dict,
) -> VideoInformationParser:
    parser_output_dir = PROJECT_ROOT / clib_scribe_parser_params.get(
        "output_dir", "parser_artifacts"
    )

    parser_max_parallel = clib_scribe_parser_params.get("max_parallel_agents", 5)
    recursion_limit = clib_scribe_parser_params.get("recursion_limit", 25)

    info_parser = VideoInformationParser(
        agent=parser_agent_params,
        platform_name=clib_scribe_platform_name,
        platform_config=clib_scribe_platform_conf,
        output_dir=str(parser_output_dir),
        logger=logger,
        max_parallel_agents=parser_max_parallel,
        recursion_limit=recursion_limit,
    )

    return info_parser


def build_extractor(
    video_name: str,
    user_hints: list[str] | None,
    clib_scribe_device: str,
    models_weights_dir: str,
    clib_scribe_extractor_params: dict,
    dino_params: dict,
    scene_analysis_params: dict,
    face_detection_params: dict,
    taxonomy_params: dict,
    audio_params: dict,
    sam2_params: dict,
):
    sam2_size: str = sam2_params.get("size", "tiny")

    taxonommy_objects_num: int = taxonomy_params.get("taxonomy_objects_num", 100)
    audio_confidence: float = audio_params.get("audio_confidence", 0.4)

    dino_text_conf: float = dino_params.get("dino_text_conf", 0.4)
    dino_box_conf: float = dino_params.get("dino_box_conf", 0.4)
    dino_size: str = dino_params.get("dino_size", "base")

    torch_face_cong: float = face_detection_params.get("torch_face_cong", 0.9)

    label_match_merge_threshold: float = clib_scribe_extractor_params.get(
        "label_match_merge_threshold", 0.6
    )
    label_no_match_merge_threshold: float = clib_scribe_extractor_params.get(
        "label_no_match_merge_threshold", 0.8
    )
    word_similarity_threshold: float = clib_scribe_extractor_params.get(
        "word_similarity_threshold", 0.4
    )
    detection_interval: int = clib_scribe_extractor_params.get("detection_interval", 10)

    reid_model_frame_check_freq: int = clib_scribe_extractor_params.get(
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    face_detection = MTCNN(keep_all=True, device="cpu")  # force cpu

    info_extractor = VideoInformationExtractor(
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

    return info_extractor


def build_clip_scribe(
    video_name: str,
    video_path: str,
    video_type: str | None,
    clib_scribe_mode: str,
    clib_scribe_device: str,
    clib_scribe_platform_name: str,
    clib_scribe_platform_conf: BasePlatformConf,
    user_hints: list[str] | None = None,
) -> ClipScribeEngine:
    try:
        load_dotenv()

        with open(LOCAL_DIR / "configs" / "clip_scribe.yaml") as f:
            _cfg = yaml.safe_load(f)

        clib_scribe_paths = {
            name: PROJECT_ROOT / rel_path for name, rel_path in _cfg["paths"].items()
        }

        models_weights_dir = clib_scribe_paths["checkpoints"]

        dino_params = _cfg["dino"]
        clip_scribe_params = _cfg["clip_scribe"]
        clib_scribe_extractor_params = clip_scribe_params["extractor"]
        face_detection_params = _cfg["face_detection"]
        taxonomy_params = _cfg["taxonomy"]
        audio_params = _cfg["audio"]
        sam2_params = _cfg["sam2"]
        scene_analysis_params = _cfg["scene_analysis"]

        clib_scribe_parser_params = clip_scribe_params.get("parser", {})
        parser_agent_params = clib_scribe_parser_params.get("agent", {})

        db_params = _cfg.get("database", {})
        db_backend = db_params.get("backend", "sqlite")

        if db_backend == "sqlite":
            db_url = os.environ.get("SQLITE_URL", "sqlite:///data/clip_scribe.db")
            if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:////"):
                relative_path = db_url[len("sqlite:///") :]
                db_url = f"sqlite:///{PROJECT_ROOT / relative_path}"
        else:
            db_url = os.environ["POSTGRESQL_URL"]

        db_engine = create_db_engine(
            database_url=db_url,
            pool_size=db_params.get("pool_size", 5),
            max_overflow=db_params.get("max_overflow", 10),
            logger=logger,
        )

        writer_db = ClipScribeWriterDB(engine=db_engine, logger=logger)
        reader_db = ClipScribeReaderDB(engine=db_engine, logger=logger)

        info_extractor = None
        info_parser = None

        if clib_scribe_mode in ["full", "extract"]:
            info_extractor = build_extractor(
                video_name,
                user_hints,
                clib_scribe_device,
                models_weights_dir,
                clib_scribe_extractor_params,
                dino_params,
                scene_analysis_params,
                face_detection_params,
                taxonomy_params,
                audio_params,
                sam2_params,
            )

        if clib_scribe_mode in ["full", "parse"]:
            info_parser = build_parser(
                clib_scribe_platform_name,
                clib_scribe_platform_conf,
                clib_scribe_parser_params,
                parser_agent_params,
            )

        clib_scribe = ClipScribeEngine(
            mode=clib_scribe_mode,
            logger=logger,
            video_name=video_name,
            video_path=video_path,
            video_type=video_type,
            extractor=info_extractor,
            parser=info_parser,
            reader_db=reader_db,
            writer_db=writer_db,
        )

        return clib_scribe

    except Exception as e:
        logger.error(e)
        raise e
