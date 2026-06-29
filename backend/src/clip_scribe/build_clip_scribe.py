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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = Path(__file__).resolve().parent


class ClipScribeBuilder:
    @property
    def ALLOWED_MODELS(self):
        return (
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
        )

    @property
    def DEFAULT_HINT_GENERATION_MODEL(self):
        return "gpt-5.4-nano"

    @property
    def DEFAULT_TARGET_GENERATION_MODEL(self):
        return "gpt-5.4-mini"

    @property
    def DEFAULT_SCENE_DETECTION_MODEL(self):
        return "gpt-5.4-mini"

    @property
    def DEFAULT_PARSER_MODEL(self):
        return "gpt-5.4-mini"

    def __init__(self):
        with open(LOCAL_DIR / "configs" / "clip_scribe.yaml") as f:
            _cfg = yaml.safe_load(f)

        clib_scribe_paths = {
            name: PROJECT_ROOT / rel_path for name, rel_path in _cfg["paths"].items()
        }

        self.models_weights_dir = clib_scribe_paths["checkpoints"]

        self.dino_params = _cfg["dino"]
        self.clip_scribe_params = _cfg["clip_scribe"]
        self.clib_scribe_extractor_params = self.clip_scribe_params["extractor"]
        self.face_detection_params = _cfg["face_detection"]
        self.taxonomy_params = _cfg["taxonomy"]
        self.audio_params = _cfg["audio"]
        self.sam2_params = _cfg["sam2"]
        self.scene_analysis_params = _cfg["scene_analysis"]

        self.clib_scribe_parser_params = self.clip_scribe_params.get("parser", {})
        self.parser_agent_params = self.clib_scribe_parser_params.get("agent", {})

        self.db_params = _cfg.get("database", {})
        self.db_backend = self.db_params.get("backend", "sqlite")

        self._assemble_db()
        self._assemble_heavy_extractor_utils()

    def build_parser(
        self,
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

        parser_detection_model = self.resolve_model(
            "parser.agent.llm",
            parser_agent_params.get("llm", self.DEFAULT_PARSER_MODEL),
        )
        parser_agent_params["llm"] = parser_detection_model

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
        self,
        video_name: str,
        user_hints: list[str] | None,
        generate_hint_from_name: bool,
        clib_scribe_device: str,
        clib_scribe_extractor_params: dict,
        dino_params: dict,
        scene_analysis_params: dict,
        face_detection_params: dict,
        taxonomy_params: dict,
        audio_params: dict,
    ):
        hint_generation_model = self.resolve_model(
            "taxonomy.hint_generation.model",
            taxonomy_params["hint_generation"].get(
                "model", self.DEFAULT_HINT_GENERATION_MODEL
            ),
        )
        scene_detection_model = self.resolve_model(
            "scene_analysis.model",
            scene_analysis_params.get("model", self.DEFAULT_SCENE_DETECTION_MODEL),
        )

        audio_confidence: float = audio_params.get("audio_confidence", 0.4)

        dino_text_conf: float = dino_params.get("dino_text_conf", 0.4)
        dino_box_conf: float = dino_params.get("dino_box_conf", 0.4)

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
        detection_interval: int = clib_scribe_extractor_params.get(
            "detection_interval", 10
        )

        reid_model_frame_check_freq: int = clib_scribe_extractor_params.get(
            "reid_model_frame_check_freq", 20
        )

        logger.info(f"word_similarity_threshold: {word_similarity_threshold}")

        logger.info(f"dino_text_conf: {dino_text_conf}")
        logger.info(f"dino_box_conf: {dino_box_conf}")

        combined_hints = user_hints
        if generate_hint_from_name:
            combined_hints = generate_hints_from_video_name(
                video_name, logger, model=hint_generation_model, user_hints=user_hints
            )

        # Scene analysis configuration
        min_samples = scene_analysis_params.get("min_samples", 1)
        max_samples = scene_analysis_params.get("max_samples", 12)
        sampling_rate = scene_analysis_params.get("sampling_rate", 2.0)
        max_frame_dim = scene_analysis_params.get("max_frame_dim", 512)
        image_detail = scene_analysis_params.get("image_detail", "low")

        scene_describer = GPTSceneDescriber(
            logger,
            model=scene_detection_model,
            max_frame_dim=max_frame_dim,
            image_detail=image_detail,
        )

        dino_reid_device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

        info_extractor = VideoInformationExtractor(
            self.sam2,
            self.dino,
            scene_describer,
            self.ocr,
            self.taxonomy_resolver,
            self.taxonomy_generator,
            combined_hints,
            self.reid_model,
            self.audio_model,
            self.embedding_transform,
            self.face_detection,
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

    def _assemble_db(self) -> None:
        try:
            if self.db_backend == "sqlite":
                db_url = os.environ.get("SQLITE_URL", "sqlite:///data/clip_scribe.db")
                if db_url.startswith("sqlite:///") and not db_url.startswith(
                    "sqlite:////"
                ):
                    relative_path = db_url[len("sqlite:///") :]
                    db_url = f"sqlite:///{PROJECT_ROOT / relative_path}"
            else:
                db_url = os.environ["POSTGRESQL_URL"]

            db_engine = create_db_engine(
                database_url=db_url,
                pool_size=self.db_params.get("pool_size", 5),
                max_overflow=self.db_params.get("max_overflow", 10),
                logger=logger,
            )

            writer_db = ClipScribeWriterDB(engine=db_engine, logger=logger)
            reader_db = ClipScribeReaderDB(engine=db_engine, logger=logger)

            self.writer_db = writer_db
            self.reader_db = reader_db

        except Exception as e:
            logger.error(e)
            raise e

    def _assemble_heavy_extractor_utils(self) -> None:
        try:
            dino = DinoDetector(
                logger,
                dino_type=self.dino_params.get("dino_size", "tiny"),
                weights_dir=self.models_weights_dir,
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
                self.sam2_params.get("sam2_size", "tiny"),
                "src.sam2.configs",
                self.models_weights_dir,
                logger,
                sam2_device.type,
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

            profiles = ProfilesPile()
            taxonomy_objects_num: int = self.taxonomy_params.get(
                "taxonomy_objects_num", 100
            )

            target_generation_model = self.resolve_model(
                "taxonomy.target_generation.model",
                self.taxonomy_params.get("target_generation").get(
                    "model", self.DEFAULT_TARGET_GENERATION_MODEL
                ),
            )

            taxonomy_resolver = TaxonomyResolver(logger)
            taxonomy_generator = TaxonomyGenerator(
                taxonomy_objects_num,
                profiles,
                logger,
                model=target_generation_model,
            )

            self.dino = dino
            self.sam2 = sam2
            self.ocr = ocr
            self.reid_model = reid_model
            self.audio_model = audio_model
            self.embedding_transform = embedding_transform
            self.face_detection = face_detection
            self.taxonomy_resolver = taxonomy_resolver
            self.taxonomy_generator = taxonomy_generator

        except Exception as e:
            logger.error(e)
            raise e

    def build_clip_scribe(
        self,
        video_name: str,
        video_path: str,
        video_type: str | None,
        clib_scribe_mode: str,
        clib_scribe_device: str,
        clib_scribe_platform_name: str,
        clib_scribe_platform_conf: BasePlatformConf,
        user_hints: list[str] | None = None,
        generate_hint_from_name: bool = False,
    ) -> ClipScribeEngine:
        try:
            info_extractor = None
            info_parser = None

            if clib_scribe_mode in ["full", "extract"]:
                info_extractor = self.build_extractor(
                    video_name,
                    user_hints,
                    generate_hint_from_name,
                    clib_scribe_device,
                    self.clib_scribe_extractor_params,
                    self.dino_params,
                    self.scene_analysis_params,
                    self.face_detection_params,
                    self.taxonomy_params,
                    self.audio_params,
                )

            if clib_scribe_mode in ["full", "parse"]:
                info_parser = self.build_parser(
                    clib_scribe_platform_name,
                    clib_scribe_platform_conf,
                    self.clib_scribe_parser_params,
                    self.parser_agent_params,
                )

            clib_scribe = ClipScribeEngine(
                mode=clib_scribe_mode,
                logger=logger,
                video_name=video_name,
                video_path=video_path,
                video_type=video_type,
                extractor=info_extractor,
                parser=info_parser,
                reader_db=self.reader_db,
                writer_db=self.writer_db,
            )

            return clib_scribe

        except Exception as e:
            logger.error(e)
            raise e

    def resolve_model(self, field_name: str, model: str) -> str:
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"Invalid model for '{field_name}': {model!r}. "
                f"Allowed values: {', '.join(self.ALLOWED_MODELS)}"
            )

        normalized_model = model.strip()
        if normalized_model not in self.ALLOWED_MODELS:
            raise ValueError(
                f"Invalid model for '{field_name}': {normalized_model!r}. "
                f"Allowed values: {', '.join(self.ALLOWED_MODELS)}"
            )
        logger.info(f"field_name -> {field_name}: model: -> {normalized_model}")
        return normalized_model
