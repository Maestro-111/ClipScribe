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
import logging
import torch
import whisper

from src.ocr.paddle_wrapper import OCRSystem

from src.sam2.sam.build_sam import build_sam2_video_predictor
from src.utils.clip_scribe_logging import configure_logging
from src.db import (
    ClipScribeWriterDB,
    ClipScribeReaderDB,
    create_db_engine,
    resolve_database_url,
)
from src.utils.cancel import CancellationToken, NullCancellationToken
from src.utils.progress import NullProgressReporter, ProgressReporter
from src.utils.artifacts import NullArtifactUploader, SimulatedGCSArtifactUploader

from .engine import ClipScribeEngine
from .platform_configs import BasePlatformConf

from pathlib import Path
import yaml  # type: ignore

from dotenv import load_dotenv, find_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = Path(__file__).resolve().parent

logger = logging.getLogger("clip_scribe")

load_dotenv(find_dotenv(filename=".env"))  # local .env


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

    @property
    def DEFAULT_VIDEO_TYPE_MATCHER_MODEL(self):
        return "gpt-5.4-mini"

    def __init__(self, device: str | None = None):
        configure_logging()

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

        # Artifact handling: cap on per-frame viz PNGs and the (simulated for
        # now) remote-upload toggle. See docs/web-app-plan.md §8.
        self.artifacts_params = _cfg.get("artifacts", {})

        if device is None:
            logger.info("Using device specified in config")
            device = self.clip_scribe_params.get("device", None)
            if device is None:
                logger.warning("No device specified in config, using CPU")
                device = "cpu"

        # Verify the requested accelerator is actually present on this host;
        # otherwise roll back to CPU so a config/CLI value of "mps"/"cuda" can
        # never hard-crash at model load on a machine that lacks it.
        if device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but unavailable; falling back to CPU")
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU")
            device = "cpu"

        logger.info(f"Clip scribe device: {device}")
        self.device = device

        self._assemble_db()
        self._assemble_heavy_extractor_utils()

    def build_parser(
        self,
        clib_scribe_platform_name: str,
        clib_scribe_platform_conf: BasePlatformConf,
        clib_scribe_parser_params: dict,
        parser_agent_params: dict,
        progress_reporter: ProgressReporter | None = None,
        cancel_token: CancellationToken | None = None,
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
            max_parallel_agents=parser_max_parallel,
            recursion_limit=recursion_limit,
            progress_reporter=progress_reporter,
            cancel_token=cancel_token,
        )

        return info_parser

    def build_extractor(
        self,
        video_name: str,
        user_hints: list[str] | None,
        generate_hint_from_name: bool,
        clib_scribe_extractor_params: dict,
        dino_params: dict,
        scene_analysis_params: dict,
        face_detection_params: dict,
        taxonomy_params: dict,
        audio_params: dict,
        progress_reporter: ProgressReporter | None = None,
        cancel_token: CancellationToken | None = None,
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

        reid_similarity_difference: float = clib_scribe_extractor_params.get(
            "reid_similarity_difference", 0.8
        )

        logger.info(f"word_similarity_threshold: {word_similarity_threshold}")

        logger.info(f"dino_text_conf: {dino_text_conf}")
        logger.info(f"dino_box_conf: {dino_box_conf}")

        combined_hints = user_hints
        if generate_hint_from_name:
            combined_hints = generate_hints_from_video_name(
                video_name, model=hint_generation_model, user_hints=user_hints
            )

        # Scene analysis configuration
        min_samples = scene_analysis_params.get("min_samples", 1)
        max_samples = scene_analysis_params.get("max_samples", 12)
        sampling_rate = scene_analysis_params.get("sampling_rate", 2.0)
        max_frame_dim = scene_analysis_params.get("max_frame_dim", 512)
        image_detail = scene_analysis_params.get("image_detail", "low")

        scene_describer = GPTSceneDescriber(
            model=scene_detection_model,
            max_frame_dim=max_frame_dim,
            image_detail=image_detail,
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
            self.device,
            word_similarity_threshold,
            dino_text_conf,
            dino_box_conf,
            torch_face_cong,
            audio_confidence,
            label_match_merge_threshold,
            label_no_match_merge_threshold,
            detection_interval,
            reid_model_frame_check_freq,
            reid_similarity_difference,
            min_samples,
            max_samples,
            sampling_rate,
            progress_reporter=progress_reporter,
            cancel_token=cancel_token,
            max_artifact_files=self.artifacts_params.get("max_artifact_files"),
        )

        return info_extractor

    def _assemble_db(self) -> None:
        try:
            db_url = resolve_database_url()

            db_engine = create_db_engine(
                database_url=db_url,
                pool_size=self.db_params.get("pool_size", 5),
                max_overflow=self.db_params.get("max_overflow", 10),
            )

            writer_db = ClipScribeWriterDB(engine=db_engine)
            reader_db = ClipScribeReaderDB(engine=db_engine)

            self.writer_db = writer_db
            self.reader_db = reader_db

        except Exception as e:
            logger.error(e)
            raise e

    def _assemble_heavy_extractor_utils(self) -> None:
        try:
            dino = DinoDetector(
                dino_type=self.dino_params.get("dino_size", "tiny"),
                weights_dir=self.models_weights_dir,
                device=self.device,
            )

            ocr = OCRSystem()

            sam2 = build_sam2_video_predictor(
                self.sam2_params.get("size", "tiny"),
                "src.sam2.configs",
                self.models_weights_dir,
                logger,
                self.device,
            )

            logger.info(f"Loading DINOv2 (ViT-S/14) on {self.device}...")

            reid_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(
                self.device
            )

            reid_model.eval()

            logger.info("loading whisper to cpu (forces action)")

            # Whisper ignores the TORCH_HOME/HF_HOME-style env vars other
            # downloaders honor, so point its weights under checkpoints/ with an
            # explicit download_root. Keeps every model cache in one directory
            # (one Docker volume in dev, one baked image layer in prod).
            #
            # Force CPU (like MTCNN above): Whisper defaults to fp16 on non-CPU
            # devices, and fp16 in the decoder overflows to NaN logits on MPS,
            # crashing Categorical() sampling. CPU runs fp32 and is robust; the
            # "base" model is fast enough for typical clips.
            audio_model = whisper.load_model(
                "base",
                device="cpu",
                download_root=str(self.models_weights_dir / "whisper"),
            )

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

            extractor_video_matcher_model = self.resolve_model(
                "extractor.video_matcher",
                self.clib_scribe_extractor_params.get("video_matcher", {}).get(
                    "llm", self.DEFAULT_VIDEO_TYPE_MATCHER_MODEL
                ),
            )

            profiles = ProfilesPile(model=extractor_video_matcher_model)

            taxonomy_objects_num: int = self.taxonomy_params.get(
                "taxonomy_objects_num", 100
            )

            target_generation_model = self.resolve_model(
                "taxonomy.target_generation.model",
                self.taxonomy_params.get("target_generation").get(
                    "model", self.DEFAULT_TARGET_GENERATION_MODEL
                ),
            )

            taxonomy_resolver = TaxonomyResolver()
            taxonomy_generator = TaxonomyGenerator(
                taxonomy_objects_num,
                profiles,
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
        clib_scribe_platform_name: str,
        clib_scribe_platform_conf: BasePlatformConf,
        user_hints: list[str] | None = None,
        generate_hint_from_name: bool = False,
        progress_reporter: ProgressReporter | None = None,
        cancel_token: CancellationToken | None = None,
    ) -> ClipScribeEngine:
        """Build a per-job engine using the builder's long-lived dependencies.

        Heavy models and DB handles are initialized once in ``__init__``. This
        method creates fresh extractor/parser wrappers for the requested job
        and wires the same progress reporter into the engine and active stages.
        Passing ``None`` uses ``NullProgressReporter`` for CLI/tests.
        """
        try:
            # One reporter per job, shared by engine + extractor + parser. The
            # CLI leaves this None (Null = no-op); web execution paths pass a
            # RedisProgressReporter bound to the job id.
            reporter = progress_reporter or NullProgressReporter()
            # Same shape as the reporter: one token per job, shared by engine +
            # extractor + parser. CLI/tests leave this None (Null = never
            # canceled); web paths pass a RedisCancellationToken for the job id.
            canceller = cancel_token or NullCancellationToken()

            info_extractor = None
            info_parser = None

            if clib_scribe_mode in ["full", "extract"]:
                info_extractor = self.build_extractor(
                    video_name,
                    user_hints,
                    generate_hint_from_name,
                    self.clib_scribe_extractor_params,
                    self.dino_params,
                    self.scene_analysis_params,
                    self.face_detection_params,
                    self.taxonomy_params,
                    self.audio_params,
                    progress_reporter=reporter,
                    cancel_token=canceller,
                )

            if clib_scribe_mode in ["full", "parse"]:
                info_parser = self.build_parser(
                    clib_scribe_platform_name,
                    clib_scribe_platform_conf,
                    self.clib_scribe_parser_params,
                    self.parser_agent_params,
                    progress_reporter=reporter,
                    cancel_token=canceller,
                )

            # Same shape as the reporter default above: the builder resolves the
            # concrete dependency (from config) so the engine takes it as given.
            artifact_uploader = (
                SimulatedGCSArtifactUploader()
                if self.artifacts_params.get("remote_artifact_write", False)
                else NullArtifactUploader()
            )

            clib_scribe = ClipScribeEngine(
                mode=clib_scribe_mode,
                video_name=video_name,
                video_path=video_path,
                video_type=video_type,
                extractor=info_extractor,
                parser=info_parser,
                reader_db=self.reader_db,
                writer_db=self.writer_db,
                progress_reporter=reporter,
                cancel_token=canceller,
                artifact_uploader=artifact_uploader,
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
