from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
import torch
from sentence_transformers import SentenceTransformer, util
from agents import Agent, Runner
import re
from .taxonomy_config import ProfilesPile

load_dotenv(find_dotenv())


def generate_hints_from_video_name(
    video_name: str, logger, model: str = "gpt-4o-mini"
) -> list[str]:
    """Infer 10-20 object hints from a descriptive video filename.
    Returns [] if the name looks generic / auto-generated."""

    logger.info(f"Attempting to generate hints from video name: '{video_name}'")

    hint_agent = Agent(
        name="VideoNameHintGenerator",
        instructions=(
            "You are given a video filename. "
            "If the filename is descriptive (contains real words, brand names, or topic clues), "
            "generate 10 to 20 generic visual object classes that are likely to appear in that video. "
            "Output ONLY generic visual classes (e.g., 'car', 'tree', 'person', 'logo'), "
            "NOT specific instances or abstract concepts. "
            "If the filename is generic, auto-generated, or contains no meaningful clues "
            "(e.g., 'abcd-1234.mp4', 'video_001.mov', 'VID_20240101'), return an EMPTY list."
        ),
        model=model,
        output_type=StructuredLeafList,
    )

    result = Runner.run_sync(hint_agent, f"Video filename: {video_name}")
    items = result.final_output.items if result and result.final_output else []
    hints = [item.anchor for item in items]

    logger.info(f"Generated {len(hints)} hints from video name: {hints}")
    return hints


class LeafItem(BaseModel):
    """A structured object representing a canonical category."""

    anchor: str = Field(description="The primary, canonical name (e.g., 'car')")


class StructuredLeafList(BaseModel):
    """The collection of generated objects."""

    items: List[LeafItem] = Field(description="List of structured leaf objects")


class TaxonomyResolver:
    def __init__(self, logger, device: str = "cpu"):
        self.logger = logger
        self.device = device
        self.targets: List[LeafItem] = []

        self.logger.info("Loading SBERT Bi-Encoder...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

        self.target_embeddings = None
        self.active_target_names: list[str] = []

    def set_active_targets(self, targets: List[LeafItem]):
        self.logger.info(f"Encoding {len(targets)} active targets...")
        self.targets = targets
        self.active_target_names = [t.anchor for t in targets]

        if not self.active_target_names:
            self.target_embeddings = None
            return

        self.target_embeddings = self.embed_model.encode(
            self.active_target_names, convert_to_tensor=True, show_progress_bar=False
        )

    def resolve(self, raw_label: str, threshold: float = 0.35) -> Optional[str]:
        clean_label = re.sub(r"[^a-zA-Z0-9 ]", "", raw_label).strip().lower()

        if len(clean_label) < 2 or self.target_embeddings is None:
            return None

        query_embedding = self.embed_model.encode(clean_label, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.target_embeddings)[0]

        best_score_tensor = torch.max(cos_scores, dim=0)
        best_score = best_score_tensor.values.item()
        best_idx = best_score_tensor.indices.item()

        best_candidate = self.active_target_names[best_idx]

        self.logger.info(
            f"Resolver: '{clean_label}' -> Top Cand: {best_candidate} (Sim: {best_score:.3f})"
        )

        if best_score < threshold:
            for cand in self.active_target_names:
                if (
                    f" {cand} " in f" {clean_label} "
                    or f" {clean_label} " in f" {cand} "
                ):
                    self.logger.info(
                        f"MATCH (String Fallback): '{clean_label}' mapped to '{cand}'"
                    )
                    return cand

        if best_score >= threshold:
            self.logger.info(f"MATCH: '{clean_label}' mapped to '{best_candidate}'")
            return best_candidate

        return None


class TaxonomyGenerator:
    def __init__(
        self,
        num_objects: int,
        profiles: ProfilesPile,
        logger,
        model: str = "gpt-4o-mini",
        user_hints: list[str] | None = None,
    ):
        self.model = model
        self.profiles = profiles
        self.logger = logger
        self.num_objects = num_objects
        self.user_hints = user_hints or []

        hints_instruction = ""
        if self.user_hints:
            hints_str = ", ".join(self.user_hints)
            hints_instruction = (
                f"The user expects these objects may appear in the video: {hints_str}. "
                "Prioritize including these in your output when relevant to the scene. "
            )

        self.leaf_agent = Agent(
            name="StructuredLeafGenerator",
            instructions=(
                "You are a computer vision expert. "
                f"Generate between 5 and {self.num_objects} unique visual objects likely to be in the scene. "
                "Include MORE objects for complex scenes, FEWER for simple scenes. "
                f"{hints_instruction}"
                "CRITICAL: Output generic, visual classes (e.g., 'car', 'suv', 'tree', 'cellphone', 'building', 'person'), "
                "NOT specific instances and NOT abstract concepts."
            ),
            model=self.model,
            output_type=StructuredLeafList,
        )

    def generate_targets(
        self, video_type: str, scene_context: str = ""
    ) -> List[LeafItem]:
        self.logger.info(f"Generating targets for context: '{scene_context}'")
        profile = self.profiles.get_video_profile(video_type)

        if not profile:
            return []

        user_input = (
            f"Video Type: {video_type}\n"
            f"{profile.get_prompt_instruction()}\n"
            f"Scene Description: {scene_context}\n"
        )

        result = Runner.run_sync(self.leaf_agent, user_input)
        taxonomy = result.final_output.items if result and result.final_output else []
        return taxonomy
