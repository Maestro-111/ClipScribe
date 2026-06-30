from typing import List, Optional
import logging

from pydantic import BaseModel, Field
import torch
from sentence_transformers import SentenceTransformer, util
from agents import Agent, Runner
import re
from .taxonomy_config import ProfilesPile

logger = logging.getLogger("clip_scribe")


def generate_hints_from_video_name(
    video_name: str, model: str, user_hints: list[str] | None = None
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

    generated_hints_norm = set([item.anchor.lower() for item in items])
    user_hints_norm = set()
    if user_hints:
        user_hints_norm = set([hint.lower() for hint in user_hints])

    combined_hints = list(generated_hints_norm | user_hints_norm)

    logger.info(
        f"Generated {len(combined_hints)} hints from video name: {combined_hints}"
    )
    return combined_hints


class LeafItem(BaseModel):
    """A structured object representing a canonical category."""

    anchor: str = Field(description="The primary, canonical name (e.g., 'car')")


class StructuredLeafList(BaseModel):
    """The collection of generated objects."""

    items: List[LeafItem] = Field(description="List of structured leaf objects")


class TaxonomyResolver:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.targets: List[LeafItem] = []

        logger.info("Loading SBERT Bi-Encoder...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

        self.target_embeddings: Optional[torch.Tensor] = None
        self.active_target_names: list[str] = []

    def set_active_targets(self, targets: List[LeafItem]):
        logger.info(f"Encoding {len(targets)} active targets...")
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
        best_idx = int(best_score_tensor.indices.item())

        best_candidate = self.active_target_names[best_idx]

        logger.info(
            f"Resolver: '{clean_label}' -> Top Cand: {best_candidate} (Sim: {best_score:.3f})"
        )

        if best_score < threshold:
            for cand in self.active_target_names:
                if (
                    f" {cand} " in f" {clean_label} "
                    or f" {clean_label} " in f" {cand} "
                ):
                    logger.info(
                        f"MATCH (String Fallback): '{clean_label}' mapped to '{cand}'"
                    )
                    return cand

        if best_score >= threshold:
            logger.info(f"MATCH: '{clean_label}' mapped to '{best_candidate}'")
            return best_candidate

        return None


class TaxonomyGenerator:
    def __init__(
        self,
        num_objects: int,
        profiles: ProfilesPile,
        model: str,
    ):
        self.model = model
        self.profiles = profiles
        self.num_objects = num_objects

        self.leaf_agent = Agent(
            name="StructuredLeafGenerator",
            instructions=(
                "You are a computer vision expert. "
                f"Generate between 5 and {self.num_objects} unique visual objects likely to be in the scene. "
                "Include MORE objects for complex scenes, FEWER for simple scenes. "
                "CRITICAL: Output generic, visual classes (e.g., 'car', 'suv', 'tree', 'cellphone', 'building', 'person'), "
                "NOT specific instances and NOT abstract concepts."
            ),
            model=self.model,
            output_type=StructuredLeafList,
        )

    def generate_targets(
        self,
        video_type: str | None,
        scene_context: str = "",
        dino_prompt: str = "",
        user_hints: list[str] | None = None,
    ) -> List[LeafItem]:
        profile = self.profiles.get_video_profile(video_type)

        if not profile:
            return []

        user_input = self.build_taxonomy_generation_input(
            video_type=video_type,
            profile_prompt=profile.get_prompt_instruction(),
            scene_context=scene_context,
            dino_prompt=dino_prompt,
            user_hints=user_hints,
        )

        result = Runner.run_sync(self.leaf_agent, user_input)
        taxonomy = result.final_output.items if result and result.final_output else []

        return taxonomy

    @staticmethod
    def build_taxonomy_generation_input(
        video_type: str | None,
        profile_prompt: str,
        scene_context: str = "",
        dino_prompt: str = "",
        user_hints: list[str] | None = None,
    ) -> str:
        lines = [
            f"Video Type: {video_type or 'general'}",
            profile_prompt,
        ]

        if scene_context:
            lines.append(f"Scene Description: {scene_context}")

        if dino_prompt:
            lines.append(f"GroundingDINO Prompt: {dino_prompt}")

        if user_hints:
            lines.append(f"User Hints: {', '.join(user_hints)}")

        return "\n".join(lines) + "\n"
