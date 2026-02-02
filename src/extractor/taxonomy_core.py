from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from agents import Agent, Runner
import re
from .taxonomy_config import ProfilesPile

load_dotenv(find_dotenv())


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
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

        self.target_embeddings = None
        self.active_target_names = []

    def set_active_targets(self, targets: List[LeafItem]):

        self.logger.info(f"Encoding {len(targets)} active targets...")
        self.targets = targets
        self.active_target_names = [t.anchor for t in targets]

        if not self.active_target_names:
            self.target_embeddings = None
            return

        self.target_embeddings = self.embed_model.encode(
            self.active_target_names,
            convert_to_tensor=True,
            show_progress_bar=False
        )

    def resolve(self, raw_label: str, threshold: float = 0.35) -> Optional[str]:

        clean_label = re.sub(r'[^a-zA-Z0-9 ]', '', raw_label).strip().lower()

        if len(clean_label) < 2 or self.target_embeddings is None:
            return None

        query_embedding = self.embed_model.encode(clean_label, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.target_embeddings)[0]

        best_score_tensor = torch.max(cos_scores, dim=0)
        best_score = best_score_tensor.values.item()
        best_idx = best_score_tensor.indices.item()

        best_candidate = self.active_target_names[best_idx]

        self.logger.info(f"Resolver: '{clean_label}' -> Top Cand: {best_candidate} (Sim: {best_score:.3f})")

        if best_score < threshold:
            for cand in self.active_target_names:
                if f" {cand} " in f" {clean_label} " or f" {clean_label} " in f" {cand} ":
                    self.logger.info(f"MATCH (String Fallback): '{clean_label}' mapped to '{cand}'")
                    return cand

        if best_score >= threshold:
            self.logger.info(f"MATCH: '{clean_label}' mapped to '{best_candidate}'")
            return best_candidate

        return None


class TaxonomyGenerator:
    def __init__(self, num_objects: int, profiles: ProfilesPile, logger, model: str = "gpt-4o-mini"):
        self.model = model
        self.profiles = profiles
        self.logger = logger
        self.num_objects = num_objects

        self.leaf_agent = Agent(
            name="StructuredLeafGenerator",
            instructions=(
                "You are a computer vision expert. "
                f"Generate EXACTLY {self.num_objects} unique visual objects likely to be in the scene. "
                "CRITICAL: Output generic, visual classes (e.g., 'car', 'suv', 'tree', 'cellphone', 'building'), "
                "NOT specific instances and NOT abstract concepts."
                "IGNORE "

            ),
            model=self.model,
            output_type=StructuredLeafList,
        )

    def generate_targets(self, video_type: str, scene_context: str = "") -> List[LeafItem]:
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