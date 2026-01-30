from typing import List, Any
from agents import Agent, Runner
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from .taxonomy_config import ProfilesPile

load_dotenv(find_dotenv())


class TaxonomyResolver:
    def __init__(self, logger, device: str = "cpu"):
        """
        Initializes a semantic mapper using Zero-Shot Classification.
        """

        # Small, high-performance model for semantic mapping

        self.logger = logger
        self.classifier = pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli", device=device
        )

    def resolve(
        self, targets: list[str], raw_label: str, threshold: float = 0.5
    ) -> str | None | Any:
        """
        Maps a raw label to the most similar category in the taxonomy.
        """

        if not targets:
            return raw_label

        result = self.classifier(raw_label, targets, multi_label=True)

        top_label = result["labels"][0]
        top_score = result["scores"][0]

        self.logger.info(f"Top label: {top_label}; Top score: {top_score}")

        if top_score >= threshold:
            return top_label

        self.logger.info(f" Top score: {top_score} is < {threshold}; returning None")
        return None


class TaxonomyList(BaseModel):
    taxonomy: List[str] = Field(description="List of extracted taxonomies")


class LeafList(BaseModel):
    items: List[str] = Field(description="List of specific objects (leaves)")


class ParentPair(BaseModel):
    item: str = Field(description="The specific object (e.g., 'driver')")
    parent: str = Field(description="The generic parent category (e.g., 'person')")


class ParentMapping(BaseModel):
    mappings: List[ParentPair] = Field(description="List of item-to-parent mappings.")


class TaxonomyGenerator:
    def __init__(
        self, num_objects: int, profiles: ProfilesPile, logger, model: str = "gpt-4o"
    ):
        self.model = model
        self.profiles = profiles
        self.logger = logger

        self.leaf_agent = Agent(
            name="LeafGenerator",
            instructions=(
                "You are a computer vision expert. "
                f"Generate EXACTLY #{num_objects} unique, specific, visible objects based strictly on the user's provided profile. "
                "Focus on 'leaf' nodes (specific items)."
            ),
            model=self.model,
            output_type=LeafList,
        )

        self.parent_agent = Agent(
            name="ParentGenerator",
            instructions=(
                "You are a taxonomy logic engine. "
                "Given a list of words, provide the immediate 'semantic parent' for each word. "
                "The parent should be 1 level more general."
                "Example: 'driver' -> 'person', 'tire' -> 'car part', 'tree' -> 'nature'. "
                "Merge distinct items into shared parents where logical."
            ),
            model=self.model,
            output_type=ParentMapping,
        )

    def _get_leaves(self, video_type: str) -> List[str]:
        self.logger.info(
            f"--- Stage 1: Create Profile and Generating leaves for {video_type}---"
        )

        profile = self.profiles.get_video_profile(video_type)

        if not profile:
            self.logger.error(f"no profile found for video type {video_type}")
            return []

        user_input = (
            f"Generate objects for a {video_type}.\n"
            f"{profile.get_prompt_instruction()}"
        )

        result = Runner.run_sync(self.leaf_agent, user_input)

        if result and result.final_output:
            return [i.lower() for i in result.final_output.items]

        return []

    def _get_parents(self, items: List[str]) -> dict:
        self.logger.info(f"--- Stage 2: Generating parents for {len(items)} items ---")

        user_input = f"Find semantic parents for these items: {items}"

        result = Runner.run_sync(self.parent_agent, user_input)

        parent_map = {}

        if result and result.final_output:
            for pair in result.final_output.mappings:
                parent_map[pair.item.lower()] = pair.parent.lower()

        return parent_map

    def build_taxonomy(self, video_type: str, levels: int = 2) -> dict:

        taxonomy_layers: dict[int, list[str]] = {}
        current_items = self._get_leaves(video_type)

        if not current_items:
            return taxonomy_layers

        taxonomy_layers[0] = current_items

        for i in range(1, levels):
            parent_map = self._get_parents(current_items)
            next_level_items = list(set(parent_map.values()))

            taxonomy_layers[i] = next_level_items
            current_items = next_level_items

            if not current_items:
                break

        return taxonomy_layers
