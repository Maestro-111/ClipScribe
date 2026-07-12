from typing import List
from pydantic import BaseModel, Field
from typing import Literal
import logging

from agents import Agent, Runner

logger = logging.getLogger("clip_scribe")


class TaxonomyProfile(BaseModel):

    """
    Defines the 'seed' strategy for a specific video type.
    """

    video_type: str = Field(
        description="The key used to identify this profile (e.g. 'car_ad')"
    )

    focus_categories: List[str] = Field(
        description="The specific buckets of objects to generate",
        default=["Physical Objects", "Background Context"],
    )

    guidance_ratio: str = Field(
        description="Instruction on how to balance the categories",
        default="Ensure a balanced mix of all categories.",
    )

    example_items: List[str] = Field(
        description="Few-shot examples to guide the LLM", default=[]
    )

    def get_prompt_instruction(self) -> str:
        """Helper to format these settings into a prompt block"""
        cats = "\n".join([f"- {c}" for c in self.focus_categories])
        examples = ", ".join(self.example_items)

        return (
            f"Context: This is for a '{self.video_type}'.\n"
            f"Mandatory Categories:\n{cats}\n"
            f"Ratio Guidance: {self.guidance_ratio}\n"
            f"Examples: {examples}"
        )


class ProfilesPile:
    """
    data holder to extract appropriate profile by name
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.__type_to_profile: dict[str, TaxonomyProfile] = {}
        self.__create_profiles()

    def __create_profiles(self) -> None:
        car_commercial_profile = TaxonomyProfile(
            video_type="car commercial",
            focus_categories=[
                "Main Subject (e.g. car, suv, vehicle, jeep, automobile)",
                "Subject Details (e.g. wheels, lights, grille, bumper)",
                "Brand Identity (e.g. logo, badge, license plate)",
                "Environmental Context (e.g. road, trees, buildings, sky)",
                "Human Context (e.g. driver, pedestrian, passenger)",
            ],
            guidance_ratio="Generate 30% main subject, 20% parts, 15% brand identity, 10% environment, and 25% human context.",
            example_items=["SUV", "alloy wheel", "truck", "driver", "car emblem"],
        )

        general_profile = TaxonomyProfile(
            video_type="general",
            focus_categories=[
                "Main Subjects (e.g. person, animal, primary object)",
                "Physical Objects (e.g. furniture, tools, devices, vehicles)",
                "Environmental Context (e.g. buildings, nature, indoor/outdoor settings)",
                "Text and Signage (e.g. signs, labels, captions)",
                "Actions and Activities (e.g. movement, interactions)",
            ],
            guidance_ratio="Generate a balanced distribution with 30% main subjects, 25% physical objects, 20% environmental context, 15% actions, and 10% text/signage.",
            example_items=[
                "person",
                "table",
                "tree",
                "building",
                "walking",
                "car",
                "sign",
                "door",
            ],
        )

        self.__type_to_profile["car commercial"] = car_commercial_profile
        self.__type_to_profile["general profile"] = general_profile

    def query_type_to_profile(self, video_type: str) -> str:
        keys = list(self.__type_to_profile.keys())
        normalized = video_type.strip().lower()

        for k in keys:
            if normalized == k.lower():
                return k

        class MatchResult(BaseModel):
            closest_type: Literal[tuple(keys)] = Field(  # type: ignore
                description="Closest matching video type from the allowed list."
            )

        agent = Agent(
            name="VideoTypeMatcher",
            instructions=(
                "You classify video descriptions into one of a fixed set of categories. "
                "Given a natural-language description, pick the single closest matching "
                "video type from the allowed list. If nothing is a clear semantic match, "
                "default to 'general profile'."
            ),
            model=self.model,
            output_type=MatchResult,
        )

        prompt = (
            f"Video description: '{video_type}'\n" f"Allowed types: {', '.join(keys)}"
        )

        result = Runner.run_sync(agent, prompt)
        return result.final_output.closest_type

    def get_video_profile(self, video_type: str | None) -> TaxonomyProfile:
        """
        video_type, if specified, is a natural language  query. get the closest available type in Taxonomy library

        :param video_type:
        :return:
        """

        if video_type is None:
            logger.info("No video type specified; defaulting to 'general'")
            return self.__type_to_profile["general profile"]

        closest_type: str = self.query_type_to_profile(video_type)

        logger.info(
            f"closest video type: {closest_type} for specified video_type: {video_type}"
        )
        return self.__type_to_profile[closest_type]
