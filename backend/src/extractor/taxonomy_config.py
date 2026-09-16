import logging
from typing import Literal

from agents import Agent, Runner
from pydantic import BaseModel, Field

logger = logging.getLogger("clip_scribe")


class TaxonomyProfile(BaseModel):

    """
    Defines the 'seed' strategy for a specific video type.
    """

    video_type: str = Field(
        description="The key used to identify this profile (e.g. 'car_ad')"
    )

    focus_categories: list[str] = Field(
        description="The specific buckets of objects to generate",
        default_factory=lambda: ["Physical Objects", "Background Context"],
    )

    guidance_ratio: str = Field(
        description="Instruction on how to balance the categories",
        default="Ensure a balanced mix of all categories.",
    )

    example_items: list[str] = Field(
        description="Few-shot examples to guide the LLM", default_factory=list
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

        self.__type_to_profile["car commercial"] = car_commercial_profile

    @staticmethod
    def _normalize_video_type(video_type: str) -> str:
        return " ".join(video_type.strip().lower().split())

    def has_profile(self, video_type: str) -> bool:
        """Return whether an exact normalized profile exists in this pile."""
        return self._normalize_video_type(video_type) in self.__type_to_profile

    def resolve_or_create(
        self,
        video_type: str | None,
        *,
        scene_context: str,
        dino_prompt: str = "",
        user_hints: list[str] | None = None,
    ) -> str:
        """Resolve a known profile or generate one for this run's pile."""
        requested_type = (
            self._normalize_video_type(video_type) if video_type is not None else None
        )
        if requested_type == "":
            requested_type = None

        if requested_type is not None and requested_type in self.__type_to_profile:
            return requested_type

        agent = Agent(
            name="DynamicTaxonomyProfileGenerator",
            instructions=(
                "Create a taxonomy seed profile for computer-vision object generation. "
                "Return a concise, specific video_type; concrete visual focus categories; "
                "percentage-based guidance covering the important categories; and generic "
                "visual object examples. When inferring a missing type, never use 'general' "
                "or 'general profile'. If a video type is supplied, build the profile for "
                "that exact type."
            ),
            model=self.model,
            output_type=TaxonomyProfile,
        )
        context_parts = []
        if requested_type is None:
            context_parts.append(
                "No video type was supplied. Infer a concise video type from the context."
            )
        else:
            context_parts.append(f"Requested video type: {requested_type}")
        context_parts.append(f"Scene description: {scene_context}")
        if dino_prompt:
            context_parts.append(f"Visible object prompt: {dino_prompt}")
        if user_hints:
            context_parts.append(f"User hints: {', '.join(user_hints)}")

        result = Runner.run_sync(agent, "\n".join(context_parts))
        generated_profile = result.final_output
        if generated_profile is None:
            raise RuntimeError(
                "Dynamic taxonomy profile generation returned no profile"
            )

        resolved_type = requested_type or self._normalize_video_type(
            generated_profile.video_type
        )
        if not resolved_type:
            raise ValueError(
                "Dynamic taxonomy profile generation returned no video type"
            )
        if requested_type is None and resolved_type in {"general", "general profile"}:
            raise ValueError(
                "Dynamic taxonomy profile generation returned a non-specific video type"
            )

        profile = generated_profile.model_copy(update={"video_type": resolved_type})
        self.__type_to_profile[resolved_type] = profile
        logger.info(
            "ClipScribe Extractor: Created run-local taxonomy profile for '%s'",
            resolved_type,
        )
        return resolved_type

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
                "video type from the allowed list. Always return one of the allowed types."
            ),
            model=self.model,
            output_type=MatchResult,
        )

        prompt = (
            f"Video description: '{video_type}'\n" f"Allowed types: {', '.join(keys)}"
        )

        result = Runner.run_sync(agent, prompt)
        return result.final_output.closest_type

    def get_video_profile(self, video_type: str) -> TaxonomyProfile:
        """
        video_type, if specified, is a natural language  query. get the closest available type in Taxonomy library

        :param video_type:
        :return:
        """
        normalized = self._normalize_video_type(video_type)
        try:
            profile = self.__type_to_profile[normalized]
            logger.info(
                f"ClipScribe Extractor: fetched video profile for video type: {normalized};"
            )
            return profile
        except KeyError as exc:
            raise KeyError(
                f"Profile {normalized!r} was not resolved or created"
            ) from exc
