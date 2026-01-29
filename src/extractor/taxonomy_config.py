from pydantic import BaseModel, Field
from typing import List


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

    # NEW: Dedicated field for brand terms
    brand_keywords: List[str] = Field(
        description="Specific brand terms, models, or text to force into the taxonomy",
        default=[],
    )

    def get_prompt_instruction(self) -> str:
        """Helper to format these settings into a prompt block"""
        cats = "\n".join([f"- {c}" for c in self.focus_categories])
        examples = ", ".join(self.example_items)
        brands = ", ".join(self.brand_keywords)

        return (
            f"Context: This is for a '{self.video_type}'.\n"
            f"Mandatory Categories:\n{cats}\n"
            f"Ratio Guidance: {self.guidance_ratio}\n"
            f"Mandatory Brand Terms (Must Include): {brands}\n"  # Force LLM to include these
            f"Examples: {examples}"
        )


class ProfilesPile:
    """
    data holder to extract appropriate profile by name
    """

    def __init__(self):
        self.__type_to_profile = {}
        self.__create_profiles()

    def __create_profiles(self):
        car_ad_profile = TaxonomyProfile(
            video_type="car ad",
            focus_categories=[
                "Main Subject (e.g. car, suv, vehicle, jeep, automobile)",
                "Subject Details (e.g. wheels, lights, grille, bumper)",
                "Brand Identity (e.g. logo, badge, license plate, text overlays)",
                "Environmental Context (e.g. road, trees, buildings, sky)",
                "Human Context (e.g. driver, pedestrian, passenger)",
            ],
            # Adjusted ratio to make room for brand identity
            guidance_ratio="Generate 30% main subject, 20% parts, 15% brand identity, 10% environment, and 25% human context.",
            # Explicitly force these terms into the search pool
            brand_keywords=[
                "Car Brand Logo",
                "Car Brand badge",
                "Car Brand emblem",
                "License Plate",
            ],
            example_items=["SUV", "alloy wheel", "pine tree", "driver", "car emblem"],
        )

        self.__type_to_profile["car ad"] = car_ad_profile

    def get_video_profile(self, video_type):
        if video_type not in self.__type_to_profile:
            return None  # Fixed explicit return None
        return self.__type_to_profile[video_type]
