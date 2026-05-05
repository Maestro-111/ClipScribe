from typing import List
import base64
import io
import json
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from openai import OpenAI


class SceneDescription(BaseModel):
    """Pydantic output model for GPT scene analysis."""

    scene_description: str = Field(
        description="Rich narrative description of what's happening in the scene"
    )
    dino_prompt: str = Field(
        description="Period-separated list of visual objects in GroundingDINO format (e.g., 'car . tree . person .')"
    )


class GPTSceneDescriber:
    """
    Uses GPT-4o-mini vision to analyze sampled frames from a video shot
    and generate both a scene description and a DINO detection prompt.
    """

    def __init__(
        self,
        logger,
        model: str = "gpt-4o-mini",
        max_frame_dim: int = 512,
        image_detail: str = "low",
    ):
        """
        Args:
            logger: Logger instance
            model: GPT model to use for vision analysis
            max_frame_dim: Maximum dimension (width or height) for frame resizing
            image_detail: OpenAI image detail mode ("low" or "high"). "low" uses 85 tokens/image.
        """
        self.logger = logger
        self.model = model
        self.max_frame_dim = max_frame_dim
        self.image_detail = image_detail
        self.client = OpenAI()

    def _resize_frame(self, frame_rgb: np.ndarray) -> Image.Image:
        """Resize frame to max_frame_dim while maintaining aspect ratio."""
        h, w = frame_rgb.shape[:2]
        scale = min(self.max_frame_dim / w, self.max_frame_dim / h)

        if scale >= 1.0:
            # No resizing needed
            return Image.fromarray(frame_rgb)

        new_w = int(w * scale)
        new_h = int(h * scale)
        pil_img = Image.fromarray(frame_rgb)
        return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def _encode_image_to_base64(self, pil_img: Image.Image) -> str:
        """Convert PIL image to base64-encoded JPEG string."""
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def _sanitize_dino_prompt(self, raw_prompt: str) -> str:
        """
        Clean and deduplicate DINO prompt.
        - Remove 'text' keyword
        - Deduplicate targets while preserving order
        - Ensure proper formatting
        """
        # Parse period-separated targets
        targets = [t.strip().lower() for t in raw_prompt.split('.') if t.strip()]

        # Filter out 'text' and deduplicate while preserving order
        seen = set()
        unique_targets = []
        for target in targets:
            if target == 'text':
                self.logger.warning(f"Filtered out 'text' from DINO prompt")
                continue
            if target not in seen:
                seen.add(target)
                unique_targets.append(target)

        # Reconstruct with proper formatting
        return ' . '.join(unique_targets) + ' .' if unique_targets else 'object .'

    def describe_scene(self, frames_rgb: List[np.ndarray]) -> tuple[str, str]:
        """
        Analyze multiple frames from a video shot and generate scene description + DINO prompt.

        Args:
            frames_rgb: List of RGB frames (np.ndarray) from the shot

        Returns:
            Tuple of (dino_prompt, scene_description) matching the old API shape
        """
        if not frames_rgb:
            self.logger.warning("No frames provided to describe_scene, using defaults")
            return ("object .", "")

        # Prepare frames as base64-encoded images
        image_content = []
        for idx, frame_rgb in enumerate(frames_rgb):
            pil_img = self._resize_frame(frame_rgb)
            b64_str = self._encode_image_to_base64(pil_img)
            image_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_str}",
                        "detail": self.image_detail,
                    },
                }
            )

        self.logger.info(f"Sending {len(frames_rgb)} frames to GPT for scene analysis")

        # Build the message content with text + images
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a computer vision expert analyzing video frames. "
                    "Generate a JSON response with two fields:\n"
                    "1. scene_description: A rich, detailed description of what's happening in the scene. "
                    "Describe the setting, main subjects, actions, mood, and composition.\n"
                    "2. dino_prompt: A period-separated list of GENERIC visual object classes visible in the scene. "
                    "Format: 'object1 . object2 . object3 .' (always end with a period). "
                    "Use simple, generic class names (e.g., 'car', 'tree', 'person', 'building'). "
                    "Include objects depending on scene complexity. "
                    "DO NOT include abstract concepts or specific instances."
                    "CRITICAL: NEVER include 'text' as an object."
                    "You MAY include text-bearing objects like 'sign', 'license plate', 'billboard', 'banner', but NOT the word 'text' itself."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Here are representative frames from a video shot. Analyze them and provide a scene description and object detection prompt.",
                    }
                ]
                + image_content,
            },
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=500,
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)

            scene_desc = SceneDescription(**parsed)

            self.logger.info(
                f"GPT Scene Description: {scene_desc.scene_description[:100]}..."
            )
            self.logger.info(f"GPT DINO Prompt (raw): {scene_desc.dino_prompt}")

            # Sanitize the DINO prompt to remove duplicates and 'text' keyword
            sanitized_prompt = self._sanitize_dino_prompt(scene_desc.dino_prompt)
            self.logger.info(f"GPT DINO Prompt (sanitized): {sanitized_prompt}")

            return (sanitized_prompt, scene_desc.scene_description)

        except Exception as e:
            self.logger.error(f"GPT scene analysis failed: {e}, using fallback")
            return ("object .", "")
