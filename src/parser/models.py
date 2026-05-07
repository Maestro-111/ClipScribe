"""Shared Pydantic models for parser module."""

from pydantic import BaseModel


class FeatureResult(BaseModel):
    """Result of evaluating a single ABCD feature."""

    video_name: str
    platform: str  # "youtube"
    feature_category: str  # "Attract" / "Brand" / "Connect" / "Direct"
    feature_name: str  # human-readable name from config
    feature_criteria: str  # criteria text from config
    evaluation: bool
    llm_prompt: str | None = None  # null for baseline
    llm_explanation: str | None = None  # null for baseline


class AgentEvaluation(BaseModel):
    """Structured output the LLM returns for agentic evaluation."""

    evaluation: bool
    explanation: str
