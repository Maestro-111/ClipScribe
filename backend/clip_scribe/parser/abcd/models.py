"""Base Pydantic models for parser module."""

from pydantic import BaseModel


class BaseFeatureResult(BaseModel):
    """Result of evaluating a single feature."""

    video_name: str
    platform: str
    evaluation: bool
    llm_prompt: str | None = None
    llm_explanation: str | None = None


class BaseAgentEvaluation(BaseModel):
    """Structured output the LLM returns for agentic evaluation."""

    evaluation: bool
    explanation: str
