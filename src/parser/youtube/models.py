"""YouTube-specific Pydantic models for parser module."""

from src.parser.models import BaseFeatureResult, BaseAgentEvaluation


class YouTubeFeatureResult(BaseFeatureResult):
    """Result of evaluating a single YouTube ABCD feature."""

    feature_category: str  # "Attract" / "Brand" / "Connect" / "Direct"
    feature_name: str
    feature_criteria: str


class YouTubeAgentEvaluation(BaseAgentEvaluation):
    """Structured output the LLM returns for YouTube agentic evaluation."""

    pass
