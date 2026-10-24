"""System prompts for the exploratory report's LLM calls.

Shared rules: describe only what the provided data shows, stay neutral (this is
exploration, not an evaluation against criteria), and refer to items by the
indices given so the code can attach real timestamps.
"""

from clip_scribe.parser.explore.models import (
    TEXT_CATEGORY_DESCRIPTIONS,
    TextCategory,
)

_SHARED_RULES = """\
Rules:
- Use only the data provided. Never invent content, numbers, or timestamps.
- Be neutral and descriptive; do not judge quality or effectiveness.
- Be concise. Always write in English, whatever language the video uses; you
  may translate or paraphrase what is said or shown.
- When asked to reference items, use the exact index numbers given.
"""

TRANSCRIPT_PROMPT = """\
You summarize the speech in a video from its transcript.

The transcript is a numbered list of timed speech segments. Produce:
- summary: up to {transcript_summary_sentences_num} sentences on what is said.
- topics: up to {transcript_topics_num} short topic labels.
- key_claims: up to {transcript_key_claims_num} concrete claims, offers, or statements, each with the
  index of the segment it comes from.
"""

ON_SCREEN_TEXT_PROMPT = (
    """\
You summarize the on-screen text of a video.

The input is a numbered list of distinct text cards read by OCR, with when they
are visible. OCR may contain errors (missing spaces, dropped or swapped
letters). Produce:
- summary: up to {text_summary_sentences_num} sentences on what the on-screen text communicates.
- cards: for every card, its index, clean_text (the most plausible correct
  reading, in the text's original language and never translated; keep it as-is
  if unsure), and a category, one of:
"""
    + "\n".join(
        f"  - {category.value}: {TEXT_CATEGORY_DESCRIPTIONS[category]}"
        for category in TextCategory
    )
    + "\n"
)

VISUALS_PROMPT = """\
You summarize what is shown in a video.

The input has per-shot scene descriptions and, when available, tracked objects
with their total screen time and average screen coverage (0-1). Produce:
- summary: up to {visual_summary_sentences_num} sentences on the main subjects, settings, and visual style.
- subjects: for each tracked object label given (use the exact label), a
  prominence of high, medium, or low based on its screen time, coverage, and
  role in the scenes. Return an empty list if no objects are given.
"""

SYNTHESIS_PROMPT = """\
You write the overview of a video from its extracted data.

The input has run metrics, short summaries of the speech, on-screen text, and
visuals, and a numbered list of shots (each with its time range, scene
description, objects, on-screen text, and speech). Produce:
- summary: {overview_summary_sentences_num} sentences describing what the video is and how it unfolds.
- chapters: split the video into up to {overview_summary_chapters_num} consecutive chapters (1 if it is very
  short or a single scene). Each chapter covers a contiguous range of shots,
  given as first_shot and last_shot indices, with a short title and a
  one-sentence summary. Chapters must be in order and must not overlap.
"""
