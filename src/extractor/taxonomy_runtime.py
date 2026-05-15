from typing import Iterable


def normalize_hints(hints: Iterable[str] | None) -> list[str]:
    if not hints:
        return []

    normalized_hints: list[str] = []
    seen: set[str] = set()

    for hint in hints:
        if not isinstance(hint, str):
            continue

        normalized_hint = hint.strip().lower()
        if not normalized_hint or normalized_hint in seen:
            continue

        seen.add(normalized_hint)
        normalized_hints.append(normalized_hint)

    return normalized_hints


def merge_hint_sources(*hint_sources: Iterable[str] | None) -> list[str]:
    merged_hints: list[str] = []
    for hint_source in hint_sources:
        merged_hints.extend(normalize_hints(hint_source))

    return normalize_hints(merged_hints)


def build_taxonomy_generation_input(
    video_type: str | None,
    profile_prompt: str,
    scene_context: str = "",
    dino_prompt: str = "",
    user_hints: Iterable[str] | None = None,
) -> str:
    normalized_hints = normalize_hints(user_hints)

    lines = [
        f"Video Type: {video_type or 'general'}",
        profile_prompt,
    ]

    if scene_context:
        lines.append(f"Scene Description: {scene_context}")

    if dino_prompt:
        lines.append(f"GroundingDINO Prompt: {dino_prompt}")

    if normalized_hints:
        lines.append(f"User Hints: {', '.join(normalized_hints)}")

    return "\n".join(lines) + "\n"
