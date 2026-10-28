from dataclasses import dataclass


@dataclass
class _PromptLimitsParametrizer:
    params: dict[str, int]

    def parametrize(self, prompt: str, shared_rules: str) -> str:
        return prompt.format_map(self.params) + shared_rules
