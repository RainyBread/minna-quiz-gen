"""Built-in LLM providers; switching provider is configuration-only."""

from __future__ import annotations

from typing import Any

from core.llm import LLMClient
from core.registry import llm_providers


class OpenAICompatibleProvider:
    def __init__(self, settings: Any, offline: bool = False):
        self.name = str(getattr(settings, "llm_provider", "openai") or "openai")
        self._client = LLMClient(settings, offline=offline)

    @property
    def available(self) -> bool:
        return bool(self._client.available)

    def complete_text(self, system: str, user: str, **kwargs: Any) -> str:
        return self._client.complete_text(system, user, **kwargs)

    def complete_json(self, *args: Any, **kwargs: Any) -> Any:
        return self._client.complete_json(*args, **kwargs)

    def complete_vision(self, *args: Any, **kwargs: Any) -> Any:
        return self._client.complete_vision(*args, **kwargs)


for _name in ("openai", "stepfun", "anthropic", "auto"):
    llm_providers.register(_name, OpenAICompatibleProvider, replace=True)
