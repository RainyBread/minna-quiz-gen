"""Interfaces implemented by plugins.

Core code depends on these contracts; concrete question types, dictionaries,
source strategies, exporters and LLM providers live in ``plugins/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.items import Item, ItemDefect
from core.render import RenderedQuestion


class PluginError(RuntimeError):
    pass


@dataclass(frozen=True)
class SlotType:
    name: str
    left_boundary: str = ""
    right_boundary: str = ""


@dataclass
class PluginContext:
    settings: Any
    context: Any
    pools: Any = None
    rng: Any = None
    llm: Any = None
    locks: Any = None
    mdict: Any = None
    extension_pool: Any = None
    item_bank: Any = None
    batch_id: str = ""
    seed: int = 0
    exposed_points: set[str] = field(default_factory=set)

    def require(self, name: str) -> Any:
        value = getattr(self, name, None)
        if value is None:
            raise PluginError(f"plugin context is missing {name!r}")
        return value


@dataclass
class DictFact:
    lemma: str
    reading: str = ""
    pos: str = ""
    inflection: list[str] = field(default_factory=list)
    senses: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    level: int | None = None
    source_kind: str = "generic"
    attestation: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class QuestionTypePlugin(Protocol):
    name: str
    legacy_type: str
    is_local: bool
    slot_type: SlotType

    def build_local(self, spec: Any, ctx: PluginContext) -> Item | None:
        ...

    def build_prompt(self, spec: Any) -> dict[str, str]:
        ...

    def distractor_pool(self, spec: Any, ctx: PluginContext) -> list[str]:
        ...

    def clue_injector(self, spec: Any, ctx: PluginContext) -> str:
        ...

    def renderer(self, item: Item, *, seed: int, number: int) -> RenderedQuestion:
        ...

    def local_assertions(self, item: Item) -> list[ItemDefect]:
        ...


@runtime_checkable
class DictSource(Protocol):
    name: str
    available: bool

    def lookup(self, lemma: str) -> DictFact | None:
        ...


@runtime_checkable
class SourceStrategy(Protocol):
    name: str

    def eligible(self, spec: Any, ctx: PluginContext) -> bool:
        ...

    def produce(self, spec: Any, ctx: PluginContext) -> Item:
        ...


@runtime_checkable
class Exporter(Protocol):
    name: str

    def export(
        self,
        items: list[Item],
        out: Path,
        *,
        seed: int = 0,
        title: str = "JLPTCraft",
        **kwargs: Any,
    ) -> Any:
        ...


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    @property
    def available(self) -> bool:
        ...

    def complete_text(self, system: str, user: str, **kwargs: Any) -> str:
        ...
