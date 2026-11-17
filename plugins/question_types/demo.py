"""Demo plugin: a local antonym-choice question type.

This file is the template for adding a question type without touching
``core/``.  The only registration hook is the decorator below.
"""

from __future__ import annotations

from typing import Any

from core.contracts import PluginContext, PluginError, SlotType
from core.items import (
    Item,
    ItemDefect,
    ItemDistractor,
    ItemKey,
    ItemMeta,
    ItemSlot,
    refresh_hashes,
    stable_item_id,
    validate_item,
)
from core.models import sha256_text
from core.prompts import build_carrier_prompt
from core.registry import question_types
from core.render import RenderedQuestion, render_item


class DemoAntonymQuestion:
    name = "demo_antonym"
    legacy_type = "vocab_choice"
    is_local = True
    slot_type = SlotType("vocab")

    def build_local(self, spec: Any, ctx: PluginContext) -> Item | None:
        if ctx.locks is None:
            raise PluginError("demo_antonym requires a lock table")
        target = (spec.written or "").strip()
        antonym = ctx.locks.antonym_of(target)
        if not antonym and ctx.locks.antonyms:
            target, antonym = ctx.locks.antonyms[0]
        if not target or not antonym:
            raise PluginError(f"no antonym for {target!r}")
        pools = ctx.require("pools")
        rng = ctx.require("rng")
        distractors: list[str] = []
        seen = {target, antonym}
        for entry in pools.vocab_candidates(range(37, 51)):
            value = entry.written
            if value and value not in seen:
                seen.add(value)
                distractors.append(value)
            if len(distractors) >= 3:
                break
        if len(distractors) < 3:
            raise PluginError("not enough demo distractors")
        carrier = f"「{target}」の反対語は「{antonym}」です。"
        start = carrier.index(antonym)
        item = Item(
            id=stable_item_id(
                seed=ctx.seed,
                qtype=self.name,
                point=f"antonym:{target}",
                carrier_hash=sha256_text(carrier),
                key_hash=sha256_text(antonym),
            ),
            type=self.name,
            carrier=carrier,
            slots=[
                ItemSlot(
                    start=start,
                    end=start + len(antonym),
                    span=antonym,
                    slot_type="vocab",
                )
            ],
            key=ItemKey(
                value=antonym,
                lemma=target,
                reading=getattr(spec, "reading", ""),
                pos="",
                sense_id=f"antonym:{target}",
                attestation=[
                    {"kind": "lock_antonym", "ref": f"{target}↔{antonym}"}
                ],
            ),
            distractors=[
                ItemDistractor(value=value, origin="textbook", slot_type="vocab")
                for value in distractors
            ],
            explanation=f"「{target}」的反义是「{antonym}」。",
            meta=ItemMeta(
                lessons=[spec.textbook_lesson] if spec.textbook_lesson else [],
                point=f"antonym:{target}",
                domain="in",
                source="ai",
                seed=ctx.seed,
                clue="反义",
                provenance={
                    "plugin": self.name,
                    "target": target,
                    "legacy_type": self.legacy_type,
                    "batch_id": ctx.batch_id,
                },
            ),
        )
        refresh_hashes(item)
        defects = validate_item(item)
        if defects:
            item.meta.defects = [defect.code for defect in defects]
            refresh_hashes(item)
        return item

    def build_prompt(self, spec: Any) -> dict[str, str]:
        return build_carrier_prompt(spec)

    def distractor_pool(self, spec: Any, ctx: PluginContext) -> list[str]:
        return []

    def clue_injector(self, spec: Any, ctx: PluginContext) -> str:
        return "反义"

    def renderer(self, item: Item, *, seed: int, number: int) -> RenderedQuestion:
        return render_item(item, seed=seed, number=number)

    def local_assertions(self, item: Item) -> list[ItemDefect]:
        return validate_item(item)


question_types.register("demo_antonym", DemoAntonymQuestion())
