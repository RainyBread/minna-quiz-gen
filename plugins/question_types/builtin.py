"""Built-in question-type adapters.

The implementation still reuses the vetted construction helpers in
``core.quizgen``; the registry contract is the extension point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.contracts import PluginContext, SlotType
from core.items import (
    Item,
    ItemDefect,
    build_item_from_generated_question,
    validate_item,
)
from core.prompts import build_carrier_prompt
from core.quizgen import (
    _fallback_grammar_distractors,
    _fallback_vocab_distractors,
    generate_local,
)
from core.registry import question_types
from core.render import RenderedQuestion, render_item


@dataclass
class BuiltinQuestionType:
    name: str
    legacy_type: str
    is_local: bool
    slot_type: SlotType

    def build_local(self, spec: Any, ctx: PluginContext) -> Item | None:
        if not self.is_local:
            return None
        question = generate_local(
            spec,
            ctx.require("pools"),
            ctx.require("rng"),
            batch_id=ctx.batch_id,
        )
        return build_item_from_generated_question(question, seed=ctx.seed)

    def build_prompt(self, spec: Any) -> dict[str, str]:
        return build_carrier_prompt(spec)

    def distractor_pool(self, spec: Any, ctx: PluginContext) -> list[str]:
        pools = ctx.require("pools")
        rng = ctx.require("rng")
        if spec.qtype.value == "vocab_choice":
            if ctx.locks is not None:
                from core.pools import entry_pos

                pos_group = (
                    pools.pos_group(entry_pos(spec.entry))
                    if spec.entry is not None
                    else ""
                )
                values = ctx.locks.select_vocab_distractors(
                    spec.correct_text,
                    pos_group,
                    "",
                    pools,
                    rng,
                    k=6,
                    mdict=ctx.mdict,
                )
                if len(values) >= 3:
                    return values
            return _fallback_vocab_distractors(spec, pools, rng, 8)
        if spec.qtype.value in {"grammar_fill", "expression_choice"}:
            return _fallback_grammar_distractors(spec, pools, rng, 8)
        return []

    def clue_injector(self, spec: Any, ctx: PluginContext) -> str:
        if ctx.locks is None:
            return ""
        return ctx.locks.clue_hint(spec.correct_text)

    def renderer(self, item: Item, *, seed: int, number: int) -> RenderedQuestion:
        return render_item(item, seed=seed, number=number)

    def local_assertions(self, item: Item) -> list[ItemDefect]:
        return validate_item(item)


_BUILTINS = {
    "kanji_to_kana": BuiltinQuestionType(
        "kanji_to_kana", "kanji_to_kana", True, SlotType("reading")
    ),
    "kana_to_kanji": BuiltinQuestionType(
        "kana_to_kanji", "kana_to_kanji", True, SlotType("kanji")
    ),
    "vocab_choice": BuiltinQuestionType(
        "vocab_choice", "vocab_choice", False, SlotType("vocab")
    ),
    "expression_choice": BuiltinQuestionType(
        "expression_choice", "expression_choice", False, SlotType("expression")
    ),
    "grammar_fill": BuiltinQuestionType(
        "grammar_fill", "grammar_fill", False, SlotType("grammar_form")
    ),
}

for _name, _plugin in _BUILTINS.items():
    question_types.register(_name, _plugin)
