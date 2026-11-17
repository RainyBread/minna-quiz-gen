"""Source strategies: original / variant / AI, all producing Items."""

from __future__ import annotations

from typing import Any

from core.contracts import PluginContext, PluginError
from core.items import Item, build_item_from_generated_question
from core.quizgen import generate_llm, generate_original, generate_variant
from core.registry import source_strategies


class OriginalStrategy:
    name = "original"

    def eligible(self, spec: Any, ctx: PluginContext) -> bool:
        return (
            spec.original_item is not None
            and spec.scope != "extend"
            and ctx.item_bank is not None
        )

    def produce(self, spec: Any, ctx: PluginContext) -> Item:
        if not self.eligible(spec, ctx):
            raise PluginError("original strategy is not eligible")
        question = generate_original(
            spec, spec.original_item, batch_id=ctx.batch_id
        )
        return build_item_from_generated_question(question, seed=ctx.seed)


class VariantStrategy:
    name = "variant"

    def eligible(self, spec: Any, ctx: PluginContext) -> bool:
        return (
            spec.original_item is not None
            and spec.scope != "extend"
            and ctx.item_bank is not None
        )

    def produce(self, spec: Any, ctx: PluginContext) -> Item:
        if not self.eligible(spec, ctx):
            raise PluginError("variant strategy is not eligible")
        question = generate_variant(
            spec,
            spec.original_item,
            ctx.require("pools"),
            ctx.require("rng"),
            batch_id=ctx.batch_id,
            point_exposed=bool(
                spec.point_key and spec.point_key in getattr(ctx, "exposed_points", set())
            ),
        )
        return build_item_from_generated_question(question, seed=ctx.seed)


class AiStrategy:
    name = "ai"

    def eligible(self, spec: Any, ctx: PluginContext) -> bool:
        return bool(ctx.llm is not None and getattr(ctx.llm, "available", False))

    def produce(self, spec: Any, ctx: PluginContext) -> Item:
        if not self.eligible(spec, ctx):
            raise PluginError("ai strategy is not eligible")
        question = generate_llm(
            spec,
            ctx.require("llm"),
            ctx.require("settings"),
            ctx.require("pools"),
            ctx.require("rng"),
            extension_pool=ctx.extension_pool,
            mdict=ctx.mdict,
            locks=ctx.locks,
            batch_id=ctx.batch_id,
        )
        return build_item_from_generated_question(question, seed=ctx.seed)


source_strategies.register("original", OriginalStrategy())
source_strategies.register("variant", VariantStrategy())
source_strategies.register("ai", AiStrategy())
