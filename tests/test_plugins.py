from __future__ import annotations

import random

import plugins  # noqa: F401 - registration side effect
from core.contracts import PluginContext
from core.items import validate_item
from core.locks import LockTable
from core.models import QuestionType
from core.pools import TextbookPools
from core.quizgen import QuestionSpec
from core.registry import (
    dict_sources,
    exporters,
    llm_providers,
    plugin_status,
    question_types,
    source_strategies,
)


def test_five_registries_are_populated():
    status = plugin_status()
    assert "kanji_to_kana" in status["question_types"]
    assert "demo_antonym" in status["question_types"]
    assert "textbook" in status["dict_sources"]
    assert "ai" in status["source_strategies"]
    assert "demo" in status["exporters"]
    assert "openai" in status["llm_providers"]


def test_demo_question_type_builds_item(settings, context):
    plugin = question_types.get("demo_antonym")
    spec = QuestionSpec(
        seq=1,
        qtype=QuestionType.VOCAB_CHOICE,
        written="かたい",
        textbook_lesson=44,
        course_lesson=8,
        point_key="かたい",
    )
    ctx = PluginContext(
        settings=settings,
        context=context,
        pools=TextbookPools(context),
        rng=random.Random(1),
        locks=LockTable({"antonyms": [["かたい", "やわらかい"]]}),
        seed=42,
    )
    item = plugin.build_local(spec, ctx)
    assert item is not None
    assert item.type == "demo_antonym"
    assert item.key.value == "やわらかい"
    assert validate_item(item) == []
    rendered = plugin.renderer(item, seed=42, number=1)
    assert "（　　）" in rendered.stem


def test_demo_exporter_writes_file(settings, context, tmp_path):
    plugin = question_types.get("demo_antonym")
    spec = QuestionSpec(
        seq=1,
        qtype=QuestionType.VOCAB_CHOICE,
        written="かたい",
        textbook_lesson=44,
        point_key="かたい",
    )
    ctx = PluginContext(
        settings=settings,
        context=context,
        pools=TextbookPools(context),
        rng=random.Random(2),
        locks=LockTable({"antonyms": [["かたい", "やわらかい"]]}),
        seed=42,
    )
    item = plugin.build_local(spec, ctx)
    target = exporters.get("demo").export([item], tmp_path, seed=42)
    assert target.exists()
    assert item.id in target.read_text(encoding="utf-8")


def test_textbook_dict_source(settings, context):
    source = dict_sources.get("textbook")(settings, context)
    entry = context.vocab.all_entries()[0]
    fact = source.lookup(entry.written)
    assert fact is not None
    assert fact.reading == entry.kana
    assert fact.source_kind == "textbook_vocab"


def test_llm_provider_is_config_selected(settings):
    provider = llm_providers.get("openai")(settings, offline=True)
    assert provider.available is False


def test_demo_type_flows_through_plan_and_runner(settings, context, tmp_path):
    from core.intent import parse_plan
    from core.runner import QuizRunner
    from tests.helpers_items import temp_settings

    local = temp_settings(settings, tmp_path)
    plan = parse_plan("生成1道反义词题", context)
    assert plan.type_plan == {"demo_antonym": 1}
    result = QuizRunner(
        local,
        context,
        llm=None,
        locks=LockTable({"antonyms": [["かたい", "やわらかい"]]}),
        workers=1,
    ).run(plan, stream=False)
    assert result.batch is not None
    assert len(result.batch.items) == 1
    assert result.batch.items[0]["type"] == "demo_antonym"


def test_ai_source_strategy_produces_item(settings, context):
    pools = TextbookPools(context)
    entry = pools.vocab_candidates([44])[0]

    class FakeLLM:
        available = True

        def complete_text(self, system, user, **kwargs):
            return f"[s]この店では[t]{entry.written}[/t]が人気です。[/s]"

    spec = QuestionSpec(
        seq=2,
        qtype=QuestionType.VOCAB_CHOICE,
        written=entry.written,
        reading=entry.kana,
        meaning=entry.translation_zh,
        correct_text=entry.written,
        textbook_lesson=entry.lesson,
        course_lesson=entry.lesson - 36,
        entry=entry,
        point_key=entry.written,
    )
    ctx = PluginContext(
        settings=settings,
        context=context,
        pools=pools,
        rng=random.Random(3),
        llm=FakeLLM(),
        seed=7,
        batch_id="plugin-test",
    )
    strategy = source_strategies.get("ai")
    assert strategy.eligible(spec, ctx)
    item = strategy.produce(spec, ctx)
    assert item.type == QuestionType.VOCAB_CHOICE.value
    assert validate_item(item) == []
