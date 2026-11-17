"""Lightweight runtime context loading.

Only compact structured JSON is read here.  Importing corpora, OCR, mining or
PDF tooling is deliberately *not* part of the runtime path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import Settings
from core.models import (
    GrammarFile,
    MistakeRulesFile,
    QuestionTypesFile,
    TextbookVocab,
)
from core.novelty import NoveltyIndex


class ContextError(RuntimeError):
    pass


def _load_json(path: Path, *, required: bool = False) -> Any:
    if not path.exists():
        if required:
            raise ContextError(f"缺少运行数据：{path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContextError(f"无法读取 {path}: {exc}") from exc


@dataclass
class AppContext:
    settings: Settings
    vocab: TextbookVocab
    grammar: GrammarFile
    question_types: QuestionTypesFile = field(default_factory=QuestionTypesFile)
    rules: MistakeRulesFile = field(default_factory=MistakeRulesFile)
    course: dict[str, Any] = field(default_factory=dict)
    novelty: NoveltyIndex = field(default_factory=NoveltyIndex)


def load_context(settings: Settings, *, require_rules: bool = False) -> AppContext:
    vocab_data = _load_json(settings.vocab_path, required=True)
    grammar_data = _load_json(settings.grammar_path, required=True)
    try:
        vocab = TextbookVocab.model_validate(vocab_data)
        grammar = GrammarFile.model_validate(grammar_data)
    except Exception as exc:
        raise ContextError(f"教材数据结构无效：{exc}") from exc

    types_data = _load_json(settings.question_types_path)
    rules_data = _load_json(settings.common_mistakes_path)
    if require_rules and (types_data is None or rules_data is None):
        raise ContextError(
            "缺少规则库 question_types.json / common_mistakes.json；"
            "请先运行 python init.py --remine 或从仓库数据恢复。"
        )
    question_types = (
        QuestionTypesFile.model_validate(types_data)
        if types_data
        else QuestionTypesFile()
    )
    rules = (
        MistakeRulesFile.model_validate(rules_data)
        if rules_data
        else MistakeRulesFile()
    )
    course = _load_json(settings.course_profile_path) or {}
    novelty = NoveltyIndex.load(settings.novelty_index_path)
    return AppContext(
        settings=settings,
        vocab=vocab,
        grammar=grammar,
        question_types=question_types,
        rules=rules,
        course=course if isinstance(course, dict) else {},
        novelty=novelty,
    )
