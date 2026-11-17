from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from config import Settings, ensure_directories
from core.models import (
    GeneratedOption,
    GeneratedQuestion,
    GeneratedTarget,
    OptionRole,
    QuestionType,
)


def temp_settings(settings: Settings, tmp_path: Path) -> Settings:
    value = replace(
        settings,
        data_dir=tmp_path / "data",
        db_dir=tmp_path / "db",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
    )
    ensure_directories(value)
    return value


def make_vocab_question() -> GeneratedQuestion:
    return GeneratedQuestion(
        question_id="q-vocab",
        type=QuestionType.VOCAB_CHOICE,
        textbook_lesson=44,
        lesson=8,
        target=GeneratedTarget(
            surface="学校",
            written="学校",
            reading="がっこう",
            meaning_zh="学校",
        ),
        stem="私は[[blank]]へ行きます。",
        options=[
            GeneratedOption(key="A", text="学校", role=OptionRole.CORRECT),
            GeneratedOption(key="B", text="会社", role=OptionRole.DISTRACTOR),
            GeneratedOption(key="C", text="銀行", role=OptionRole.DISTRACTOR),
            GeneratedOption(key="D", text="駅", role=OptionRole.DISTRACTOR),
        ],
        answer_key="A",
        correct_option_text="学校",
        explanation_zh="正确项是「学校」；中文为「学校」。",
        source_kind="new",
        scope="in_syllabus",
        point_key="学校",
    )


def make_grammar_question() -> GeneratedQuestion:
    return GeneratedQuestion(
        question_id="q-grammar",
        type=QuestionType.GRAMMAR_FILL,
        textbook_lesson=44,
        lesson=8,
        target=GeneratedTarget(
            surface="食べたそうな",
            written="食べたそうな",
            reading="たべたそうな",
            meaning_zh="看起来想吃",
        ),
        stem="弟はケーキを[[blank]]顔をしています。",
        options=[
            GeneratedOption(key="A", text="食べたそうな", role=OptionRole.CORRECT),
            GeneratedOption(key="B", text="食べそうな", role=OptionRole.DISTRACTOR),
            GeneratedOption(key="C", text="食べるそうな", role=OptionRole.DISTRACTOR),
            GeneratedOption(key="D", text="食べたような", role=OptionRole.DISTRACTOR),
        ],
        answer_key="A",
        correct_option_text="食べたそうな",
        explanation_zh="正确项是「食べたそうな」；表示看起来想吃。",
        source_kind="new",
        scope="in_syllabus",
        point_key="gp-44-01",
    )


def make_reading_question() -> GeneratedQuestion:
    return GeneratedQuestion(
        question_id="q-reading",
        type=QuestionType.KANJI_TO_KANA,
        textbook_lesson=44,
        lesson=8,
        target=GeneratedTarget(
            surface="学校",
            written="学校",
            reading="がっこう",
            meaning_zh="学校",
        ),
        stem="次の語の読み方として最もよいものを選びなさい。\n学校。",
        options=[
            GeneratedOption(key="A", text="がっこう", role=OptionRole.CORRECT),
            GeneratedOption(key="B", text="がこう", role=OptionRole.DISTRACTOR),
            GeneratedOption(key="C", text="かっこう", role=OptionRole.DISTRACTOR),
            GeneratedOption(key="D", text="がっこ", role=OptionRole.DISTRACTOR),
        ],
        answer_key="A",
        correct_option_text="がっこう",
        explanation_zh="「学校」读作「がっこう」。",
        source_kind="new",
        scope="in_syllabus",
        point_key="学校",
    )
