from dataclasses import replace

from config import ensure_directories
from core.intent import QuizPlan
from core.runner import QuizRunner


def test_local_reading_and_kanji_without_unidic(settings, context, tmp_path):
    tmp_settings = replace(
        settings,
        data_dir=tmp_path / "data",
        db_dir=tmp_path / "db",
        cache_dir=tmp_path / "cache",
        log_dir=tmp_path / "logs",
    )
    ensure_directories(tmp_settings)
    plan = QuizPlan(
        raw="test",
        mode="practice",
        scope="in_syllabus",
        count=2,
        lessons=[37],
        type_plan={"kanji_to_kana": 1, "kana_to_kanji": 1},
        seed=7,
        source_mix={"new": 1.0},
    )
    result = QuizRunner(
        tmp_settings, context, llm=None, item_bank=None, workers=1
    ).run(plan, stream=False)
    assert result.batch is not None
    assert len(result.batch.questions) == 2
    for question in result.batch.questions:
        assert len(question.options) == 4
        correct = [option for option in question.options if option.role.value == "correct"]
        assert len(correct) == 1
        assert correct[0].key == question.answer_key
