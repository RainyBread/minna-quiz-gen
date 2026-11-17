import random

from core.pools import TextbookPools
from core.quizgen import QuestionSpec, generate_llm
from core.models import QuestionType


class FakeLLM:
    available = True

    def __init__(self, correct: str):
        self.correct = correct

    def complete_text(self, system, user, **kwargs):
        return (
            "[CLUE]店で人気の品[/CLUE]\n"
            f"[SENTENCE]この店では [[ans]]{self.correct}[[/ans]] が人気です。[/SENTENCE]\n"
            "[ZH]测试翻译[/ZH]"
        )


def test_llm_question_uses_full_sentence_and_pins_correct(settings, context):
    pools = TextbookPools(context)
    entry = pools.vocab_candidates([44])[0]
    spec = QuestionSpec(
        seq=1,
        qtype=QuestionType.VOCAB_CHOICE,
        scope="in_syllabus",
        source_kind="vocab",
        source_id=entry.entry_id,
        written=entry.written,
        reading=entry.kana,
        meaning=entry.translation_zh,
        correct_text=entry.written,
        textbook_lesson=entry.lesson,
        course_lesson=entry.lesson - 36,
        entry=entry,
        point_key=entry.written,
    )
    question = generate_llm(
        spec,
        FakeLLM(entry.written),
        settings,
        pools,
        random.Random(3),
        batch_id="test",
    )
    assert len(question.options) == 4
    assert question.answer_key in {option.key for option in question.options}
    assert "[[blank]]" in question.stem
    assert question.correct_option_text == entry.written
