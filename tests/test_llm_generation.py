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


class CarrierFakeLLM:
    available = True

    def __init__(self, correct: str):
        self.correct = correct
        self.prompts: list[tuple[str, str]] = []

    def complete_text(self, system, user, **kwargs):
        self.prompts.append((system, user))
        return (
            f"[s]弟はケーキを[t]{self.correct}[/t]顔をしています。[/s]"
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


def test_carrier_prompt_does_not_send_answer(settings, context):
    pools = TextbookPools(context)
    correct = "食べたそうな"
    spec = QuestionSpec(
        seq=2,
        qtype=QuestionType.GRAMMAR_FILL,
        scope="in_syllabus",
        source_kind="new",
        pattern="〜そうな",
        meaning="看起来想吃",
        correct_text=correct,
        approved_forms=[correct],
        textbook_lesson=44,
        course_lesson=8,
        point_key="gp-44-01",
    )
    fake = CarrierFakeLLM(correct)
    question = generate_llm(
        spec,
        fake,
        settings,
        pools,
        random.Random(3),
        batch_id="test-carrier",
    )
    assert len(question.options) == 4
    assert "[[blank]]" in question.stem
    assert question.correct_option_text == correct
    assert fake.prompts
    system, user = fake.prompts[0]
    assert correct not in user
    assert correct not in system
    assert "[s]" in user and "[t]" in user


def test_non_approved_grammar_span_is_kept_with_defect(settings, context):
    pools = TextbookPools(context)
    spec = QuestionSpec(
        seq=3,
        qtype=QuestionType.GRAMMAR_FILL,
        scope="in_syllabus",
        source_kind="new",
        pattern="〜ために",
        meaning="为了……",
        correct_text="ために",
        approved_forms=["ために"],
        textbook_lesson=44,
        course_lesson=8,
        point_key="gp-44-02",
    )
    question = generate_llm(
        spec,
        CarrierFakeLLM("から"),
        settings,
        pools,
        random.Random(3),
        batch_id="test-defect",
    )
    assert "E_ZERO" in spec.pending_defects
    assert "[[blank]]" in question.stem
