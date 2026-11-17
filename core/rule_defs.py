"""Stable mistake-rule definitions shared by miner, validator and generator."""

from __future__ import annotations

from dataclasses import dataclass

from core.models import QuestionType, RuleSeverity


@dataclass(frozen=True)
class RuleDefinition:
    id: str
    name_zh: str
    description: str
    applies_to: tuple[QuestionType, ...]
    severity: RuleSeverity
    checker: str | None
    semantic: bool = False


R = RuleDefinition

RULE_DEFINITIONS: dict[str, RuleDefinition] = {
    "long_vowel_error": R(
        "long_vowel_error",
        "长音缺失/赘加/位置错误",
        "删除、添加或移动长音（ー、おう、えい等），制造读音长度错误。",
        (QuestionType.KANJI_TO_KANA,),
        RuleSeverity.HIGH,
        "long_vowel_error",
    ),
    "small_kana_error": R(
        "small_kana_error",
        "促音/拗音等小写假名错误",
        "把「っ」「ゃ」「ゅ」「ょ」等小写假名写成大写，或反向误写。",
        (QuestionType.KANJI_TO_KANA,),
        RuleSeverity.HIGH,
        "small_kana_error",
    ),
    "dakuten_error": R(
        "dakuten_error",
        "清音/浊音/半浊音混淆",
        "在か行/さ行/た行/は行之间错误添加或删除浊点、半浊点。",
        (QuestionType.KANJI_TO_KANA,),
        RuleSeverity.HIGH,
        "dakuten_error",
    ),
    "mora_error": R(
        "mora_error",
        "拨音/促音/长音或词尾截断",
        "误加/漏掉「ん」「っ」「ー」，或截断/多写词尾，导致音节结构错误。",
        (QuestionType.KANJI_TO_KANA,),
        RuleSeverity.HIGH,
        "mora_error",
    ),
    "katakana_spelling_error": R(
        "katakana_spelling_error",
        "片假名表记错误",
        "外来语片假名中的长音、促音、拨音或音节数错误。",
        (QuestionType.KANJI_TO_KANA, QuestionType.VOCAB_CHOICE),
        RuleSeverity.HIGH,
        "katakana_spelling_error",
    ),
    "kanji_form_error": R(
        "kanji_form_error",
        "形近汉字/汉字组合错误",
        "用形近字或错误汉字组合替换正确写法（如 准/準、屋頂/屋上）。",
        (QuestionType.KANA_TO_KANJI, QuestionType.VOCAB_CHOICE),
        RuleSeverity.HIGH,
        "kanji_form_error",
    ),
    "homophone_kanji_error": R(
        "homophone_kanji_error",
        "同音汉字词替换",
        "使用读音相同或相近、但意义不同的汉字词作为干扰项。",
        (QuestionType.KANA_TO_KANJI,),
        RuleSeverity.HIGH,
        "homophone_kanji_error",
    ),
    "similar_word_error": R(
        "similar_word_error",
        "近义/同前缀/同课词替换",
        "使用同一语义场、同前缀或同课词汇替换，制造意义混淆。",
        (
            QuestionType.KANA_TO_KANJI,
            QuestionType.VOCAB_CHOICE,
            QuestionType.TRANSLATION_CHOICE,
        ),
        RuleSeverity.MEDIUM,
        None,
        semantic=True,
    ),
    "particle_error": R(
        "particle_error",
        "助词误用",
        "替换句中的助词，破坏格关系、时间/场所/方向等语法功能。",
        (QuestionType.GRAMMAR_FILL, QuestionType.EXPRESSION_CHOICE),
        RuleSeverity.HIGH,
        "particle_error",
    ),
    "conjugation_error": R(
        "conjugation_error",
        "活用形错误",
        "使用同一词干的其他活用形（ます形/て形/た形/假定形等）替换正确形式。",
        (QuestionType.GRAMMAR_FILL, QuestionType.EXPRESSION_CHOICE),
        RuleSeverity.HIGH,
        "conjugation_error",
    ),
    "tense_aspect_error": R(
        "tense_aspect_error",
        "时态/体错误",
        "在需要过去/非过去、完成/持续等时体意义的位置使用错误形式。",
        (QuestionType.GRAMMAR_FILL,),
        RuleSeverity.HIGH,
        "tense_aspect_error",
    ),
    "voice_error": R(
        "voice_error",
        "语态误用（被动/使役/授受）",
        "在需要被动、使役或授受表达的位置使用主动/错误语态。",
        (QuestionType.GRAMMAR_FILL,),
        RuleSeverity.HIGH,
        "voice_error",
    ),
    "collocation_error": R(
        "collocation_error",
        "词语搭配错误",
        "词语本身成立，但与题干中的名词/动词/助词搭配不自然。",
        (QuestionType.VOCAB_CHOICE, QuestionType.EXPRESSION_CHOICE),
        RuleSeverity.MEDIUM,
        None,
        semantic=True,
    ),
    "situational_error": R(
        "situational_error",
        "会话情景/语域不匹配",
        "应答在语法上成立，但不符合对话场景、礼貌程度或说话人关系。",
        (QuestionType.EXPRESSION_CHOICE, QuestionType.GRAMMAR_FILL),
        RuleSeverity.MEDIUM,
        None,
        semantic=True,
    ),
    "counter_affix_error": R(
        "counter_affix_error",
        "接头/接尾/量词错误",
        "量词、接头词或接尾词与名词类别不匹配。",
        (
            QuestionType.KANJI_TO_KANA,
            QuestionType.VOCAB_CHOICE,
            QuestionType.TRANSLATION_CHOICE,
        ),
        RuleSeverity.MEDIUM,
        "counter_affix_error",
    ),
}

CHECKER_IDS: frozenset[str] = frozenset(
    definition.checker
    for definition in RULE_DEFINITIONS.values()
    if definition.checker
)

SEMANTIC_RULE_IDS: frozenset[str] = frozenset(
    definition.id for definition in RULE_DEFINITIONS.values() if definition.semantic
)


def rules_for_type(question_type: QuestionType) -> list[RuleDefinition]:
    return [
        definition
        for definition in RULE_DEFINITIONS.values()
        if question_type in definition.applies_to
    ]


def rule_definition(rule_id: str) -> RuleDefinition | None:
    return RULE_DEFINITIONS.get(rule_id)
