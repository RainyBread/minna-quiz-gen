"""Deterministic rule mining, optional LLM refinement, and human review.

The deterministic pass is always authoritative for statistics and evidence.
The LLM may add evidence or semantic rules, but it can never overwrite an
approved/locked rule, invent a checker, or silently change the blueprint.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from config import Settings
from core.llm import LLMClient, LLMError
from core.models import (
    BLUEPRINT_2024,
    BLUEPRINT_JP1_NOTICE,
    Blueprint,
    CorrectDesign,
    MistakeRule,
    MistakeRulesFile,
    QualityStatus,
    QuestionType,
    QuestionTypeRule,
    QuestionTypesFile,
    RawBank,
    RawQuestion,
    RuleEvidence,
    RuleSeverity,
    RuleStatus,
    TYPE_LABELS_ZH,
    contains_kanji,
    has_japanese,
    is_katakana_only,
    is_reading_text,
    normalize_text,
    now_iso,
    sha256_text,
    to_hiragana,
)
from core.rule_defs import (
    CHECKER_IDS,
    RULE_DEFINITIONS,
    RuleDefinition,
    rule_definition,
)

try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


PARTICLES = {
    "は", "が", "を", "に", "で", "と", "も", "の", "から", "まで", "より",
    "へ", "や", "か", "しか", "だけ", "とか", "など", "ので", "のに", "ながら",
    "ば", "たら", "ても", "ては", "では", "こそ", "さえ", "でも", "および",
    "によって", "について", "として", "に対して", "にとって", "ために", "ように",
    "ほど", "くらい", "ぐらい", "ばかり", "ずつ", "って", "なら", "でも",
}
SMALL_KANA_MAP = str.maketrans("っゃゅょゎ", "つやゆよわ")
DAKUTEN_MAP = str.maketrans(
    "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ",
    "かきくけこさしすせそたちつてとはひふへほはひふへほ",
)
KANA_PUNCT = "～〜~－-－/・「」『』（）()。、，,！？!?　 "
TENSE_ENDINGS = ("ました", "ません", "ませんでした", "ました", "ている", "ています", "ていた", "ていました", "た", "て", "ない", "なかった", "でしょう", "だろう")
VOICE_ENDINGS = ("れる", "られる", "せる", "させる", "させられる", "てもらう", "ていただく", "てくれる", "てくださる", "てあげる", "てやる")


@dataclass
class RuleAggregate:
    rule_id: str
    count: int = 0
    evidence: list[RuleEvidence] = field(default_factory=list)
    source_families: set[str] = field(default_factory=set)
    types: Counter[str] = field(default_factory=Counter)

    def add(self, question: RawQuestion, correct: str, distractor: str, note: str) -> None:
        self.count += 1
        self.types[question.type.value] += 1
        family = question.source.kind or "unknown"
        self.source_families.add(family)
        if len(self.evidence) < 5:
            self.evidence.append(
                RuleEvidence(
                    source_id=question.source.source_id,
                    question_id=question.raw_id,
                    target=question.target or "",
                    correct=correct,
                    distractor=distractor,
                    note=note,
                    source_family=family,
                    ocr_quality="flagged" if question.flags.has_blocking_issues else "clean",
                )
            )


@dataclass
class MiningResult:
    question_types: QuestionTypesFile
    mistakes: MistakeRulesFile
    report: dict[str, Any]
    draft_types_path: Path
    draft_mistakes_path: Path
    has_existing_approved: bool = False
    first_approval: bool = False
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    llm_notes: list[str] = field(default_factory=list)
    aggregates: dict[str, RuleAggregate] = field(default_factory=dict)
    observed_counts: Counter[str] = field(default_factory=Counter)
    clean_questions: list[RawQuestion] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loading and statistics
# ---------------------------------------------------------------------------


def _load_bank(path: Path) -> RawBank | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return RawBank.model_validate(json.load(handle))
    except Exception:
        return None


def _load_questions(settings: Settings) -> tuple[list[RawQuestion], dict[str, Any]]:
    banks: list[RawBank] = []
    source_status: dict[str, Any] = {}
    for label, path in [
        ("mock", settings.raw_mock_path),
        ("legacy", settings.raw_legacy_path),
        ("exam2024", settings.raw_exams_path),
    ]:
        bank = _load_bank(path)
        if bank is None:
            source_status[label] = {"status": "missing_or_invalid", "path": str(path)}
            continue
        banks.append(bank)
        source_status[label] = {
            "status": "loaded",
            "path": str(path),
            "questions": len(bank.questions),
            "meta_quality": bank.meta.quality,
        }
    questions = [question for bank in banks for question in bank.questions]
    return questions, source_status


def _is_clean(question: RawQuestion) -> bool:
    return (
        question.status == QualityStatus.IMPORTED
        and not question.flags.has_blocking_issues
    )


def _corpus_hash(settings: Settings) -> str:
    parts: list[str] = []
    for path in [
        settings.raw_mock_path,
        settings.raw_legacy_path,
        settings.raw_exams_path,
        settings.blueprint_path,
    ]:
        if path.exists():
            parts.append(f"{path.name}:{sha256_text(path.read_bytes().decode('utf-8', 'ignore'))}")
    return sha256_text("|".join(parts))


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _kana_clean(text: str) -> str:
    value = to_hiragana(normalize_text(text))
    return value.translate({ord(ch): None for ch in KANA_PUNCT}).strip()


def _strip_dakuten(value: str) -> str:
    return value.translate(DAKUTEN_MAP)


def _long_vowel_diff(a: str, b: str) -> bool:
    if a == b:
        return False
    if a.replace("ー", "") == b.replace("ー", ""):
        return True
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    if len(longer) - len(shorter) == 1:
        for index, char in enumerate(longer):
            if longer[:index] + longer[index + 1 :] == shorter and (
                char == "ー" or char in "あいうえお"
            ):
                return True
    if len(a) == len(b):
        diffs = [(x, y) for x, y in zip(a, b) if x != y]
        if 0 < len(diffs) <= 2 and all(
            x == "ー" or y == "ー" or x in "あいうえお" or y in "あいうえお"
            for x, y in diffs
        ):
            return True
    return False


def _classify_kana(correct: str, distractor: str) -> tuple[str | None, str]:
    a = _kana_clean(correct)
    b = _kana_clean(distractor)
    if not a or not b or a == b:
        return None, ""
    if _long_vowel_diff(a, b):
        return "long_vowel_error", "长音增删/位置差异"
    if a.translate(SMALL_KANA_MAP) == b.translate(SMALL_KANA_MAP):
        return "small_kana_error", "促音/拗音小写假名差异"
    if _strip_dakuten(a) == _strip_dakuten(b):
        return "dakuten_error", "清浊音/半浊音差异"
    if (is_katakana_only(correct) or is_katakana_only(distractor)) and not (
        contains_kanji(correct) or contains_kanji(distractor)
    ):
        return "katakana_spelling_error", "片假名长音/音节表记差异"
    if a.startswith(b) or b.startswith(a):
        return "mora_error", "词尾截断/多写"
    if SequenceMatcher(None, a, b).ratio() >= 0.35:
        return "mora_error", "音节/拨音/促音差异"
    return None, ""


def _classify_kanji(correct: str, distractor: str, qtype: QuestionType) -> tuple[str | None, str]:
    a = normalize_text(correct)
    b = normalize_text(distractor)
    if not a or not b or a == b:
        return None, ""
    ratio = SequenceMatcher(None, a, b).ratio()
    shared_kanji = set(c for c in a if contains_kanji(c)) & set(c for c in b if contains_kanji(c))
    if ratio >= 0.4 or shared_kanji:
        return "kanji_form_error", "形近汉字/汉字组合差异"
    if qtype == QuestionType.KANA_TO_KANJI:
        return "similar_word_error", "同音/同前缀/形近词替换"
    return "collocation_error", "同类词但搭配不成立"


def _classify_grammar(correct: str, distractor: str, qtype: QuestionType) -> tuple[str | None, str]:
    a = normalize_text(correct).strip("。、！？!? ")
    b = normalize_text(distractor).strip("。、！？!? ")
    if not a or not b or a == b:
        return None, ""
    stripped = lambda value: value.replace("……", "").replace("...", "").replace("…", "")
    if stripped(a) in PARTICLES and stripped(b) in PARTICLES:
        return "particle_error", "助词替换"
    if len(a) > 8 or len(b) > 8:
        return "situational_error", "会话情景/语体不匹配"
    if qtype == QuestionType.EXPRESSION_CHOICE:
        return "situational_error", "会话应答不匹配"
    common = os.path.commonprefix([a, b])
    if len(common) >= max(1, min(len(a), len(b)) - 4):
        if any(end in b for end in VOICE_ENDINGS) or any(end in a for end in VOICE_ENDINGS):
            return "voice_error", "语态/授受表达差异"
        if any(end in b for end in TENSE_ENDINGS) or any(end in a for end in TENSE_ENDINGS):
            return "tense_aspect_error", "时态/体差异"
        return "conjugation_error", "活用形差异"
    if len(a) <= 3 and len(b) <= 3 and all(is_reading_text(x) for x in (a, b)):
        return "particle_error", "短助词/功能词替换"
    return "situational_error", "语法/情景不匹配"


def _classify_pair(
    question: RawQuestion, correct: str, distractor: str
) -> tuple[str | None, str]:
    qtype = question.type
    if qtype == QuestionType.KANJI_TO_KANA:
        return _classify_kana(correct, distractor)
    if qtype == QuestionType.KANA_TO_KANJI:
        return _classify_kanji(correct, distractor, qtype)
    if qtype == QuestionType.VOCAB_CHOICE:
        if contains_kanji(correct) and contains_kanji(distractor):
            return _classify_kanji(correct, distractor, qtype)
        if is_katakana_only(correct) and is_katakana_only(distractor):
            return "similar_word_error", "同类外来语替换"
        return "collocation_error", "词语搭配不成立"
    if qtype == QuestionType.TRANSLATION_CHOICE:
        return "similar_word_error", "同课/同类词义替换"
    if qtype in {QuestionType.GRAMMAR_FILL, QuestionType.EXPRESSION_CHOICE}:
        return _classify_grammar(correct, distractor, qtype)
    return None, ""


def _rule_confidence(aggregate: RuleAggregate, definition: RuleDefinition) -> float:
    base = 0.52 + min(0.28, aggregate.count * 0.012) + min(0.15, len(aggregate.source_families) * 0.05)
    if definition.checker is None:
        base = min(base, 0.82)
    return round(min(0.98, base), 3)


# ---------------------------------------------------------------------------
# Deterministic mining
# ---------------------------------------------------------------------------


def _mine_deterministic(
    settings: Settings, questions: list[RawQuestion], source_status: dict[str, Any]
) -> tuple[dict[str, RuleAggregate], Counter[str], dict[str, Any]]:
    aggregates: dict[str, RuleAggregate] = {}
    observed: Counter[str] = Counter()
    unclassified = 0
    skipped_no_answer = 0
    clean_count = 0

    for question in questions:
        if not _is_clean(question):
            continue
        clean_count += 1
        observed[question.type.value] += 1
        if question.type in {
            QuestionType.SENTENCE_ORDERING,
            QuestionType.CLOZE,
            QuestionType.READING,
        }:
            continue
        option_map = {opt.key.upper(): opt.text for opt in question.options}
        answer_key = (question.answer_key or "").upper()
        if not answer_key or answer_key not in option_map:
            skipped_no_answer += 1
            continue
        correct = option_map[answer_key]
        for key, text in option_map.items():
            if key == answer_key:
                continue
            rule_id, note = _classify_pair(question, correct, text)
            if not rule_id:
                unclassified += 1
                continue
            definition = rule_definition(rule_id)
            if definition is None or question.type not in definition.applies_to:
                unclassified += 1
                continue
            aggregate = aggregates.setdefault(rule_id, RuleAggregate(rule_id=rule_id))
            aggregate.add(question, correct, text, note)

    stats = {
        "clean_questions": clean_count,
        "observed_counts": dict(observed),
        "unclassified_distractors": unclassified,
        "skipped_no_answer": skipped_no_answer,
        "source_status": source_status,
        "rule_aggregates": {
            rule_id: {
                "count": aggregate.count,
                "types": dict(aggregate.types),
                "source_families": sorted(aggregate.source_families),
            }
            for rule_id, aggregate in aggregates.items()
        },
    }
    return aggregates, observed, stats


def _build_mistake_rules(
    aggregates: dict[str, RuleAggregate],
    existing: MistakeRulesFile | None,
    first_approval: bool,
) -> list[MistakeRule]:
    existing_map = {rule.id: rule for rule in existing.rules} if existing else {}
    rules: list[MistakeRule] = []
    for rule_id, aggregate in sorted(aggregates.items(), key=lambda item: (-item[1].count, item[0])):
        definition = rule_definition(rule_id)
        if definition is None:
            continue
        if rule_id in existing_map and existing_map[rule_id].status in {
            RuleStatus.APPROVED,
            RuleStatus.LOCKED,
        }:
            rules.append(existing_map[rule_id])
            continue
        rules.append(
            MistakeRule(
                id=rule_id,
                name_zh=definition.name_zh,
                description=definition.description,
                applies_to=list(definition.applies_to),
                severity=definition.severity,
                checker=definition.checker,
                confidence=_rule_confidence(aggregate, definition),
                evidence_count=aggregate.count,
                evidence=aggregate.evidence,
                locked=False,
                status=RuleStatus.DRAFT,
                version=1,
            )
        )
    if existing:
        for rule in existing.rules:
            if rule.id not in {r.id for r in rules}:
                rules.append(rule)
    return rules


def _correct_design_for(qtype: QuestionType) -> CorrectDesign:
    mapping = {
        QuestionType.KANJI_TO_KANA: CorrectDesign(
            strategy="从教材词条取 written，正确答案由 UniDic 的 kana 字段确定性注入",
            answer_source="unidic.kana",
            verification="unidic_reading_equals_answer",
        ),
        QuestionType.KANA_TO_KANJI: CorrectDesign(
            strategy="从教材词条取 written，正确答案为教材汉字写法",
            answer_source="textbook.vocab.written",
            verification="textbook_written_equals_answer_and_homophone_check",
        ),
        QuestionType.VOCAB_CHOICE: CorrectDesign(
            strategy="从教材词条/释义选定正确词，LLM 只补语境与干扰项",
            answer_source="textbook.vocab",
            verification="source_exists_plus_blind_solve",
        ),
        QuestionType.EXPRESSION_CHOICE: CorrectDesign(
            strategy="从已审核 GrammarPoint 的 approved_answer_forms 选定正确应答",
            answer_source="grammar.approved_answer_forms",
            verification="exactly_one_approved_form_plus_blind_solve",
        ),
        QuestionType.GRAMMAR_FILL: CorrectDesign(
            strategy="从已审核 GrammarPoint 的 approved_answer_forms 选定正确形式",
            answer_source="grammar.approved_answer_forms",
            verification="exactly_one_approved_form_plus_blind_solve",
        ),
    }
    return mapping.get(
        qtype,
        CorrectDesign(
            strategy="记录型题型：首版不生成",
            answer_source="none",
            verification="record_only",
        ),
    )


def _prompt_hint_for(qtype: QuestionType) -> str:
    mapping = {
        QuestionType.KANJI_TO_KANA: "干扰项优先用长音/促音/浊音/音节变形；不要用无关词义替换。",
        QuestionType.KANA_TO_KANJI: "干扰项用形近汉字、同音词或同前缀词，并确保读音不重现题干假名。",
        QuestionType.VOCAB_CHOICE: "干扰项与正确词同词性/同语义场，但搭配或语境不成立。",
        QuestionType.EXPRESSION_CHOICE: "干扰项语法上可能成立，但情景、礼貌程度或说话人关系不匹配。",
        QuestionType.GRAMMAR_FILL: "干扰项优先助词/活用/时态/语态错误；禁止多解。",
    }
    return mapping.get(qtype, "记录型题型，首版不生成。")


def _build_question_types(
    aggregates: dict[str, RuleAggregate],
    observed: Counter[str],
    existing: QuestionTypesFile | None,
    corpus_sha: str,
    first_approval: bool,
) -> QuestionTypesFile:
    observed_total = max(1, sum(observed.values()))
    blueprint_types = list(BLUEPRINT_2024.keys())
    blueprint_observed_total = max(
        1, sum(observed.get(qtype, 0) for qtype in blueprint_types)
    )
    raw_weights: dict[str, float] = {}
    for qtype in QuestionType:
        if qtype.value in BLUEPRINT_2024:
            blueprint_weight = BLUEPRINT_2024[qtype.value] / 100.0
            observed_weight = observed.get(qtype.value, 0) / blueprint_observed_total
            raw_weights[qtype.value] = 0.7 * blueprint_weight + 0.3 * observed_weight
        else:
            raw_weights[qtype.value] = 0.0
    weight_sum = sum(raw_weights.values()) or 1.0
    weights = {key: round(value / weight_sum, 6) for key, value in raw_weights.items()}

    existing_types = existing.types if existing else {}
    types: dict[str, QuestionTypeRule] = {}
    for qtype in QuestionType:
        existing_rule = existing_types.get(qtype.value)
        allowed = [
            rule_id
            for rule_id, aggregate in aggregates.items()
            if qtype in (rule_definition(rule_id).applies_to if rule_definition(rule_id) else ())
            and aggregate.count > 0
        ]
        if existing_rule is not None:
            merged_allowed = sorted(set(existing_rule.allowed_mistake_rules) | set(allowed))
            types[qtype.value] = QuestionTypeRule(
                type=qtype,
                name_zh=existing_rule.name_zh,
                weight=existing_rule.weight,
                observed_weight=observed.get(qtype.value, 0) / observed_total,
                blueprint_count=BLUEPRINT_2024.get(qtype.value, existing_rule.blueprint_count or 0),
                supported=existing_rule.supported,
                option_count=existing_rule.option_count,
                correct_design=existing_rule.correct_design,
                allowed_mistake_rules=merged_allowed,
                prompt_hint=existing_rule.prompt_hint,
                evidence_count=observed.get(qtype.value, 0),
            )
        else:
            types[qtype.value] = QuestionTypeRule(
                type=qtype,
                name_zh=TYPE_LABELS_ZH.get(qtype.value, qtype.value),
                weight=weights.get(qtype.value, 0.0),
                observed_weight=observed.get(qtype.value, 0) / observed_total,
                blueprint_count=BLUEPRINT_2024.get(qtype.value, 0),
                supported=qtype.is_generatable,
                option_count=4,
                correct_design=_correct_design_for(qtype),
                allowed_mistake_rules=sorted(set(allowed)),
                prompt_hint=_prompt_hint_for(qtype),
                evidence_count=observed.get(qtype.value, 0),
            )

    blueprint = Blueprint(
        source="2024年日语期末试卷（官方100题）",
        total_questions=100,
        counts=BLUEPRINT_2024,
        missing=["q13（OCR 不可用）"],
        reference_blueprints=[
            {
                "source": "25级综合日语1期末测试通知（95题，跨课程参考）",
                "total_questions": 95,
                "counts": BLUEPRINT_JP1_NOTICE,
                "confidence": "low",
            }
        ],
        confidence="high",
        note="主配比来自 2024 大二上期末真题；第13题 OCR 不可用；记录型题型首版不生成。",
    )
    version = 1
    if existing:
        version = int(existing.meta.get("version", 1)) + (0 if first_approval else 1)
    return QuestionTypesFile(
        meta={
            "semester": "big2_autumn",
            "version": version,
            "reviewed": False,
            "generated_at": now_iso(),
            "generator": "deterministic",
            "corpus_sha256": corpus_sha,
            "notes": [
                "设正库：每个题型由 correct_design 描述正确答案来源。",
                "设误库：每个干扰项必须绑定 common_mistakes.rules 中 approved/locked 的规则。",
                "记录型题型（translation_choice/sentence_ordering/cloze/reading）只记录配比，首版不生成。",
            ],
        },
        blueprint=blueprint,
        types=types,
    )


# ---------------------------------------------------------------------------
# Optional LLM refinement
# ---------------------------------------------------------------------------


def _sample_for_llm(questions: list[RawQuestion], max_per_type: int = 6) -> dict[str, list[dict[str, Any]]]:
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question in questions:
        if not _is_clean(question):
            continue
        if question.type in {
            QuestionType.SENTENCE_ORDERING,
            QuestionType.CLOZE,
            QuestionType.READING,
        }:
            continue
        if len(samples[question.type.value]) >= max_per_type:
            continue
        samples[question.type.value].append(
            {
                "id": question.raw_id,
                "type": question.type.value,
                "stem": question.stem[:400],
                "options": [opt.text for opt in question.options],
                "answer": question.answer_text,
                "explanation": question.explanation[:300],
                "source": question.source.kind,
            }
        )
    return dict(samples)


def _llm_enhance(
    settings: Settings,
    questions: list[RawQuestion],
    aggregates: dict[str, RuleAggregate],
    observed: Counter[str],
    stats: dict[str, Any],
    existing: MistakeRulesFile | None,
    offline: bool,
) -> tuple[dict[str, Any], list[str]]:
    notes: list[str] = []
    if offline or not settings.has_llm:
        notes.append("未配置/已禁用 LLM，使用纯确定性挖掘。")
        return {}, notes
    client = LLMClient(settings, offline=offline)
    existing_locked = [
        rule.id for rule in (existing.rules if existing else []) if rule.locked
    ]
    payload = {
        "task": "分析大二上日语题库的题型配比与设误规律，复核/补充规则。",
        "blueprint_2024": BLUEPRINT_2024,
        "observed_counts": dict(observed),
        "rule_aggregates": stats.get("rule_aggregates", {}),
        "samples": _sample_for_llm(questions),
        "allowed_rule_ids": sorted(RULE_DEFINITIONS.keys()),
        "locked_rule_ids": existing_locked,
        "constraints": [
            "只能输出 JSON，不要 Markdown。",
            "不得删除确定性高频规则；不得改写 locked 规则。",
            "checker 只能使用允许的 checker ID 或 null。",
            "新规则必须有证据；语义规则 checker 必须为 null。",
            "干扰项规则必须能解释为什么错。",
        ],
        "output_schema": {
            "rules": [
                {
                    "id": "string",
                    "name_zh": "string",
                    "description": "string",
                    "applies_to": ["question type"],
                    "severity": "high|medium|low",
                    "checker": "string|null",
                    "confidence": 0.0,
                    "evidence": [
                        {
                            "question_id": "string",
                            "correct": "string",
                            "distractor": "string",
                            "note": "string",
                        }
                    ],
                }
            ],
            "types": {
                "<type>": {
                    "correct_design": {
                        "strategy": "string",
                        "answer_source": "string",
                        "verification": "string",
                    },
                    "allowed_mistake_rules": ["rule_id"],
                    "prompt_hint": "string",
                }
            },
            "review_notes": ["string"],
        },
    }
    system = (
        "你是 JLPTCraft 的日语命题规则分析器。只输出严格 JSON。"
        "你的任务是复核确定性统计给出的设误规则，补充语义类规则，并完善各题型的设正策略。"
        "不得发明未在 allowed_rule_ids 中的规则 ID；如确需新增语义规则，id 使用新的英文 slug，checker 必须为 null。"
        "不得修改 locked_rule_ids。证据必须来自输入样本。"
    )
    try:
        response = client.complete_json(
            system,
            json.dumps(payload, ensure_ascii=False),
            temperature=0.1,
            max_tokens=12000,
        )
    except LLMError as exc:
        notes.append(f"LLM 增强失败，回退确定性规则：{exc}")
        return {}, notes
    if not isinstance(response, dict):
        notes.append("LLM 返回不是 JSON 对象，已忽略。")
        return {}, notes
    review_notes = response.get("review_notes") or []
    if isinstance(review_notes, list):
        notes.extend(str(note) for note in review_notes[:10])

    merged_rules: list[dict[str, Any]] = []
    for item in response.get("rules") or []:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        rid = str(item["id"]).strip()
        if not rid or rid in existing_locked:
            continue
        evidence = []
        for ev in item.get("evidence") or []:
            if isinstance(ev, dict):
                evidence.append(
                    RuleEvidence(
                        question_id=str(ev.get("question_id") or ""),
                        correct=str(ev.get("correct") or ""),
                        distractor=str(ev.get("distractor") or ""),
                        note=str(ev.get("note") or ""),
                        source_family="llm",
                    )
                )
        merged_rules.append(
            {
                "id": rid,
                "name_zh": str(item.get("name_zh") or rid),
                "description": str(item.get("description") or ""),
                "applies_to": [str(x) for x in (item.get("applies_to") or [])],
                "severity": str(item.get("severity") or "medium"),
                "checker": item.get("checker") if item.get("checker") in CHECKER_IDS else None,
                "confidence": float(item.get("confidence") or 0.5),
                "evidence": evidence,
            }
        )
    type_overrides = response.get("types") if isinstance(response.get("types"), dict) else {}
    return {"rules": merged_rules, "types": type_overrides}, notes


def _apply_llm(
    types: QuestionTypesFile,
    rules: list[MistakeRule],
    enhancement: dict[str, Any],
) -> tuple[QuestionTypesFile, list[MistakeRule]]:
    rule_map = {rule.id: rule for rule in rules}
    for item in enhancement.get("rules", []) or []:
        rid = item["id"]
        definition = rule_definition(rid)
        if rid in rule_map:
            existing = rule_map[rid]
            if existing.status in {RuleStatus.APPROVED, RuleStatus.LOCKED}:
                continue
            if not existing.description and item.get("description"):
                existing.description = item["description"]
            for ev in item.get("evidence", [])[:3]:
                if len(existing.evidence) < 5:
                    existing.evidence.append(ev)
            if item.get("confidence"):
                existing.confidence = max(existing.confidence, round(float(item["confidence"]), 3))
            continue
        # New semantic rule: keep only if it has evidence or explicit confidence.
        evidence = item.get("evidence", [])
        if not evidence:
            continue
        applies: list[QuestionType] = []
        for raw_type in item.get("applies_to", []):
            try:
                applies.append(QuestionType(raw_type))
            except ValueError:
                continue
        if not applies:
            applies = list(definition.applies_to) if definition else []
        if not applies:
            continue
        severity = {
            "high": RuleSeverity.HIGH,
            "medium": RuleSeverity.MEDIUM,
            "low": RuleSeverity.LOW,
        }.get(str(item.get("severity") or "medium"), RuleSeverity.MEDIUM)
        rule_map[rid] = MistakeRule(
            id=rid,
            name_zh=str(item.get("name_zh") or rid),
            description=str(item.get("description") or ""),
            applies_to=applies,
            severity=severity,
            checker=item.get("checker"),
            confidence=round(float(item.get("confidence") or 0.5), 3),
            evidence_count=len(evidence),
            evidence=evidence[:5],
            locked=False,
            status=RuleStatus.DRAFT,
            version=1,
        )
    for type_key, override in (enhancement.get("types") or {}).items():
        type_rule = types.types.get(type_key)
        if type_rule is None or not isinstance(override, dict):
            continue
        design = override.get("correct_design")
        if isinstance(design, dict):
            type_rule.correct_design = CorrectDesign(
                strategy=str(design.get("strategy") or type_rule.correct_design.strategy),
                answer_source=str(design.get("answer_source") or type_rule.correct_design.answer_source),
                verification=str(design.get("verification") or type_rule.correct_design.verification),
            )
        allowed = override.get("allowed_mistake_rules")
        if isinstance(allowed, list):
            valid = {rule_id for rule_id in allowed if rule_id in rule_map}
            type_rule.allowed_mistake_rules = sorted(
                set(type_rule.allowed_mistake_rules) | valid
            )
        if override.get("prompt_hint"):
            type_rule.prompt_hint = str(override["prompt_hint"])
    return types, list(rule_map.values())


# ---------------------------------------------------------------------------
# Merge with approved baseline
# ---------------------------------------------------------------------------


def _load_existing(settings: Settings) -> tuple[QuestionTypesFile | None, MistakeRulesFile | None]:
    types = None
    mistakes = None
    if settings.question_types_path.exists():
        try:
            types = QuestionTypesFile.model_validate(json.loads(settings.question_types_path.read_text(encoding="utf-8")))
        except Exception:
            types = None
    if settings.common_mistakes_path.exists():
        try:
            mistakes = MistakeRulesFile.model_validate(json.loads(settings.common_mistakes_path.read_text(encoding="utf-8")))
        except Exception:
            mistakes = None
    return types, mistakes


def _merge_existing(
    new_types: QuestionTypesFile,
    new_rules: list[MistakeRule],
    existing_types: QuestionTypesFile | None,
    existing_mistakes: MistakeRulesFile | None,
) -> tuple[QuestionTypesFile, list[MistakeRule], bool, list[dict[str, Any]]]:
    has_existing = bool(
        existing_types
        and existing_mistakes
        and existing_types.meta.get("reviewed")
        and existing_mistakes.meta.get("reviewed")
    )
    if not has_existing:
        return new_types, new_rules, False, []

    conflicts: list[dict[str, Any]] = []
    existing_map = {rule.id: rule for rule in existing_mistakes.rules}
    merged: list[MistakeRule] = []
    for rule in new_rules:
        old = existing_map.get(rule.id)
        if old and old.status in {RuleStatus.APPROVED, RuleStatus.LOCKED}:
            merged.append(old)
            if old.evidence_count != rule.evidence_count or old.description != rule.description:
                conflicts.append(
                    {
                        "rule_id": rule.id,
                        "kind": "changed_or_locked",
                        "kept": old.status.value,
                        "new_count": rule.evidence_count,
                    }
                )
        else:
            merged.append(rule)
    new_ids = {rule.id for rule in new_rules}
    for old in existing_mistakes.rules:
        if old.id not in new_ids:
            merged.append(old)
            conflicts.append({"rule_id": old.id, "kind": "removed_keep_approved"})

    # Keep approved type weights/design; update observed statistics.
    merged_types: dict[str, QuestionTypeRule] = {}
    for key, rule in existing_types.types.items():
        new_rule = new_types.types.get(key)
        if new_rule is None:
            merged_types[key] = rule
            continue
        if abs(rule.weight - new_rule.weight) > 1e-6:
            conflicts.append(
                {
                    "type": key,
                    "kind": "weight_changed_keep_approved",
                    "old": rule.weight,
                    "new": new_rule.weight,
                }
            )
        merged_types[key] = QuestionTypeRule(
            type=rule.type,
            name_zh=rule.name_zh,
            weight=rule.weight,
            observed_weight=new_rule.observed_weight,
            blueprint_count=rule.blueprint_count,
            supported=rule.supported,
            option_count=rule.option_count,
            correct_design=rule.correct_design,
            allowed_mistake_rules=sorted(
                set(rule.allowed_mistake_rules) | set(new_rule.allowed_mistake_rules)
            ),
            prompt_hint=rule.prompt_hint,
            evidence_count=new_rule.evidence_count,
        )
    for key, rule in new_types.types.items():
        if key not in merged_types:
            merged_types[key] = rule
    new_types.types = merged_types
    return new_types, merged, True, conflicts


# ---------------------------------------------------------------------------
# Draft / final persistence
# ---------------------------------------------------------------------------


def _write_draft(result: MiningResult) -> None:
    result.draft_types_path.parent.mkdir(parents=True, exist_ok=True)
    result.question_types.meta["reviewed"] = False
    result.question_types.meta["generated_at"] = now_iso()
    result.mistakes.meta["reviewed"] = False
    result.mistakes.meta["generated_at"] = now_iso()
    result.draft_types_path.write_text(
        json.dumps(result.question_types.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result.draft_mistakes_path.write_text(
        json.dumps(result.mistakes.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path = result.draft_types_path.parent / "_mining_report.draft.json"
    report_path.write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _auto_promotable(rule: MistakeRule) -> bool:
    families = {ev.source_family for ev in rule.evidence if ev.source_family}
    return (
        rule.confidence >= 0.8
        and rule.evidence_count >= 2
        and len(families) >= 2
        and rule.checker is not None
    )


def _backup_existing(settings: Settings) -> None:
    stamp = now_iso().replace(":", "").replace("+", "_").replace("-", "")
    target = settings.rules_history_dir / stamp
    target.mkdir(parents=True, exist_ok=True)
    for path in [settings.question_types_path, settings.common_mistakes_path, settings.mining_report_path]:
        if path.exists():
            shutil.copy2(path, target / path.name)


def _save_final(
    settings: Settings,
    result: MiningResult,
    *,
    promote_all_new: bool,
) -> None:
    _backup_existing(settings)
    version = int(result.question_types.meta.get("version", 1))
    approved_rules: list[MistakeRule] = []
    for rule in result.mistakes.rules:
        if rule.status in {RuleStatus.APPROVED, RuleStatus.LOCKED}:
            approved_rules.append(rule)
            continue
        if promote_all_new and rule.evidence_count >= 1:
            rule.status = RuleStatus.APPROVED
            approved_rules.append(rule)
        elif _auto_promotable(rule):
            rule.status = RuleStatus.APPROVED
            approved_rules.append(rule)
        else:
            rule.status = RuleStatus.DRAFT
            approved_rules.append(rule)

    result.question_types.meta.update(
        {
            "reviewed": True,
            "reviewed_at": now_iso(),
            "version": version,
            "generator": result.question_types.meta.get("generator", "deterministic"),
            "corpus_sha256": result.question_types.meta.get("corpus_sha256", ""),
            "conflicts": result.conflicts,
        }
    )
    result.mistakes.meta.update(
        {
            "reviewed": True,
            "reviewed_at": now_iso(),
            "version": version,
            "corpus_sha256": result.question_types.meta.get("corpus_sha256", ""),
            "conflicts": result.conflicts,
        }
    )
    result.question_types.types = result.question_types.types
    result.question_types.meta["approved_rule_ids"] = [
        rule.id for rule in approved_rules if rule.status in {RuleStatus.APPROVED, RuleStatus.LOCKED}
    ]
    result.question_types.meta["draft_rule_ids"] = [
        rule.id for rule in approved_rules if rule.status == RuleStatus.DRAFT
    ]
    settings.question_types_path.parent.mkdir(parents=True, exist_ok=True)
    settings.question_types_path.write_text(
        json.dumps(result.question_types.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    settings.common_mistakes_path.write_text(
        json.dumps(
            MistakeRulesFile(meta=result.mistakes.meta, rules=approved_rules).model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    settings.mining_report_path.write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Rendering / review
# ---------------------------------------------------------------------------


def _render_summary(result: MiningResult) -> None:
    report = result.report
    lines = [
        f"清洗后语料：{report.get('clean_questions', 0)} 题",
        f"题型分布：{report.get('observed_counts', {})}",
        f"规则数：{len(result.mistakes.rules)}；已隔离干扰项：{report.get('unclassified_distractors', 0)}",
    ]
    if result.llm_notes:
        lines.extend(f"LLM：{note}" for note in result.llm_notes[:5])
    if result.conflicts:
        lines.append(f"与已审核规则的冲突：{len(result.conflicts)} 项（已保留已审核版本）")

    if Console is None or Table is None:
        print("\n".join(lines))
        for rule in sorted(result.mistakes.rules, key=lambda r: -r.evidence_count)[:20]:
            print(f"  {rule.id}: {rule.name_zh} 次数={rule.evidence_count} checker={rule.checker}")
        return

    console = Console()
    console.print("[bold cyan]" + " | ".join(lines) + "[/bold cyan]")
    type_table = Table(title="题型配比 / 设正策略")
    type_table.add_column("题型", no_wrap=True)
    type_table.add_column("观察数", justify="right")
    type_table.add_column("2024题数", justify="right")
    type_table.add_column("权重", justify="right")
    type_table.add_column("生成", justify="center")
    type_table.add_column("设正来源")
    for key, rule in result.question_types.types.items():
        type_table.add_row(
            key,
            str(rule.evidence_count),
            str(rule.blueprint_count or 0),
            f"{rule.weight:.3f}",
            "是" if rule.supported else "记录",
            rule.correct_design.answer_source,
        )
    console.print(type_table)

    rule_table = Table(title="设误规则（按证据数）")
    rule_table.add_column("规则 ID", no_wrap=True)
    rule_table.add_column("名称")
    rule_table.add_column("次数", justify="right")
    rule_table.add_column("来源", justify="center")
    rule_table.add_column("checker", no_wrap=True)
    rule_table.add_column("状态", no_wrap=True)
    for rule in sorted(result.mistakes.rules, key=lambda r: -r.evidence_count):
        rule_table.add_row(
            rule.id,
            rule.name_zh,
            str(rule.evidence_count),
            str(len({ev.source_family for ev in rule.evidence})),
            rule.checker or "语义/审计",
            rule.status.value,
        )
    console.print(rule_table)
    if result.mistakes.rules:
        example = next((r for r in result.mistakes.rules if r.evidence), None)
        if example and example.evidence:
            ev = example.evidence[0]
            console.print(
                f"[dim]示例：{example.id} | {ev.correct} → {ev.distractor} | {ev.note} | {ev.question_id}[/dim]"
            )


def _edit_draft(settings: Settings, result: MiningResult) -> bool:
    _write_draft(result)
    editor = os.environ.get("EDITOR")
    if not editor:
        editor = "notepad" if os.name == "nt" else "vi"
    try:
        for path in [result.draft_mistakes_path, result.draft_types_path]:
            subprocess.run([editor, str(path)], check=False)
        mistakes = MistakeRulesFile.model_validate(
            json.loads(result.draft_mistakes_path.read_text(encoding="utf-8"))
        )
        types = QuestionTypesFile.model_validate(
            json.loads(result.draft_types_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        message = f"草稿编辑后 JSON 无效，保留原草稿：{type(exc).__name__}: {exc}"
        if Console:
            Console().print(f"[red]{message}[/red]")
        else:
            print(message)
        return False
    result.mistakes = mistakes
    result.question_types = types
    return True


def mine(settings: Settings, offline: bool = False) -> MiningResult:
    questions, source_status = _load_questions(settings)
    existing_types, existing_mistakes = _load_existing(settings)
    has_existing = bool(
        existing_types
        and existing_mistakes
        and existing_types.meta.get("reviewed")
        and existing_mistakes.meta.get("reviewed")
    )
    first_approval = not has_existing
    aggregates, observed, stats = _mine_deterministic(settings, questions, source_status)
    corpus_sha = _corpus_hash(settings)

    rules = _build_mistake_rules(aggregates, existing_mistakes if has_existing else None, first_approval)
    types = _build_question_types(
        aggregates, observed, existing_types if has_existing else None, corpus_sha, first_approval
    )

    enhancement, llm_notes = _llm_enhance(
        settings,
        questions,
        aggregates,
        observed,
        stats,
        existing_mistakes if has_existing else None,
        offline,
    )
    if enhancement:
        types, rules = _apply_llm(types, rules, enhancement)
        types.meta["generator"] = "deterministic+llm"

    types, rules, has_existing, conflicts = _merge_existing(
        types, rules, existing_types, existing_mistakes
    )
    first_approval = not has_existing

    mistakes = MistakeRulesFile(
        meta={
            "semester": settings.active_semester.value,
            "version": int(types.meta.get("version", 1)),
            "reviewed": False,
            "generated_at": now_iso(),
            "generator": types.meta.get("generator", "deterministic"),
            "corpus_sha256": corpus_sha,
            "notes": [
                "设误库：每个规则必须绑定 checker 或标记为语义规则（由 blind-solve 审计）。",
                "已审核/锁定规则不会被重新挖掘覆盖；冲突会写入 conflicts。",
            ],
        },
        rules=rules,
    )
    types.meta["conflicts"] = conflicts
    types.meta["llm_notes"] = llm_notes
    report = {
        **stats,
        "corpus_sha256": corpus_sha,
        "rule_count": len(rules),
        "conflicts": conflicts,
        "llm_notes": llm_notes,
        "existing_approved": has_existing,
        "first_approval": first_approval,
        "generated_at": now_iso(),
        "coverage": {
            "approved_lessons": _approved_lessons(settings),
            "drafts": _draft_files(settings),
        },
    }
    result = MiningResult(
        question_types=types,
        mistakes=mistakes,
        report=report,
        draft_types_path=settings.draft_rules_dir / "question_types.draft.json",
        draft_mistakes_path=settings.draft_rules_dir / "common_mistakes.draft.json",
        has_existing_approved=has_existing,
        first_approval=first_approval,
        conflicts=conflicts,
        llm_notes=llm_notes,
        aggregates=aggregates,
        observed_counts=observed,
        clean_questions=[q for q in questions if _is_clean(q)],
        stats=stats,
    )
    return result


def _approved_lessons(settings: Settings) -> list[int]:
    if not settings.vocab_path.exists():
        return []
    try:
        data = json.loads(settings.vocab_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    lessons = set()
    for lesson in data.get("lessons", []) or []:
        entries = lesson.get("vocab", []) or []
        if any(entry.get("reviewed", True) for entry in entries):
            try:
                lessons.add(int(lesson.get("lesson")))
            except Exception:
                pass
    return sorted(lessons)


def _draft_files(settings: Settings) -> list[str]:
    if not settings.drafts_dir.exists():
        return []
    return sorted(path.name for path in settings.drafts_dir.glob("*.draft.json"))


def run_mining_and_review(
    settings: Settings,
    force: bool = False,
    assume_yes: bool = False,
    offline: bool = False,
) -> bool:
    existing_types, existing_mistakes = _load_existing(settings)
    existing_ok = bool(
        existing_types
        and existing_mistakes
        and existing_types.meta.get("reviewed")
        and existing_mistakes.meta.get("reviewed")
    )
    corpus_sha = _corpus_hash(settings)
    if (
        not force
        and existing_ok
        and existing_types.meta.get("corpus_sha256") == corpus_sha
    ):
        message = "规则已是最新（语料 hash 未变）；跳过挖掘。使用 --remine 可重新挖掘并进入人工审核。"
        if Console:
            Console().print(f"[green]{message}[/green]")
        else:
            print(message)
        return True

    while True:
        result = mine(settings, offline=offline)
        _write_draft(result)
        _render_summary(result)
        prompt = (
            "规则已生成，请人工审核。确认后按 Enter 进入主菜单\n"
            "[Enter] 确认并保存 / e 编辑 draft / r 重新挖掘 / q 退出："
        )
        if assume_yes:
            print(prompt)
            print("[--yes] 自动确认规则。")
            _save_final(settings, result, promote_all_new=result.first_approval)
            return True
        try:
            choice = input(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return False
        if choice == "":
            _save_final(settings, result, promote_all_new=result.first_approval)
            return True
        if choice == "e":
            if _edit_draft(settings, result):
                _save_final(settings, result, promote_all_new=False)
                return True
            continue
        if choice == "r":
            continue
        if choice == "q":
            return False
        print("请输入 Enter / e / r / q。")
