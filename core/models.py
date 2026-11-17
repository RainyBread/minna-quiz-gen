"""Pydantic models for JLPTCraft's corpus, textbook, rules and generated batches."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from config import Semester


class QuestionType(str, Enum):
    KANJI_TO_KANA = "kanji_to_kana"
    KANA_TO_KANJI = "kana_to_kanji"
    VOCAB_CHOICE = "vocab_choice"
    EXPRESSION_CHOICE = "expression_choice"
    GRAMMAR_FILL = "grammar_fill"
    TRANSLATION_CHOICE = "translation_choice"
    SENTENCE_ORDERING = "sentence_ordering"
    CLOZE = "cloze"
    READING = "reading"

    @property
    def is_generatable(self) -> bool:
        return self in GENERATION_TYPES

    @property
    def is_reading_or_kanji(self) -> bool:
        return self in {QuestionType.KANJI_TO_KANA, QuestionType.KANA_TO_KANJI}

    @property
    def is_local(self) -> bool:
        """Reading/kanji questions are built locally without an LLM."""
        return self.is_reading_or_kanji


GENERATION_TYPES: tuple[QuestionType, ...] = (
    QuestionType.KANJI_TO_KANA,
    QuestionType.KANA_TO_KANJI,
    QuestionType.VOCAB_CHOICE,
    QuestionType.EXPRESSION_CHOICE,
    QuestionType.GRAMMAR_FILL,
)

RECORD_ONLY_TYPES: tuple[QuestionType, ...] = (
    QuestionType.TRANSLATION_CHOICE,
    QuestionType.SENTENCE_ORDERING,
    QuestionType.CLOZE,
    QuestionType.READING,
)

TYPE_LABELS_ZH: dict[str, str] = {
    QuestionType.KANJI_TO_KANA.value: "日文汉字选假名",
    QuestionType.KANA_TO_KANJI.value: "假名选日文汉字",
    QuestionType.VOCAB_CHOICE.value: "词汇选择",
    QuestionType.EXPRESSION_CHOICE.value: "表达选择",
    QuestionType.GRAMMAR_FILL.value: "语法填空",
    QuestionType.TRANSLATION_CHOICE.value: "释义选择（记录型）",
    QuestionType.SENTENCE_ORDERING.value: "句子排序（记录型）",
    QuestionType.CLOZE.value: "完形填空（记录型）",
    QuestionType.READING.value: "阅读理解（记录型）",
}

BLUEPRINT_2024: dict[str, int] = {
    QuestionType.KANJI_TO_KANA.value: 30,
    QuestionType.KANA_TO_KANJI.value: 10,
    QuestionType.VOCAB_CHOICE.value: 5,
    QuestionType.EXPRESSION_CHOICE.value: 10,
    QuestionType.GRAMMAR_FILL.value: 25,
    QuestionType.SENTENCE_ORDERING.value: 5,
    QuestionType.CLOZE.value: 5,
    QuestionType.READING.value: 10,
}

BLUEPRINT_JP1_NOTICE: dict[str, int] = {
    QuestionType.KANJI_TO_KANA.value: 20,
    QuestionType.KANA_TO_KANJI.value: 20,
    QuestionType.VOCAB_CHOICE.value: 15,  # 外来语 10 + 词汇/搭配 5
    QuestionType.GRAMMAR_FILL.value: 30,
    QuestionType.SENTENCE_ORDERING.value: 5,
    QuestionType.CLOZE.value: 5,
    QuestionType.READING.value: 5,
}


class RuleStatus(str, Enum):
    CANDIDATE = "candidate"
    DRAFT = "draft"
    APPROVED = "approved"
    LOCKED = "locked"
    DEPRECATED = "deprecated"


class RuleSeverity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QualityStatus(str, Enum):
    IMPORTED = "imported"
    QUARANTINED = "quarantined"
    REFERENCE_ONLY = "reference_only"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class OptionRole(str, Enum):
    CORRECT = "correct"
    DISTRACTOR = "distractor"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=False)


class LenientModel(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=False)


# ---------------------------------------------------------------------------
# Corpus / raw questions
# ---------------------------------------------------------------------------


class SourceRef(LenientModel):
    source_id: str = ""
    file: str = ""
    page: int | None = None
    section: str | None = None
    question_no: int | None = None
    content_hash: str = ""
    legacy_id: str | None = None
    status: str | None = None
    answer_provenance: str | None = None
    kind: str | None = None
    target_has_kanji: bool | None = None
    type_basis: str = ""
    exam_no: int | None = None


class RawFlags(LenientModel):
    ocr_unclear: list[str] = Field(default_factory=list)
    ocr_uncertain: bool = False
    missing_option_keys: list[str] = Field(default_factory=list)
    duplicate_option_texts: list[str] = Field(default_factory=list)
    multi_blank_count: int = 0
    answer_conflict: bool = False
    answer_key_mismatch: bool = False
    script_mismatch: bool = False
    non_japanese_options: list[str] = Field(default_factory=list)
    duplicate_of: str | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def has_blocking_issues(self) -> bool:
        return bool(
            self.ocr_unclear
            or self.ocr_uncertain
            or self.missing_option_keys
            or self.duplicate_option_texts
            or self.answer_conflict
            or self.answer_key_mismatch
            or self.script_mismatch
            or self.non_japanese_options
        )


class RawOption(LenientModel):
    key: str
    text: str


class RawQuestion(LenientModel):
    raw_id: str
    type: QuestionType
    lesson: int | None = None
    textbook_lesson: int | None = None
    stem: str = ""
    target: str | None = None
    options: list[RawOption] = Field(default_factory=list)
    answer_key: str | None = None
    answer_text: str | None = None
    explanation: str = ""
    translation: str = ""
    passage: str | None = None
    source: SourceRef = Field(default_factory=SourceRef)
    flags: RawFlags = Field(default_factory=RawFlags)
    quality: dict[str, Any] = Field(default_factory=dict)
    status: QualityStatus = QualityStatus.IMPORTED

    @property
    def option_map(self) -> dict[str, str]:
        return {opt.key.upper(): opt.text for opt in self.options}


class RawBankMeta(LenientModel):
    semester: str = Semester.BIG2_AUTUMN.value
    source: str = ""
    textbook: str = ""
    lesson_range: list[int] = Field(default_factory=lambda: [37, 50])
    quality: dict[str, Any] = Field(default_factory=dict)
    generated_from: list[str] = Field(default_factory=list)
    source_sha256: str = ""
    generated_at: str = ""
    notes: list[str] = Field(default_factory=list)


class RawBank(LenientModel):
    meta: RawBankMeta = Field(default_factory=RawBankMeta)
    questions: list[RawQuestion] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Textbook
# ---------------------------------------------------------------------------


class VocabEntry(LenientModel):
    entry_id: str
    lesson: int
    course_lesson: int | None = None
    written: str
    kana: str
    translation_zh: str = ""
    pos: str = ""
    notes: str = ""
    confidence: str = "medium"
    source: dict[str, Any] = Field(default_factory=dict)
    has_kanji: bool = False
    is_loanword: bool = False
    unidic_verified: bool | None = None
    unidic_reading: str = ""
    unidic_pos: str = ""
    reading_source: str = ""
    verification_note: str = ""
    uncertainty: str = ""
    reviewed: bool = True

    @property
    def surface_key(self) -> str:
        return f"{self.written}\u0000{self.kana}"


class VocabLesson(LenientModel):
    lesson: int  # textbook absolute lesson, 37-50
    course_lesson: int | None = None
    title: str | None = None
    vocab: list[VocabEntry] = Field(default_factory=list)


class TextbookVocab(LenientModel):
    meta: dict[str, Any] = Field(default_factory=dict)
    lessons: list[VocabLesson] = Field(default_factory=list)

    def all_entries(self) -> list[VocabEntry]:
        return [entry for lesson in self.lessons for entry in lesson.vocab]

    def entries_for_lesson(self, textbook_lesson: int) -> list[VocabEntry]:
        for lesson in self.lessons:
            if lesson.lesson == textbook_lesson:
                return list(lesson.vocab)
        return []

    def approved_lessons(self) -> list[int]:
        return sorted(
            {
                entry.lesson
                for entry in self.all_entries()
                if entry.reviewed
            }
        )


class GrammarPoint(LenientModel):
    gp_id: str
    lesson: int | None = None
    pattern_ja: str = ""
    meaning_zh: str = ""
    formation: str = ""
    notes: str = ""
    examples: list[str] = Field(default_factory=list)
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    approved_answer_forms: list[str] = Field(default_factory=list)
    allowed_distractor_rule_ids: list[str] = Field(default_factory=list)
    form_tags: list[str] = Field(default_factory=list)
    forms_source: str = ""
    forms_reviewed_at: str = ""
    confidence: str = "low"
    status: RuleStatus = RuleStatus.DRAFT
    reviewed: bool = False


class GrammarFile(LenientModel):
    meta: dict[str, Any] = Field(default_factory=dict)
    points: list[GrammarPoint] = Field(default_factory=list)

    def approved_points(self) -> list[GrammarPoint]:
        return [
            p
            for p in self.points
            if p.reviewed and p.status in {RuleStatus.APPROVED, RuleStatus.LOCKED}
        ]

    def points_for_lesson(self, textbook_lesson: int) -> list[GrammarPoint]:
        return [
            p
            for p in self.approved_points()
            if p.lesson is None or p.lesson == textbook_lesson
        ]


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


class RuleEvidence(LenientModel):
    source_id: str = ""
    question_id: str = ""
    target: str = ""
    correct: str = ""
    distractor: str = ""
    note: str = ""
    source_family: str = ""
    ocr_quality: str = ""


class MistakeRule(StrictModel):
    id: str
    name_zh: str
    description: str = ""
    applies_to: list[QuestionType] = Field(default_factory=list)
    severity: RuleSeverity = RuleSeverity.MEDIUM
    checker: str | None = None
    confidence: float = 0.0
    evidence_count: int = 0
    evidence: list[RuleEvidence] = Field(default_factory=list)
    locked: bool = False
    status: RuleStatus = RuleStatus.DRAFT
    version: int = 1


class MistakeRulesFile(StrictModel):
    meta: dict[str, Any] = Field(default_factory=dict)
    rules: list[MistakeRule] = Field(default_factory=list)

    def rule_map(self) -> dict[str, MistakeRule]:
        return {rule.id: rule for rule in self.rules}

    def approved_rules(self) -> list[MistakeRule]:
        return [
            rule
            for rule in self.rules
            if rule.status in {RuleStatus.APPROVED, RuleStatus.LOCKED}
        ]


class CorrectDesign(StrictModel):
    strategy: str
    answer_source: str
    verification: str


class QuestionTypeRule(StrictModel):
    type: QuestionType
    name_zh: str
    weight: float = 0.0
    observed_weight: float | None = None
    blueprint_count: int | None = None
    supported: bool = True
    option_count: int = 4
    correct_design: CorrectDesign
    allowed_mistake_rules: list[str] = Field(default_factory=list)
    prompt_hint: str = ""
    evidence_count: int = 0


class Blueprint(StrictModel):
    source: str = ""
    total_questions: int = 0
    counts: dict[str, int] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)
    reference_blueprints: list[dict[str, Any]] = Field(default_factory=list)
    confidence: str = "high"
    note: str = ""


class QuestionTypesFile(StrictModel):
    meta: dict[str, Any] = Field(default_factory=dict)
    blueprint: Blueprint = Field(default_factory=Blueprint)
    types: dict[str, QuestionTypeRule] = Field(default_factory=dict)

    def supported_types(self) -> dict[str, QuestionTypeRule]:
        return {key: rule for key, rule in self.types.items() if rule.supported}


# ---------------------------------------------------------------------------
# Generation / validation
# ---------------------------------------------------------------------------


class QuizRequest(LenientModel):
    raw: str
    count: int = 3
    lessons: list[int] = Field(default_factory=list)  # textbook absolute lessons
    types: list[QuestionType] = Field(default_factory=list)
    type_plan: dict[str, int] = Field(default_factory=dict)
    difficulty: str | None = None
    seed: int | None = None

    @property
    def course_lessons(self) -> list[int]:
        return [lesson - 36 for lesson in self.lessons]


class GeneratedTarget(StrictModel):
    surface: str
    written: str = ""
    reading: str = ""
    meaning_zh: str = ""
    vocab_id: str | None = None
    grammar_id: str | None = None


class GeneratedOption(StrictModel):
    key: str
    text: str
    role: OptionRole
    mistake_rule_id: str | None = None
    wrongness_zh: str | None = None
    checker_evidence: dict[str, Any] = Field(default_factory=dict)


class GeneratedQuestion(StrictModel):
    schema_version: str = "1.0"
    question_id: str
    batch_id: str = ""
    semester: Semester = Semester.BIG2_AUTUMN
    type: QuestionType
    lesson: int = 37
    textbook_lesson: int = 37
    target: GeneratedTarget
    stem: str
    options: list[GeneratedOption] = Field(default_factory=list)
    answer_key: str
    correct_option_text: str = ""
    explanation_zh: str = ""
    translation_zh: str = ""
    difficulty: int = 3
    source_refs: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    scope: str = "in_syllabus"
    topic: str = ""
    source_kind: str = "new"
    point_key: str = ""
    semantic_audit: dict[str, Any] | None = None


class BatchMeta(StrictModel):
    id: str
    semester: Semester = Semester.BIG2_AUTUMN
    request: str = ""
    mode: str = "practice"
    scope: str = "in_syllabus"
    topic: str = ""
    type_plan: dict[str, int] = Field(default_factory=dict)
    lessons: list[int] = Field(default_factory=list)
    model: str = ""
    rules_version: int = 1
    ruleset_hash: str = ""
    created_at: str = ""
    warnings: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    expected: int = 0
    accepted: int = 0
    metrics: dict[str, Any] = Field(default_factory=dict)


class GeneratedBatch(StrictModel):
    batch_meta: BatchMeta
    questions: list[GeneratedQuestion] = Field(default_factory=list)


class AuditResult(LenientModel):
    question_id: str
    chosen_key: str | None = None
    multiple_valid: bool = False
    defensible_options: list[str] = Field(default_factory=list)
    reason: str = ""
    rule_confirmations: dict[str, bool] = Field(default_factory=dict)


class ValidationIssue(StrictModel):
    check_id: str
    severity: Severity
    question_id: str = ""
    field: str = ""
    message: str
    rule_id: str | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)
    blocking: bool = True


class ValidationReport(StrictModel):
    ok: bool = True
    issues: list[ValidationIssue] = Field(default_factory=list)
    per_question: dict[str, list[str]] = Field(default_factory=dict)
    audit: dict[str, Any] = Field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]


class BatchManifest(LenientModel):
    batch_id: str
    semester: str
    created_at: str = ""
    request: dict[str, Any] = Field(default_factory=dict)
    ruleset_hash: str = ""
    model: str = ""
    prompt_version: str = ""
    expected: int = 0
    accepted: int = 0
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    export_paths: list[str] = Field(default_factory=list)
    db_ids: list[str] = Field(default_factory=list)
    fingerprints: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KANJI_RE = re.compile(r"[㐀-䶿一-鿿]")
_HIRAGANA_RE = re.compile(r"[぀-ゟ]")
_KATAKANA_RE = re.compile(r"[゠-ヿ]")
_WHITESPACE_RE = re.compile(r"\s+")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = value.replace("　", " ")
    return _WHITESPACE_RE.sub(" ", value).strip()


def normalize_key(text: str) -> str:
    return normalize_text(text).casefold()


def to_hiragana(text: str) -> str:
    chars: list[str] = []
    for char in text or "":
        code = ord(char)
        if 0x30A1 <= code <= 0x30F6:
            chars.append(chr(code - 0x60))
        else:
            chars.append(char)
    return "".join(chars)


def contains_kanji(text: str) -> bool:
    return bool(_KANJI_RE.search(text or ""))


def contains_hiragana(text: str) -> bool:
    return bool(_HIRAGANA_RE.search(text or ""))


def contains_katakana(text: str) -> bool:
    return bool(_KATAKANA_RE.search(text or ""))


def is_kana_only(text: str) -> bool:
    value = normalize_text(text)
    if not value:
        return False
    return all(
        _HIRAGANA_RE.match(ch) or _KATAKANA_RE.match(ch) or ch == "ー"
        for ch in value
    )


def is_katakana_only(text: str) -> bool:
    value = normalize_text(text)
    if not value:
        return False
    return all(_KATAKANA_RE.match(ch) or ch == "ー" for ch in value)


def is_reading_text(text: str) -> bool:
    """True for kana readings that may contain counter/prefix punctuation.

    Real textbook options include forms such as ``～じゅう``, ``－ほん`` and
    ``はかります/はかります``.  They are still readings, not kanji choices.
    """
    value = normalize_text(text)
    if not value:
        return False
    allowed_punct = set("~～〜－-ー/・（）()「」【】[] 　。、，,！？!?")
    return all(
        _HIRAGANA_RE.match(ch) or _KATAKANA_RE.match(ch) or ch in allowed_punct
        for ch in value
    )


def has_japanese(text: str) -> bool:
    return contains_kanji(text) or contains_hiragana(text) or contains_katakana(text)


def resolve_semester(value: str | Semester) -> Semester:
    if isinstance(value, Semester):
        return value
    return Semester.parse(value)
