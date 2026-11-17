"""Natural-language intent -> quiz plan.

Local regex parsing is the fast path.  A lightweight tag-based LLM call is only
used when the local parser cannot identify a question type and the user did not
ask for a mock exam.  Every result is clamped/validated locally.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

from core.context import AppContext
from core.models import (
    BLUEPRINT_2024,
    GENERATION_TYPES,
    QuestionType,
    normalize_text,
)
from core.tagformat import first_tag, parse_tags

MAX_COUNT = 200
DEFAULT_COUNT = 3

TYPE_ALIASES: dict[str, QuestionType] = {
    "日文汉字选假名": QuestionType.KANJI_TO_KANA,
    "汉字选假名": QuestionType.KANJI_TO_KANA,
    "汉字读音": QuestionType.KANJI_TO_KANA,
    "读音": QuestionType.KANJI_TO_KANA,
    "假名选日文汉字": QuestionType.KANA_TO_KANJI,
    "假名选汉字": QuestionType.KANA_TO_KANJI,
    "写汉字": QuestionType.KANA_TO_KANJI,
    "汉字题": QuestionType.KANA_TO_KANJI,
    "词汇": QuestionType.VOCAB_CHOICE,
    "词语": QuestionType.VOCAB_CHOICE,
    "单词": QuestionType.VOCAB_CHOICE,
    "外来语": QuestionType.VOCAB_CHOICE,
    "表达": QuestionType.EXPRESSION_CHOICE,
    "会话": QuestionType.EXPRESSION_CHOICE,
    "应答": QuestionType.EXPRESSION_CHOICE,
    "交际": QuestionType.EXPRESSION_CHOICE,
    "语法": QuestionType.GRAMMAR_FILL,
    "助词": QuestionType.GRAMMAR_FILL,
    "填空": QuestionType.GRAMMAR_FILL,
}

UNSUPPORTED_ALIASES: dict[str, QuestionType] = {
    "排序": QuestionType.SENTENCE_ORDERING,
    "排列": QuestionType.SENTENCE_ORDERING,
    "完形": QuestionType.CLOZE,
    "完型": QuestionType.CLOZE,
    "阅读": QuestionType.READING,
    "読解": QuestionType.READING,
    "读解": QuestionType.READING,
}

MOCK_KEYWORDS = (
    "模拟卷",
    "模拟试卷",
    "一套卷",
    "整套卷",
    "期末卷",
    "期末模拟",
    "模拟考试",
    "考卷",
    "试卷",
    "一份卷",
    "一张卷",
    "出卷",
    "组卷",
    "试题卷",
    "期末考试卷",
)
SCOPE_STRICT = ("不超纲", "课内", "大纲内", "教材内", "只用教材")
SCOPE_DISTRACTOR = ("拓展干扰", "拓展选项", "扩展干扰")
SCOPE_EXTEND = ("拓展", "课外", "超纲", "扩展学习", "自愿扩展", "教材外")

USE_CASE_KEYWORDS = {
    "unit_test": ("单元测", "单元测试", "单元考", "周测", "小测"),
    "review": ("错题", "复习", "重做", "巩固复习"),
    "first_learn": ("首学", "初学", "第一次学", "新学", "巩固"),
}
DEFAULT_SOURCE_MIX = {
    "first_learn": {"original": 0.5, "variant": 0.3, "new": 0.2},
    "unit_test": {"original": 0.2, "variant": 0.3, "new": 0.5},
    "mock": {"original": 0.05, "variant": 0.2, "new": 0.75},
    "review": {"original": 0.0, "variant": 0.4, "new": 0.6},
}

_CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


class IntentError(RuntimeError):
    pass


class UnsupportedTypeError(IntentError):
    pass


@dataclass
class QuizPlan:
    raw: str
    mode: str = "practice"  # practice | mock
    scope: str = "in_syllabus"  # in_syllabus | mdict_distractors | extend
    topic: str = ""
    count: int = DEFAULT_COUNT
    lessons: list[int] = field(default_factory=list)
    type_plan: dict[str, int] = field(default_factory=dict)
    seed: int | None = None
    source: str = "local"  # local | llm | default
    use_case: str = "first_learn"  # first_learn | unit_test | mock | review
    source_mix: dict[str, float] = field(default_factory=dict)
    explicit_count: bool = False
    explicit_lessons: bool = False
    explicit_types: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.type_plan.values())

    def type_summary(self) -> str:
        labels = {
            QuestionType.KANJI_TO_KANA.value: "读音",
            QuestionType.KANA_TO_KANJI.value: "汉字",
            QuestionType.VOCAB_CHOICE.value: "词汇",
            QuestionType.EXPRESSION_CHOICE.value: "表达",
            QuestionType.GRAMMAR_FILL.value: "语法",
        }
        return " + ".join(
            f"{labels.get(key, key)}×{value}" for key, value in self.type_plan.items()
        )

    def scope_label(self) -> str:
        return {
            "in_syllabus": "课内",
            "mdict_distractors": "课内 + 拓展干扰项",
            "extend": "拓展学习",
        }.get(self.scope, self.scope)


def _normalize(text: str) -> str:
    return normalize_text(text or "")


def _cn_number(text: str) -> int | None:
    text = (text or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if text == "十":
        return 10
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        return tens * 10 + ones
    if len(text) == 1:
        return _CN_DIGITS.get(text)
    return None


def _count_from_text(text: str) -> int | None:
    match = re.search(r"(\d+)\s*(?:道|题|題|个|個)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"([零一二两三四五六七八九十]+)\s*(?:道|题|題|个|個)", text)
    if match:
        return _cn_number(match.group(1))
    return None


def _expand_lessons(start: int, end: int, warnings: list[str]) -> list[int]:
    if start > end:
        start, end = end, start
    values: list[int] = []
    for number in range(start, end + 1):
        if 37 <= number <= 50:
            values.append(number)
        elif 1 <= number <= 14:
            values.append(number + 36)
        else:
            warnings.append(f"课次 {number} 超出大二上范围，已忽略。")
    return values


def _lessons_from_text(text: str, warnings: list[str]) -> list[int]:
    range_match = re.search(
        r"第?\s*(\d+)\s*[-–~〜到至]\s*(\d+)\s*[课課]", text
    )
    if range_match:
        return _expand_lessons(
            int(range_match.group(1)), int(range_match.group(2)), warnings
        )
    singles = [
        int(match.group(1))
        for match in re.finditer(r"第?\s*(\d+)\s*[课課]", text)
    ]
    lessons: list[int] = []
    for number in singles:
        lessons.extend(_expand_lessons(number, number, warnings))
    return sorted(set(lessons))


def _topic_from_text(text: str) -> str:
    patterns = [
        r"围绕\s*([^\s，,。；;]+?)\s*(?:生成|出|来|做|的|主题|$)",
        r"关于\s*([^\s，,。；;]+?)\s*(?:生成|出|来|做|的|主题|$)",
        r"以\s*([^\s，,。；;]+?)\s*为主题",
        r"主题\s*[:：]\s*([^\s，,。；;]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            topic = match.group(1).strip().rstrip("的")
            if topic:
                return topic
    return ""


def _scope_from_text(text: str) -> str | None:
    if any(word in text for word in SCOPE_STRICT):
        return "in_syllabus"
    if any(word in text for word in SCOPE_DISTRACTOR):
        return "mdict_distractors"
    if any(word in text for word in SCOPE_EXTEND):
        return "extend"
    return None


def _supported_aliases(text: str) -> list[QuestionType]:
    found: list[tuple[int, int, QuestionType]] = []
    for alias, qtype in TYPE_ALIASES.items():
        index = text.find(alias)
        if index >= 0:
            found.append((index, -len(alias), qtype))
    # Longer aliases at the same position win ("假名选汉字" over "汉字").
    found.sort(key=lambda item: (item[0], item[1]))
    result: list[QuestionType] = []
    for _, _, qtype in found:
        if qtype not in result:
            result.append(qtype)
    return result


def _per_type_counts(text: str) -> dict[str, int]:
    plan: dict[str, int] = {}
    for alias, qtype in sorted(TYPE_ALIASES.items(), key=lambda item: -len(item[0])):
        patterns = [
            rf"{re.escape(alias)}\s*(\d+)\s*(?:道|题|題|个|個)",
            rf"(\d+)\s*(?:道|题|題|个|個)\s*(?:第?\s*\d+\s*[课課]\s*)?{re.escape(alias)}",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                plan[qtype.value] = max(0, int(match.group(1)))
                break
    return plan


def _allocate(count: int, types: list[QuestionType]) -> dict[str, int]:
    if not types:
        return {}
    unique = list(dict.fromkeys(types))
    if count < len(unique):
        unique = unique[:count]
    base = count // len(unique)
    remainder = count % len(unique)
    plan: dict[str, int] = {}
    for index, qtype in enumerate(unique):
        amount = base + (1 if index < remainder else 0)
        if amount:
            plan[qtype.value] = amount
    return plan


def _weighted_plan(count: int, context: AppContext) -> dict[str, int]:
    types = context.question_types.supported_types()
    weighted = [
        (key, rule.weight)
        for key, rule in types.items()
        if rule.weight > 0 and key in {t.value for t in GENERATION_TYPES}
    ]
    if not weighted:
        weighted = [
            (qtype.value, float(BLUEPRINT_2024.get(qtype.value, 1)))
            for qtype in GENERATION_TYPES
        ]
    total = sum(weight for _, weight in weighted) or 1.0
    quotas = [(key, count * weight / total) for key, weight in weighted]
    plan = {key: int(math.floor(quota)) for key, quota in quotas}
    remainder = count - sum(plan.values())
    ranked = sorted(
        quotas,
        key=lambda item: (item[1] - math.floor(item[1]), item[1]),
        reverse=True,
    )
    for key, _ in ranked[: max(0, remainder)]:
        plan[key] = plan.get(key, 0) + 1
    return {key: value for key, value in plan.items() if value > 0}


def _mock_plan(context: AppContext, count: int | None = None) -> dict[str, int]:
    course = context.course or {}
    mock = course.get("mock_blueprint", {}) if isinstance(course, dict) else {}
    counts = mock.get("counts") if isinstance(mock, dict) else None
    if not isinstance(counts, dict) or not counts:
        counts = {
            QuestionType.KANJI_TO_KANA.value: 30,
            QuestionType.KANA_TO_KANJI.value: 10,
            QuestionType.VOCAB_CHOICE.value: 5,
            QuestionType.EXPRESSION_CHOICE.value: 10,
            QuestionType.GRAMMAR_FILL.value: 25,
        }
    allowed = {qtype.value for qtype in GENERATION_TYPES}
    plan = {
        key: int(value)
        for key, value in counts.items()
        if key in allowed and int(value) > 0
    }
    base = sum(plan.values())
    if count and count > 0 and count != base:
        plan = _scale_counts(plan, count)
    return plan


def _scale_counts(counts: dict[str, int], target: int) -> dict[str, int]:
    """Scale a type plan proportionally to an exact target, largest remainder."""
    base = sum(counts.values()) or 1
    quotas = [(key, target * value / base) for key, value in counts.items()]
    result = {key: int(math.floor(quota)) for key, quota in quotas}
    remainder = target - sum(result.values())
    ranked = sorted(
        quotas,
        key=lambda item: (item[1] - math.floor(item[1]), item[1]),
        reverse=True,
    )
    for key, _ in ranked[: max(0, remainder)]:
        result[key] = result.get(key, 0) + 1
    return {key: value for key, value in result.items() if value > 0}


def _detect_use_case(text: str, is_mock: bool) -> str:
    if is_mock:
        return "mock"
    for use_case, keywords in USE_CASE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return use_case
    return "first_learn"


def _source_mix(context: AppContext, use_case: str) -> dict[str, float]:
    course = context.course or {}
    presets = course.get("source_mix_presets", {}) if isinstance(course, dict) else {}
    raw = presets.get(use_case) if isinstance(presets, dict) else None
    if isinstance(raw, dict) and raw:
        values = {
            key: float(raw.get(key, 0) or 0)
            for key in ("original", "variant", "new")
        }
        total = sum(values.values())
        if total > 0:
            return {key: value / total for key, value in values.items()}
    return dict(DEFAULT_SOURCE_MIX.get(use_case, DEFAULT_SOURCE_MIX["first_learn"]))


def _llm_plan(
    text: str,
    context: AppContext,
    llm: Any,
    warnings: list[str],
) -> dict[str, Any]:
    system = (
        "你是 JLPTCraft 的意图解析器。请分析用户需求，并用标签输出可执行计划。"
        "题型从 kanji_to_kana、kana_to_kanji、vocab_choice、expression_choice、grammar_fill 中选择；"
        "范围从 in_syllabus、mdict_distractors、extend 中选择。"
        "课次 37-50 为教材绝对课次，1-14 为课程课次（教材=+36）。"
    )
    user = (
        f"用户需求：{text}\n"
        "请输出：\n"
        "[MODE]practice 或 mock[/MODE]\n"
        "[COUNT]题量整数[/COUNT]\n"
        "[LESSONS]教材课次，如 37-50 或 3 或 37,39[/LESSONS]\n"
        "[TYPES]题型=数量，如 kanji_to_kana=5,grammar_fill=5[/TYPES]\n"
        "[SCOPE]in_syllabus、mdict_distractors 或 extend[/SCOPE]\n"
        "[TOPIC]主题，可留空[/TOPIC]"
    )
    try:
        raw = llm.complete_text(system, user, temperature=0.0, max_tokens=300)
    except Exception as exc:
        warnings.append(f"LLM 意图解析失败，使用本地默认：{type(exc).__name__}")
        return {}
    tags = parse_tags(raw)
    result: dict[str, Any] = {}
    mode = first_tag(tags, ("MODE",)).strip().lower()
    if mode in {"mock", "practice"}:
        result["mode"] = mode
    count_text = first_tag(tags, ("COUNT",))
    count = _cn_number(count_text) if count_text else None
    if count:
        result["count"] = max(1, min(MAX_COUNT, count))
    lessons_text = first_tag(tags, ("LESSONS", "LESSON"))
    if lessons_text:
        parsed = _lessons_from_text(lessons_text, warnings)
        if parsed:
            result["lessons"] = parsed
    types_text = first_tag(tags, ("TYPES", "TYPE_PLAN"))
    if types_text:
        plan: dict[str, int] = {}
        for part in re.split(r"[,，;；\s]+", types_text):
            if not part:
                continue
            if "=" in part or ":" in part:
                key, value = re.split(r"[=:]", part, maxsplit=1)
                key = key.strip()
                if key in TYPE_ALIASES:
                    key = TYPE_ALIASES[key].value
                try:
                    amount = int(value.strip())
                except ValueError:
                    continue
                if key in {q.value for q in GENERATION_TYPES} and amount > 0:
                    plan[key] = amount
        if plan:
            result["type_plan"] = plan
    scope = first_tag(tags, ("SCOPE",)).strip().lower()
    if scope in {"in_syllabus", "mdict_distractors", "extend"}:
        result["scope"] = scope
    topic = first_tag(tags, ("TOPIC",)).strip()
    if topic:
        result["topic"] = topic
    return result


def parse_plan(
    text: str,
    context: AppContext,
    *,
    llm: Any | None = None,
    offline: bool = False,
    scope_override: str | None = None,
    topic_override: str | None = None,
    force_mock: bool = False,
    seed: int | None = None,
    use_case_override: str | None = None,
) -> QuizPlan:
    raw = (text or "").strip()
    normalized = _normalize(raw)
    warnings: list[str] = []
    if not normalized:
        raise IntentError("请输入需求，例如：生成一套模拟卷 / 生成3道第44课读音题")

    is_mock = force_mock or any(keyword in normalized for keyword in MOCK_KEYWORDS)
    explicit_count_value = _count_from_text(normalized)
    explicit_count = explicit_count_value is not None
    lessons = _lessons_from_text(normalized, warnings)
    explicit_lessons = bool(lessons)
    topic = topic_override or _topic_from_text(normalized)
    scope = scope_override or _scope_from_text(normalized)

    unsupported = [
        qtype for alias, qtype in UNSUPPORTED_ALIASES.items() if alias in normalized
    ]
    if unsupported and not is_mock:
        labels = {
            QuestionType.SENTENCE_ORDERING: "排序",
            QuestionType.CLOZE: "完形",
            QuestionType.READING: "阅读",
        }
        names = sorted({labels.get(qtype, qtype.value) for qtype in unsupported})
        raise UnsupportedTypeError(
            "本期不生成：" + "、".join(names) + "。可用题型：读音/汉字/词汇/表达/语法。"
        )
    if unsupported and is_mock:
        warnings.append("模拟卷按 80 题五类生成；阅读/排序/完形本期不生成。")

    types = _supported_aliases(normalized)
    per_type = _per_type_counts(normalized)
    explicit_types = bool(types or per_type)
    if per_type:
        type_plan = per_type
        source = "local"
    elif types:
        count = explicit_count_value or DEFAULT_COUNT
        type_plan = _allocate(max(1, min(MAX_COUNT, count)), types)
        source = "local"
    elif is_mock:
        type_plan = _mock_plan(context)
        source = "local"
    else:
        type_plan = {}
        source = "default"
        extension_default = scope == "extend" or (
            bool(topic) and scope is None and not any(
                word in normalized for word in SCOPE_STRICT
            )
        )
        if extension_default:
            # Topic/extension practice is vocabulary-first by default.
            type_plan = {
                QuestionType.VOCAB_CHOICE.value: explicit_count_value or DEFAULT_COUNT
            }
        # Lightweight LLM fallback for free-form requests without a recognizable type.
        elif llm is not None and not offline and getattr(llm, "available", False):
            parsed = _llm_plan(raw, context, llm, warnings)
            if parsed:
                source = "llm"
                if parsed.get("mode") == "mock":
                    is_mock = True
                if parsed.get("count"):
                    explicit_count_value = parsed["count"]
                    explicit_count = True
                if parsed.get("lessons"):
                    lessons = parsed["lessons"]
                    explicit_lessons = True
                if parsed.get("type_plan"):
                    type_plan = parsed["type_plan"]
                    explicit_types = True
                if parsed.get("scope"):
                    scope = scope or parsed["scope"]
                if parsed.get("topic"):
                    topic = topic or parsed["topic"]
                if parsed.get("mode") == "mock":
                    type_plan = _mock_plan(context)
        if not type_plan:
            type_plan = _weighted_plan(explicit_count_value or DEFAULT_COUNT, context)

    if is_mock:
        if not explicit_types:
            type_plan = _mock_plan(
                context, explicit_count_value if explicit_count else None
            )
        count = sum(type_plan.values())
        if count > MAX_COUNT:
            warnings.append(f"题量上限 {MAX_COUNT}，已从 {count} 截断。")
            type_plan = _scale_counts(type_plan, MAX_COUNT)
            count = MAX_COUNT
        if not explicit_lessons:
            lessons = list(range(37, 51))
        scope = "in_syllabus"
        source = "local"
        warnings.append(
            f"考卷：{count} 题五类；不含阅读/排序/完形。"
        )
    else:
        count = sum(type_plan.values())
        if count <= 0:
            raise IntentError("题量必须大于 0。")
        if count > MAX_COUNT:
            warnings.append(f"题量上限 {MAX_COUNT}，已从 {count} 截断。")
            count = MAX_COUNT
            # Trim the plan deterministically.
            trimmed: dict[str, int] = {}
            remaining = count
            for key, value in type_plan.items():
                if remaining <= 0:
                    break
                take = min(value, remaining)
                trimmed[key] = take
                remaining -= take
            type_plan = trimmed
            count = sum(type_plan.values())

    available = context.vocab.approved_lessons()
    if not lessons:
        lessons = available
    lessons = sorted(set(lessons))
    if not lessons:
        raise IntentError("教材词表为空：缺少可用的 vocab.json。")
    missing_lessons = [lesson for lesson in lessons if lesson not in available]
    if missing_lessons:
        warnings.append(f"以下课次没有已审核词表，已从范围移除：{missing_lessons}")
        lessons = [lesson for lesson in lessons if lesson in available]
        if not lessons:
            raise IntentError("请求的课次都没有已审核教材数据。")

    if scope is None:
        scope = "extend" if topic else "in_syllabus"
    if is_mock:
        scope = "in_syllabus"
    if topic and scope == "in_syllabus" and not any(
        word in normalized for word in SCOPE_STRICT
    ):
        # A topic request is an extension-learning request unless explicitly restricted.
        scope = "extend"

    use_case = use_case_override or _detect_use_case(normalized, is_mock)
    if use_case not in DEFAULT_SOURCE_MIX:
        use_case = "mock" if is_mock else "first_learn"
    return QuizPlan(
        raw=raw,
        mode="mock" if is_mock else "practice",
        scope=scope,
        topic=topic,
        count=sum(type_plan.values()),
        lessons=lessons,
        type_plan=type_plan,
        seed=seed if seed is not None else random.SystemRandom().randint(1, 2**31 - 1),
        source=source,
        use_case=use_case,
        source_mix=_source_mix(context, use_case),
        explicit_count=explicit_count,
        explicit_lessons=explicit_lessons,
        explicit_types=explicit_types,
        warnings=warnings,
    )
