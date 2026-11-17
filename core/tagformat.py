"""Tolerant tag parser for one-question LLM output.

Preferred format::

    [STEM]...[[blank]]...[/STEM]
    [D]distractor 1[/D]
    [D]distractor 2[/D]
    [D]distractor 3[/D]
    [EXP]中文解析[/EXP]
    [ZH]中文翻译[/ZH]

The parser intentionally accepts messy output: full-width brackets, code
fences, lower-case tags, missing closing tags, and ``D1``/``D2`` aliases.
It never treats ``[[blank]]`` as a tag.
"""

from __future__ import annotations

import re
from typing import Iterable

_FENCE_RE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*(.*?)```", re.S)
_TAG_RE = re.compile(
    r"\[([A-Za-z][A-Za-z0-9_]*)\](.*?)(?:\[/\1\]|(?=\[[A-Za-z][A-Za-z0-9_]*\]|\Z))",
    re.S | re.I,
)
_OPTION_LINE_RE = re.compile(
    r"^\s*(?:[-*・]\s*)?(?:[A-DＡ-Ｄ]|[1-4１-４])\s*[.、):：]\s*(.+?)\s*$"
)
_TRIM_RE = re.compile(r"^[\s\"'`]+|[\s\"'`]+$")
_BLANK_SENTINEL = "JLPTCRAFTBLANKTOKEN"
_ANS_OPEN_SENTINEL = "JLPTCRAFTANSOPEN"
_ANS_CLOSE_SENTINEL = "JLPTCRAFTANSCLOSE"


def normalize_brackets(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("［", "[")
        .replace("］", "]")
        .replace("【", "[")
        .replace("】", "]")
        .replace("　", " ")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    if match:
        return match.group(1)
    return text


def clean_value(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(
        r"^\s*(?:\[/?[A-Za-z][A-Za-z0-9_]*\]|</?[A-Za-z][A-Za-z0-9_]*>)",
        "",
        value,
        flags=re.I,
    )
    value = re.sub(
        r"(?:\[/?[A-Za-z][A-Za-z0-9_]*\]|</?[A-Za-z][A-Za-z0-9_]*>)\s*$",
        "",
        value,
        flags=re.I,
    )
    value = _TRIM_RE.sub("", value)
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


def parse_tags(text: str) -> dict[str, list[str]]:
    """Return a case-insensitive tag -> values mapping."""
    if not text:
        return {}
    normalized = _strip_fences(normalize_brackets(text))
    normalized = (
        normalized.replace("[[blank]]", _BLANK_SENTINEL)
        .replace("[[ans]]", _ANS_OPEN_SENTINEL)
        .replace("[[/ans]]", _ANS_CLOSE_SENTINEL)
    )
    result: dict[str, list[str]] = {}
    for match in _TAG_RE.finditer(normalized):
        name = match.group(1).strip().upper()
        value = (
            clean_value(match.group(2))
            .replace(_BLANK_SENTINEL, "[[blank]]")
            .replace(_ANS_OPEN_SENTINEL, "[[ans]]")
            .replace(_ANS_CLOSE_SENTINEL, "[[/ans]]")
        )
        if not value:
            continue
        result.setdefault(name, []).append(value)
    return result


def first_tag(
    tags: dict[str, list[str]],
    names: Iterable[str],
    default: str = "",
) -> str:
    for name in names:
        values = tags.get(str(name).upper())
        if values:
            return values[0]
    return default


def all_tags(
    tags: dict[str, list[str]],
    names: Iterable[str],
) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for name in names:
        for value in tags.get(str(name).upper(), []):
            key = re.sub(r"\s+", "", value)
            if key and key not in seen:
                seen.add(key)
                values.append(value)
    return values


def parse_option_lines(text: str) -> list[str]:
    """Fallback: parse lines like ``A. ...`` / ``1) ...`` from plain output."""
    if not text:
        return []
    options: list[str] = []
    for line in normalize_brackets(text).splitlines():
        match = _OPTION_LINE_RE.match(line)
        if match:
            value = clean_value(match.group(1))
            if value:
                options.append(value)
    return options


def parse_question_tags(text: str) -> dict[str, object]:
    """Parse one generated question into a small, provider-agnostic dict."""
    tags = parse_tags(text)
    stem = first_tag(tags, ("STEM", "QUESTION", "Q", "题干"))
    full = first_tag(tags, ("FULL", "SENTENCE", "DIALOGUE", "CHECK", "完整句"))
    clue = first_tag(tags, ("CLUE", "HINT", "线索"))
    distractors = all_tags(
        tags,
        ("D", "D1", "D2", "D3", "DISTRACTOR", "DISTRACTORS", "OPTION", "干扰项"),
    )
    if not distractors:
        # Some models put all options in one tag, one per line.
        combined = first_tag(tags, ("OPTIONS", "OPTION", "DISTRACTORS", "干扰项"))
        if combined:
            for line in combined.splitlines():
                line = re.sub(
                    r"^\s*(?:[-*・]\s*)?(?:[A-DＡ-Ｄ]|[1-4１-４])\s*[.、):：]\s*",
                    "",
                    line,
                ).strip()
                if line:
                    distractors.append(line)
    if not distractors:
        distractors = parse_option_lines(text)
    return {
        "stem": stem,
        "full": full,
        "clue": clue,
        "distractors": distractors,
        "explanation": first_tag(tags, ("EXP", "EXPLANATION", "EXPLAIN", "解析", "解释")),
        "translation": first_tag(tags, ("ZH", "TRANSLATION", "TRANS", "翻译", "译文")),
        "tags": tags,
    }


def parse_simple_tags(text: str, names: Iterable[str]) -> dict[str, str]:
    tags = parse_tags(text)
    return {name: first_tag(tags, (name,)) for name in names if first_tag(tags, (name,))}
