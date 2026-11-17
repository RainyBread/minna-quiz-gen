"""Cheap local surface-overlap checks (no LLM judge)."""

from __future__ import annotations

import re

_CLEAN_RE = re.compile(r"[\s。、，,.!?！？「」『』()（）\[\]【】]+")


def _clean(text: str) -> str:
    return _CLEAN_RE.sub("", text or "")


def ngrams(text: str, n: int = 5) -> set[str]:
    value = _clean(text)
    if len(value) <= n:
        return {value} if value else set()
    return {value[index : index + n] for index in range(len(value) - n + 1)}


def overlap(a: str, b: str, n: int = 5) -> float:
    grams_a = ngrams(a, n)
    grams_b = ngrams(b, n)
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / max(1, min(len(grams_a), len(grams_b)))


def max_overlap(stem: str, references: list[str], n: int = 5) -> float:
    return max((overlap(stem, reference, n) for reference in references if reference), default=0.0)
