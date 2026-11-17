"""Question fingerprints and novelty content hashes."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from core.models import GeneratedQuestion, normalize_text


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text or "")).lower()


def fingerprint_question(question: GeneratedQuestion) -> str:
    """Stable DB dedupe fingerprint for a generated question."""
    options = sorted(_norm(option.text) for option in question.options)
    payload = "|".join(
        [
            question.type.value,
            str(question.textbook_lesson),
            _norm(question.stem),
            _norm(question.correct_option_text),
            "|".join(options),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def content_hash(
    stem: str,
    options: list[str],
    answer: str = "",
) -> str:
    """Content hash comparable between raw corpus and generated questions."""
    payload = "|".join(
        [
            _norm(stem),
            "|".join(sorted(_norm(item) for item in options)),
            _norm(answer),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def question_content_hash(question: GeneratedQuestion) -> str:
    return content_hash(
        question.stem,
        [option.text for option in question.options],
        question.correct_option_text or question.answer_key,
    )


def normalize_target(text: str) -> str:
    return _norm(text)


def dump_index_payload(index: dict[str, Any]) -> dict[str, Any]:
    """Keep only JSON-serializable, non-copyrighted fields."""
    return {
        "content_hashes": list(index.get("content_hashes", [])),
        "target_counts": dict(index.get("target_counts", {})),
        "meta": dict(index.get("meta", {})),
    }
