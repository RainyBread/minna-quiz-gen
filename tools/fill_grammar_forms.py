#!/usr/bin/env python
"""Propose answer forms for lessons 44-50 grammar points from their examples."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import load_settings  # noqa: E402
from core.llm import LLMClient  # noqa: E402
from core.models import GrammarFile, now_iso  # noqa: E402
from core.tagformat import all_tags, parse_tags  # noqa: E402

_MATCH_STRIP = " \t　。、，,．.!?！？「」『』()（）"


def _compact(text: str) -> str:
    return "".join(char for char in (text or "") if char not in _MATCH_STRIP)


def _valid_forms(forms: list[str], examples: list[str]) -> list[str]:
    compact_examples = [_compact(example) for example in examples]
    result: list[str] = []
    seen: set[str] = set()
    for raw in forms:
        form = (raw or "").strip().strip("「」『』\"'")
        if not form or len(form) < 2 or len(form) > 30:
            continue
        if form in seen:
            continue
        compact = _compact(form)
        if not compact or not any(compact in example for example in compact_examples):
            continue
        seen.add(form)
        result.append(form)
        if len(result) >= 3:
            break
    return result


def _suggest_tags(pattern: str, forms: list[str], examples: list[str]) -> list[str]:
    tags = ["grammar_fill"]
    conversational = ("ください", "ます", "ですか", "ましょう", "ござい", "いたし", "ませんか")
    if any(token in pattern for token in ("尊敬", "謙譲", "丁寧")) or any(
        token in form for form in forms for token in conversational
    ):
        tags.append("expression_choice")
    return tags


def _propose(point: dict[str, Any], llm: LLMClient) -> dict[str, Any]:
    pattern = str(point.get("pattern_ja") or "")
    meaning = str(point.get("meaning_zh") or "")
    examples = [str(item) for item in point.get("examples", []) or [] if str(item).strip()]
    system = (
        "你是日语语法教研助手。请从给定例句中提取 1-3 个可以直接放进空格、"
        "且符合该语法点的日语形式。选择例句中实际出现的连续片段，"
        "优先完整、自然的谓语或表达。请用 FORM 标签输出。"
    )
    user = (
        f"语法点：{pattern}\n意思：{meaning}\n"
        "例句：\n" + "\n".join(f"- {example}" for example in examples[:5])
        + "\n请按以下标签输出：\n[FORM]形式1[/FORM]\n[FORM]形式2[/FORM]"
    )
    raw = ""
    error = ""
    try:
        raw = llm.complete_text(system, user, temperature=0.2, max_tokens=400)
        tags = parse_tags(raw)
        proposed = all_tags(tags, ("FORM", "ANSWER", "F"))
    except Exception as exc:
        proposed = []
        error = f"{type(exc).__name__}: {exc}"
    valid = _valid_forms(proposed, examples)
    return {
        "gp_id": point.get("gp_id"),
        "lesson": point.get("lesson"),
        "pattern_ja": pattern,
        "meaning_zh": meaning,
        "examples": examples,
        "proposed_forms": proposed,
        "accepted_forms": valid,
        "suggested_form_tags": _suggest_tags(pattern, valid, examples),
        "confidence": "high" if valid else "none",
        "error": error,
        "reviewed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", help="draft output path")
    args = parser.parse_args(argv)
    settings = load_settings(ROOT)
    grammar = GrammarFile.model_validate(
        json.loads(settings.grammar_path.read_text(encoding="utf-8"))
    )
    targets = [
        point
        for point in grammar.points
        if point.lesson and 44 <= point.lesson <= 50 and not point.approved_answer_forms
    ]
    if not targets:
        print("no empty-form 44-50 grammar points")
        return 0
    llm = LLMClient(settings)
    if not llm.available:
        print("LLM key not configured")
        return 1
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(_propose, point.model_dump(mode="json"), llm): point
            for point in targets
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            forms = ", ".join(result["accepted_forms"]) or "（无有效候选）"
            print(f"{result['gp_id']} {result['pattern_ja']} -> {forms}")
    results.sort(key=lambda item: (item.get("lesson") or 0, item.get("gp_id") or ""))
    output = (
        Path(args.output)
        if args.output
        else settings.drafts_dir / "grammar_forms_44_50.draft.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "generated_at": now_iso(),
            "source": "LLM extraction from attested examples; requires human review",
            "count": len(results),
        },
        "points": results,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"draft written: {output}")
    print("review with: python tools/review_grammar_forms.py --merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
