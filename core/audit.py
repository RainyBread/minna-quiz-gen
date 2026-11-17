"""Read-only Item/view/request audit.

The audit never calls an LLM or MDict and never prints full answer text or
textbook passages.  It reports counts and masked item-level details.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config import Settings
from core.db import latest_batch_id, list_items
from core.items import Item, item_from_json, validate_item
from core.models import normalize_text
from core.render import (
    render_answer_book,
    render_item,
    render_question_book,
)
from core.request_log import find_requests


@dataclass
class AuditReport:
    batch_id: str = ""
    items: int = 0
    round_trip_violations: int = 0
    missing_carrier: int = 0
    view_leaks: int = 0
    request_key_leaks: int = 0
    answer_mismatch: int = 0
    reconstructed_carrier: int = 0
    key_prompt_visible: int = 0
    legacy_skipped: int = 0
    request_checked: int = 0
    no_request: int = 0
    defect_counts: dict[str, int] = field(default_factory=dict)
    details: list[dict[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(
            [
                self.round_trip_violations,
                self.missing_carrier,
                self.view_leaks,
                self.request_key_leaks,
                self.answer_mismatch,
            ]
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "items": self.items,
            "round_trip_violations": self.round_trip_violations,
            "missing_carrier": self.missing_carrier,
            "view_leaks": self.view_leaks,
            "request_key_leaks": self.request_key_leaks,
            "answer_mismatch": self.answer_mismatch,
            "reconstructed_carrier": self.reconstructed_carrier,
            "key_prompt_visible": self.key_prompt_visible,
            "legacy_skipped": self.legacy_skipped,
            "request_checked": self.request_checked,
            "no_request": self.no_request,
            "defect_counts": dict(self.defect_counts),
            "ok": self.ok,
            "details": list(self.details),
        }


def _mask(text: str, limit: int = 12) -> str:
    value = (text or "").replace("\n", " ")
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def _check_view_leaks(item: Item, rendered: Any) -> list[str]:
    leaks: list[str] = []
    stem = rendered.stem or ""
    key = item.key.value or ""
    if key and normalize_text(key) in normalize_text(stem):
        leaks.append("key in stem")
    for marker in ("答案：", "解析：", "正确答案", "Answer:"):
        if marker in stem:
            leaks.append(f"answer marker: {marker}")
    return leaks


def _check_request_leaks(item: Item, prompt_text: str) -> list[str]:
    leaks: list[str] = []
    prompt = normalize_text(prompt_text)
    key = normalize_text(item.key.value)
    if key and not item.meta.prompt_visible and key in prompt:
        leaks.append("key in prompt")
    for distractor in item.distractors:
        value = normalize_text(distractor.value)
        if value and value in prompt:
            leaks.append(f"distractor in prompt: {_mask(distractor.value)}")
            break
    explanation = normalize_text(item.explanation)
    if len(explanation) >= 8 and explanation in prompt:
        leaks.append("explanation in prompt")
    return leaks


def audit_items(
    settings: Settings,
    batch_id: str = "",
    *,
    include_requests: bool = True,
) -> AuditReport:
    if not batch_id:
        batch_id = latest_batch_id(settings)
    report = AuditReport(batch_id=batch_id)
    rows = list_items(settings, batch_id) if batch_id else []
    items: list[Item] = []
    for data in rows:
        try:
            items.append(item_from_json(data))
        except Exception:
            report.legacy_skipped += 1
    report.items = len(items)
    rendered = []
    for index, item in enumerate(items, start=1):
        seed = int(item.meta.seed or 0)
        try:
            rq = render_item(item, seed=seed, number=index)
            rendered.append(rq)
        except Exception as exc:
            report.details.append(
                {"item_id": item.id, "code": "E_PARSE", "message": str(exc)}
            )
            report.missing_carrier += 1
            continue
        defects = validate_item(item)
        for defect in defects:
            if defect.code == "E_ANCHOR":
                report.round_trip_violations += 1
                if "carrier" in defect.field:
                    report.missing_carrier += 1
            elif defect.code == "E_PARSE" and "carrier" in defect.field:
                report.missing_carrier += 1
            if defect.code in {"E_ANCHOR", "E_PARSE", "E_ZERO"}:
                report.details.append(
                    {
                        "item_id": item.id,
                        "code": defect.code,
                        "field": defect.field,
                        "message": defect.message,
                    }
                )
        view_leaks = _check_view_leaks(item, rq)
        if view_leaks:
            report.view_leaks += 1
            report.details.append(
                {
                    "item_id": item.id,
                    "code": "VIEW_LEAK",
                    "message": "; ".join(view_leaks),
                }
            )
        for code in item.meta.defects:
            report.defect_counts[code] = report.defect_counts.get(code, 0) + 1
        if item.meta.provenance.get("carrier_source") == "reconstructed":
            report.reconstructed_carrier += 1
        if item.meta.prompt_visible:
            report.key_prompt_visible += 1
        if include_requests:
            request_id = str(item.meta.provenance.get("request_id") or "")
            if not request_id:
                report.no_request += 1
            else:
                records = [
                    record
                    for record in find_requests(settings)
                    if str(record.get("request_id") or "") == request_id
                ]
                report.request_checked += 1
                for record in records:
                    prompt_text = (
                        str(record.get("system") or "")
                        + "\n"
                        + str(record.get("user") or "")
                    )
                    leaks = _check_request_leaks(item, prompt_text)
                    if leaks:
                        report.request_key_leaks += 1
                        report.details.append(
                            {
                                "item_id": item.id,
                                "code": "REQUEST_LEAK",
                                "message": "; ".join(leaks),
                            }
                        )
                        break

    if rendered:
        question_text = render_question_book(rendered, title="试题册")
        answer_text = render_answer_book(rendered, title="答案册")
        # The question book must not carry answer markers or explanations.
        for marker in ("答案：", "解析：", "正确答案"):
            if marker in question_text:
                report.view_leaks += question_text.count(marker)
                report.details.append(
                    {
                        "item_id": "",
                        "code": "VIEW_LEAK",
                        "message": f"question book contains {marker}",
                    }
                )
        for item in items:
            if len(normalize_text(item.explanation)) >= 8:
                if item.explanation in question_text:
                    report.view_leaks += 1
                    report.details.append(
                        {
                            "item_id": item.id,
                            "code": "VIEW_LEAK",
                            "message": "explanation in question book",
                        }
                    )
        q_numbers = sorted(
            int(match.group(1))
            for match in re.finditer(r"^\[(\d+)\]", question_text, re.M)
        )
        a_numbers = sorted(
            int(match.group(1))
            for match in re.finditer(r"^\[(\d+)\]\s*答案", answer_text, re.M)
        )
        if q_numbers != a_numbers:
            report.answer_mismatch = 1
            report.details.append(
                {
                    "item_id": "",
                    "code": "ANSWER_MISMATCH",
                    "message": f"question={q_numbers} answer={a_numbers}",
                }
            )
    return report


def audit_text(report: AuditReport) -> str:
    lines = [
        f"审计批次：{report.batch_id or '-'} | items {report.items}",
        f"回环恒等 violations：{report.round_trip_violations}",
        f"无 carrier：{report.missing_carrier}",
        f"试题册泄漏：{report.view_leaks}",
        f"请求 key 泄漏：{report.request_key_leaks}"
        f"（已查 {report.request_checked}，无请求 {report.no_request}）",
        f"答案册题号不一致：{report.answer_mismatch}",
        f"迁移重建 carrier：{report.reconstructed_carrier}",
        f"prompt_visible 目标词：{report.key_prompt_visible}",
        f"缺陷码：{report.defect_counts or '-'}",
        f"旧 schema 跳过：{report.legacy_skipped}",
        "结论：" + ("OK" if report.ok else "FAIL"),
    ]
    for detail in report.details[:20]:
        lines.append(
            f"  - {detail.get('item_id') or '-'} {detail.get('code')}: {detail.get('message')}"
        )
    return "\n".join(lines)
