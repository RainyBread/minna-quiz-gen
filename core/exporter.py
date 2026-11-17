"""Markdown/JSON archival for validated batches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import Settings
from core.items import Item, item_from_json
from core.models import GeneratedBatch, MistakeRulesFile, QuestionType, TYPE_LABELS_ZH, now_iso
from core.render import (
    AnswerMode,
    RenderedQuestion,
    render_answer_book,
    render_inline_book,
    render_item,
    render_question_book,
    resolve_answer_mode,
)


def _rule_name(rules: MistakeRulesFile | None, rule_id: str | None) -> str:
    if not rule_id:
        return ""
    if rules is not None:
        rule = rules.rule_map().get(rule_id)
        if rule:
            return f"{rule.id}（{rule.name_zh}）"
    return rule_id


def _display_stem(stem: str) -> str:
    return stem.replace("[[blank]]", "（　　）")


def _safe_output_path(directory: Path, filename: str) -> Path:
    """Resolve an archive path and ensure it stays inside ``directory``."""
    base = directory.resolve()
    target = (base / filename).resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"unsafe archive filename: {filename!r}")
    return target


def export_batch_markdown(
    batch: GeneratedBatch,
    path: Path,
    *,
    rules: MistakeRulesFile | None = None,
    include_answers: bool = True,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = batch.batch_meta
    title = "模拟卷" if meta.mode == "mock" else "练习"
    scope_label = {
        "in_syllabus": "课内",
        "mdict_distractors": "课内 + 拓展干扰项",
        "extend": "拓展学习",
    }.get(meta.scope, meta.scope)
    lines: list[str] = [
        f"# JLPTCraft 大二上{title} · {meta.id}",
        "",
        f"- 学期：`{meta.semester.value}`",
        f"- 生成时间：{meta.created_at or now_iso()}",
        f"- 请求：{meta.request or '-'}",
        f"- 范围：{scope_label}" + (f" / 主题：{meta.topic}" if meta.topic else ""),
        f"- 课次：{', '.join(str(x) for x in meta.lessons) or '-'}",
        f"- 题型：{', '.join(f'{k}×{v}' for k, v in meta.type_plan.items()) or '-'}",
        f"- 模型：`{meta.model}`",
        f"- 规则版本：v{meta.rules_version} / hash `{meta.ruleset_hash[:12]}`",
        "",
        "> 本卷由 JLPTCraft 生成；读音来自已复核教材词表，扩展词中文释义来自本地 MDict。",
        "> LLM 只负责线索与完整句；正确答案、选项和解析由本地锁词表/词表构造；未做语义审计。",
        "",
    ]
    if meta.warnings:
        lines.append("**批次警告：**")
        lines.extend(f"- {warning}" for warning in meta.warnings)
        lines.append("")

    grouped: dict[str, list] = {}
    for question in batch.questions:
        grouped.setdefault(question.type.value, []).append(question)

    number = 0
    for type_value, questions in grouped.items():
        label = TYPE_LABELS_ZH.get(type_value, type_value)
        lines.append(f"## {label}（{len(questions)}题）")
        lines.append("")
        for question in questions:
            number += 1
            suffix = " 〔拓展〕" if question.scope == "extend" else ""
            lines.append(f"### {number}. {_display_stem(question.stem)}{suffix}")
            lines.append("")
            for option in question.options:
                lines.append(f"- **{option.key}.** {option.text}")
            lines.append("")
            if question.translation_zh:
                lines.append(f"> 参考翻译：{question.translation_zh}")
                lines.append("")
    if include_answers:
        lines.append("## 答案与解析")
        lines.append("")
        number = 0
        for type_value, questions in grouped.items():
            for question in questions:
                number += 1
                lines.append(f"### {number}. 答案：{question.answer_key}. {question.correct_option_text}")
                lines.append("")
                if question.explanation_zh:
                    lines.append(question.explanation_zh)
                    lines.append("")
                distractor_lines = []
                for option in question.options:
                    if option.role.value != "distractor":
                        continue
                    rule_name = _rule_name(rules, option.mistake_rule_id)
                    if not rule_name and not option.wrongness_zh:
                        continue
                    distractor_lines.append(
                        f"- {option.key}. {option.text}"
                        + (f"：{rule_name}" if rule_name else "")
                        + (f" — {option.wrongness_zh}" if option.wrongness_zh else "")
                    )
                if distractor_lines:
                    lines.append("**设误点：**")
                    lines.extend(distractor_lines)
                    lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _render_batch_views(batch: GeneratedBatch) -> list[RenderedQuestion]:
    rendered: list[RenderedQuestion] = []
    seed = int(batch.batch_meta.seed or 0)
    for index, data in enumerate(batch.items, start=1):
        try:
            item = item_from_json(data)
            rendered.append(render_item(item, seed=seed, number=index))
        except Exception:
            continue
    return rendered


def _book_header(batch: GeneratedBatch, book_title: str) -> str:
    meta = batch.batch_meta
    mode = "模拟卷" if meta.mode == "mock" else "练习"
    lessons = ", ".join(str(x) for x in meta.lessons) or "-"
    types = ", ".join(f"{k}×{v}" for k, v in meta.type_plan.items()) or "-"
    return "\n".join(
        [
            f"# JLPTCraft 大二上{mode}{book_title} · {meta.id}",
            "",
            f"- 学期：`{meta.semester.value}`",
            f"- 请求：{meta.request or '-'}",
            f"- 课次：{lessons}",
            f"- 题型：{types}",
            f"- 种子：{meta.seed}",
            "",
        ]
    )


def _book_body(text: str) -> str:
    # Drop the renderer's generic ``# title`` line; the exporter supplies a
    # stable header with batch/profile facts.
    parts = text.split("\n", 1)
    return parts[1].lstrip("\n") if len(parts) > 1 else text


def export_question_book(
    rendered: list[RenderedQuestion],
    path: Path,
    *,
    batch: GeneratedBatch,
) -> Path:
    text = _book_header(batch, "试题册") + _book_body(
        render_question_book(rendered, title="试题册")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def export_answer_book(
    rendered: list[RenderedQuestion],
    path: Path,
    *,
    batch: GeneratedBatch,
) -> Path:
    text = _book_header(batch, "答案册") + _book_body(
        render_answer_book(rendered, title="答案册")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def export_full_jsonl(batch: GeneratedBatch, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(item, ensure_ascii=False, sort_keys=True)
        for item in batch.items
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def archive_batch(
    settings: Settings,
    batch: GeneratedBatch,
    *,
    rules: MistakeRulesFile | None = None,
    include_answers: bool = True,
) -> tuple[Path, Path]:
    settings.exam_dir.mkdir(parents=True, exist_ok=True)
    json_path = _safe_output_path(settings.exam_dir, f"{batch.batch_meta.id}.json")
    md_path = _safe_output_path(settings.exam_dir, f"{batch.batch_meta.id}.md")
    payload = {
        "exported_at": now_iso(),
        "batch": batch.model_dump(mode="json"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not batch.items:
        export_batch_markdown(batch, md_path, rules=rules, include_answers=include_answers)
        return json_path, md_path

    rendered = _render_batch_views(batch)
    if not rendered:
        export_batch_markdown(batch, md_path, rules=rules, include_answers=include_answers)
        return json_path, md_path
    mode = resolve_answer_mode(
        batch.batch_meta.answers_mode,
        is_mock=batch.batch_meta.mode == "mock",
    )
    export_full_jsonl(batch, _safe_output_path(settings.exam_dir, f"{batch.batch_meta.id}.items.jsonl"))
    answer_path = _safe_output_path(
        settings.exam_dir, f"{batch.batch_meta.id}-answers.md"
    )
    if mode == AnswerMode.ONLY:
        export_answer_book(rendered, md_path, batch=batch)
        return json_path, md_path
    if mode == AnswerMode.INLINE:
        inline_path = _safe_output_path(
            settings.exam_dir, f"{batch.batch_meta.id}-inline.md"
        )
        text = _book_header(batch, "题目与答案") + _book_body(
            render_inline_book(rendered, title="题目与答案")
        )
        inline_path.write_text(text.rstrip() + "\n", encoding="utf-8")
        return json_path, inline_path
    export_question_book(rendered, md_path, batch=batch)
    if include_answers:
        export_answer_book(rendered, answer_path, batch=batch)
    return json_path, md_path


def load_batch_json(path: Path) -> GeneratedBatch:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "batch" in data:
        data = data["batch"]
    return GeneratedBatch.model_validate(data)


def export_existing_batch(
    settings: Settings,
    batch: GeneratedBatch,
    *,
    rules: MistakeRulesFile | None = None,
    include_answers: bool = True,
) -> Path:
    if not batch.items:
        # Migrated legacy batches keep their canonical Items in the DB.
        from core.db import list_items

        batch.items = list_items(settings, batch.batch_meta.id)
    target = _safe_output_path(settings.exam_dir, f"{batch.batch_meta.id}.md")
    if not include_answers:
        target = _safe_output_path(
            settings.exam_dir, f"{batch.batch_meta.id}-student.md"
        )
    if batch.items:
        rendered = _render_batch_views(batch)
        if rendered:
            export_question_book(rendered, target, batch=batch)
            if include_answers:
                answer_path = _safe_output_path(
                    settings.exam_dir, f"{batch.batch_meta.id}-answers.md"
                )
                export_answer_book(rendered, answer_path, batch=batch)
            return target
    return export_batch_markdown(
        batch, target, rules=rules, include_answers=include_answers
    )


def render_batch_console(batch: GeneratedBatch) -> str:
    lines = [
        f"批次 {batch.batch_meta.id} | {len(batch.questions)} 题 | "
        f"{batch.batch_meta.semester.value} | {batch.batch_meta.model}",
    ]
    for index, question in enumerate(batch.questions, start=1):
        lines.append(
            f"{index}. [{question.type.value}] {question.stem.replace('[[blank]]', '（　）')}"
        )
        for option in question.options:
            lines.append(f"   {option.key}. {option.text}")
        lines.append(f"   答案：{question.answer_key}. {question.correct_option_text}")
    return "\n".join(lines)
