"""Targeted OCR extraction for scanned textbook pages (lessons 44-50).

Outputs are always drafts.  Nothing here is trusted for generation until the
human runs a review/approval step.  Vision LLM is preferred; local RapidOCR
plus an optional text-LLM cleanup is the fallback.
"""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

from config import Settings
from core.llm import LLMClient, LLMError, VisionUnavailable
from core.models import (
    GrammarPoint,
    QuestionType,
    RuleStatus,
    VocabEntry,
    contains_kanji,
    is_katakana_only,
    now_iso,
    sha256_text,
)

MAIN_PDF_NAME = "大家的日语2-外研社.pdf"
WORKBOOK_PDF_NAME = "大家的日语2 辅导用书.pdf"

# Verified from the printed TOC / PDF page offset.  Lessons 44-50.
DEFAULT_LESSON_PAGES: dict[int, int] = {
    44: 173,
    45: 181,
    46: 191,
    47: 199,
    48: 207,
    49: 215,
    50: 223,
}

# Workbook book-page numbers from its TOC; the PDF pages were verified by OCR
# (workbook PDF page = printed page + 19).
DEFAULT_WORKBOOK_PAGES: dict[int, int | None] = {
    44: 112,
    45: 118,
    46: 124,
    47: 130,
    48: 136,
    49: 142,
    50: 148,
}
DEFAULT_WORKBOOK_PDF_PAGES: dict[int, int] = {
    44: 131,
    45: 137,
    46: 143,
    47: 149,
    48: 155,
    49: 161,
    50: 167,
}


def _default_page_map(settings: Settings) -> dict[str, Any]:
    return {
        "meta": {
            "semester": settings.active_semester.value,
            "source": "TOC OCR + verified page offset",
            "generated_at": now_iso(),
            "notes": [
                "main_pages 是主教材 PDF 的 1-indexed 页码；lesson 44=173, 45=181, 46=191, 47=199, 48=207, 49=215, 50=223。",
                "lesson 46 的起始页由 TOC 推断（书页约170），OCR 时请核对；可直接修改本文件。",
                "workbook_pdf_pages 是辅导用书单词页的 PDF 页码（44=131, 45=137, 46=143, 47=149, 48=155, 49=161, 50=167），用于词汇提取。",
                "workbook_book_page 是辅导用书印刷页码（44=112, 45=118, ...）。",
                "页面映射可人工修改；OCR 产物必须审核后才并入 vocab/grammar。",
            ],
        },
        "main_pdf": MAIN_PDF_NAME,
        "workbook_pdf": WORKBOOK_PDF_NAME,
        "lessons": {
            str(lesson): {
                "lesson": lesson,
                "course_lesson": lesson - 36,
                "main_pages": [page],
                "workbook_book_page": DEFAULT_WORKBOOK_PAGES[lesson],
                "workbook_pdf_pages": [DEFAULT_WORKBOOK_PDF_PAGES[lesson]],
            }
            for lesson, page in DEFAULT_LESSON_PAGES.items()
        },
    }


def load_page_map(settings: Settings) -> dict[str, Any]:
    path = settings.pdf_page_map_path
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict) and data.get("lessons"):
                changed = False
                for key, info in data["lessons"].items():
                    try:
                        lesson = int(key)
                    except Exception:
                        continue
                    if not info.get("workbook_pdf_pages") and lesson in DEFAULT_WORKBOOK_PDF_PAGES:
                        info["workbook_pdf_pages"] = [DEFAULT_WORKBOOK_PDF_PAGES[lesson]]
                        changed = True
                if changed:
                    path.write_text(
                        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                return data
        except Exception:
            pass
    data = _default_page_map(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def _render_png(pdf_path: Path, page_no: int, dpi: int = 220) -> tuple[bytes, Any]:
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    try:
        if page_no < 1 or page_no > doc.page_count:
            raise ValueError(f"PDF 页码越界：{page_no} / {doc.page_count}")
        page = doc[page_no - 1]
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
        png = pix.tobytes("png")
        try:
            import numpy as np

            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )[:, :, :3]
        except Exception:
            image = None
        return png, image
    finally:
        doc.close()


_OCR_ENGINE: Any = None


def _ocr_lines(image: Any) -> list[str]:
    global _OCR_ENGINE
    if image is None:
        return []
    try:
        from rapidocr_onnxruntime import RapidOCR
    except Exception:
        return []
    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()
    result, _ = _OCR_ENGINE(image)
    items: list[tuple[float, float, str]] = []
    for item in result or []:
        box, text, _score = item
        ys = [point[1] for point in box]
        xs = [point[0] for point in box]
        items.append((min(ys), min(xs), str(text)))
    items.sort(key=lambda row: (round(row[0] / 20), row[1]))
    return [text for _, _, text in items]


def _vision_prompt_vocab(lesson: int, page_no: int, book_page: int | None = None) -> tuple[str, str]:
    system = (
        "你是日语教材结构化提取器。只输出严格 JSON，不要 Markdown。"
        "只提取图片中实际可见的内容，不要编造；无法辨认的条目不要输出，或在 uncertain 中列出。"
    )
    book = f"（印刷页码 {book_page}）" if book_page else ""
    user = (
        f"这是《大家的日语》初级2 第{lesson}课辅导用书的单词表页{book}（PDF第{page_no}页）。"
        "页面通常有三列：假名读音、汉字写法、中文释义；词性用 I/II/III、[な]、[い] 等标记。"
        "请按页面顺序逐行提取全部词条，不要遗漏。输出 schema：\n"
        '{"lesson": ' + str(lesson) + ', '
        '"vocab": [{"kana": "平假名读音", "written": "汉字/假名写法", '
        '"translation_zh": "中文", "pos": "词性", "confidence": "high|medium|low"}], '
        '"uncertain": ["无法辨认的内容"]}\n'
        "不要输出页码、练习编号、例句或语法点；读音必须与页面一致。"
    )
    return system, user


def _vision_prompt_grammar(lesson: int, page_no: int) -> tuple[str, str]:
    system = (
        "你是日语教材结构化提取器。只输出严格 JSON，不要 Markdown。"
        "只提取图片中实际可见的语法点，不要编造；无法辨认的内容放入 uncertain。"
    )
    user = (
        f"这是《大家的日语》初级2 第{lesson}课的主教材页（PDF第{page_no}页）。"
        "只提取文型/语法点（不要提取单词表）。输出 schema：\n"
        '{"lesson": ' + str(lesson) + ', '
        '"grammar": [{"pattern_ja": "句型", "meaning_zh": "中文意思", '
        '"formation": "接续/形式", "examples": ["例句"], "confidence": "high|medium|low"}], '
        '"uncertain": []}\n'
        "如果该页没有明确语法点，grammar 返回空数组，不要编造。"
    )
    return system, user


_ACCENT_RE = re.compile(r"[①-⑳⓪㉑-㉟]")
_MARKER_RE = re.compile(r"[\[［][^\]］]*[\]］]")


def _strip_ocr_noise(value: str) -> str:
    value = _ACCENT_RE.sub("", value or "")
    value = _MARKER_RE.sub("", value)
    value = value.replace("　", " ").strip()
    return re.sub(r"\s+", "", value)


def _normalize_pos(value: str) -> str:
    raw = (
        (value or "")
        .strip()
        .replace("[", "")
        .replace("]", "")
        .replace("［", "")
        .replace("］", "")
    )
    mapping = {
        "I": "动I",
        "Ⅱ": "动II",
        "II": "动II",
        "III": "动III",
        "na": "ナ形",
        "な": "ナ形",
        "i": "イ形",
        "い": "イ形",
    }
    return mapping.get(raw, raw)


def _parse_vision_vocab(
    data: dict[str, Any],
    lesson: int,
    page_no: int,
    *,
    source_file: str = WORKBOOK_PDF_NAME,
    source_kind: str = "pdf_vision_workbook",
    book_page: int | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in data.get("vocab") or []:
        if not isinstance(item, dict):
            continue
        written = _strip_ocr_noise(str(item.get("written") or ""))
        kana = _strip_ocr_noise(str(item.get("kana") or ""))
        if not kana:
            continue
        # Skip pattern / affix rows such as [事故が～] or ～倍.
        if any(mark in f"{kana}{written}" for mark in "[]［］～〜－-/・"):
            continue
        if not written:
            written = kana
        if not (is_katakana_only(kana) or contains_kanji(written) or written == kana):
            continue
        confidence = str(item.get("confidence") or "low")
        if confidence not in {"high", "medium", "low"}:
            confidence = "low"
        entries.append(
            {
                "entry_id": f"ocr{lesson}-{len(entries)+1:03d}",
                "lesson": lesson,
                "course_lesson": lesson - 36,
                "written": written,
                "kana": kana,
                "translation_zh": str(item.get("translation_zh") or ""),
                "pos": _normalize_pos(str(item.get("pos") or "")),
                "notes": str(item.get("notes") or ""),
                "confidence": confidence,
                "has_kanji": contains_kanji(written),
                "is_loanword": is_katakana_only(written),
                "source": {
                    "file": source_file,
                    "page": page_no,
                    "book_page": book_page,
                    "kind": source_kind,
                },
                "reviewed": False,
                "unidic_verified": None,
            }
        )
    return entries


def _parse_vision_grammar(
    data: dict[str, Any],
    lesson: int,
    page_no: int,
    *,
    source_file: str = MAIN_PDF_NAME,
    source_kind: str = "pdf_vision",
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for item in data.get("grammar") or []:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("pattern_ja") or "").strip()
        if not pattern:
            continue
        points.append(
            {
                "gp_id": f"ocrg{lesson}-{len(points)+1:03d}",
                "lesson": lesson,
                "pattern_ja": pattern,
                "meaning_zh": str(item.get("meaning_zh") or ""),
                "formation": str(item.get("formation") or ""),
                "examples": [str(x) for x in (item.get("examples") or []) if x],
                "source_refs": [
                    {"file": source_file, "page": page_no, "kind": source_kind}
                ],
                "approved_answer_forms": [],
                "allowed_distractor_rule_ids": [
                    "particle_error",
                    "conjugation_error",
                    "tense_aspect_error",
                    "voice_error",
                    "collocation_error",
                    "situational_error",
                ],
                "confidence": str(item.get("confidence") or "low"),
                "status": RuleStatus.DRAFT.value,
                "reviewed": False,
            }
        )
    return points


def _text_cleanup_prompt(lesson: int, lines: list[str]) -> tuple[str, str]:
    system = (
        "你是日语教材 OCR 结构化助手。只输出严格 JSON。"
        "只根据给出的 OCR 行提取，不要编造；无法确定的读音/中文留空，并在 uncertain 中标注。"
    )
    user = (
        f"第{lesson}课教材页 OCR 行如下：\n"
        + "\n".join(lines[:120])
        + '\n输出：{"lesson": ' + str(lesson) + ', "vocab": [{"written","kana","translation_zh","pos","confidence","notes"}], '
        '"grammar": [{"pattern_ja","meaning_zh","formation","examples","confidence"}], "uncertain": []}'
    )
    return system, user


def maybe_extract_textbook_drafts(
    settings: Settings,
    *,
    force: bool = False,
    offline: bool = False,
    max_lessons: int | None = None,
) -> dict[str, Any]:
    """Extract lessons 44-50 into draft JSON files (never auto-approved)."""
    vocab_draft = settings.drafts_dir / "vocab_44_50.draft.json"
    grammar_draft = settings.drafts_dir / "grammar_44_50.draft.json"
    if not force and vocab_draft.exists() and grammar_draft.exists():
        return {"status": "exists", "files": [vocab_draft.name, grammar_draft.name]}

    page_map = load_page_map(settings)
    main_pdf = settings.textbook_pdf_dir / page_map.get("main_pdf", MAIN_PDF_NAME)
    workbook_pdf = settings.textbook_pdf_dir / page_map.get("workbook_pdf", WORKBOOK_PDF_NAME)
    if not main_pdf.exists():
        return {"status": "skipped", "reason": f"缺少主教材 PDF：{main_pdf}"}
    try:
        import fitz  # noqa: F401
    except Exception:
        return {"status": "skipped", "reason": "未安装 pymupdf；pip install -r requirements-ocr.txt"}

    lessons = sorted(
        (int(key), value) for key, value in page_map.get("lessons", {}).items()
    )
    if max_lessons:
        lessons = lessons[:max_lessons]

    llm = LLMClient(settings, offline=offline)
    vision = bool(settings.has_vision and not offline)
    text_llm = bool(settings.has_llm and not offline)
    if not vision and not text_llm:
        return {
            "status": "skipped",
            "reason": (
                "未配置视觉/文本 LLM，本地 RapidOCR 对日文假名/小字可靠性不足；"
                "44–50 课 OCR 草稿未生成。配置 VISION_API_KEY 或 LLM_API_KEY 后运行 "
                "python tools/extract_textbook.py。"
            ),
        }
    vocab_draft_lessons: list[dict[str, Any]] = []
    grammar_points: list[dict[str, Any]] = []
    grammar_raw_by_lesson: dict[str, list[str]] = {}
    notes: list[str] = []
    total_vocab = 0

    for lesson, info in lessons:
        main_pages = [int(p) for p in (info.get("main_pages") or []) if p]
        workbook_pages = [int(p) for p in (info.get("workbook_pdf_pages") or []) if p]
        if not main_pages and not workbook_pages:
            notes.append(f"第{lesson}课没有配置 PDF 页码。")
            continue

        # ---- vocabulary: prefer the workbook's clear three-column word list ----
        use_workbook = bool(workbook_pages and workbook_pdf.exists())
        vocab_page = workbook_pages[0] if use_workbook else main_pages[0]
        vocab_pdf = workbook_pdf if use_workbook else main_pdf
        vocab_source_file = WORKBOOK_PDF_NAME if use_workbook else MAIN_PDF_NAME
        vocab_source_kind = "pdf_vision_workbook" if use_workbook else "pdf_vision_main"
        try:
            vocab_png, vocab_image = _render_png(vocab_pdf, vocab_page)
        except Exception as exc:
            notes.append(f"第{lesson}课单词页 PDF第{vocab_page}页渲染失败：{exc}")
            continue
        vocab_ocr_lines = _ocr_lines(vocab_image)
        structured_vocab: dict[str, Any] = {}
        if vision:
            try:
                system, user = _vision_prompt_vocab(
                    lesson, vocab_page, info.get("workbook_book_page")
                )
                structured_vocab = llm.complete_vision_json(
                    system, user, vocab_png, max_tokens=16000
                )
                if not isinstance(structured_vocab, dict):
                    structured_vocab = {}
            except (VisionUnavailable, LLMError) as exc:
                notes.append(f"第{lesson}课单词视觉 OCR 失败：{exc}")
        if not structured_vocab and text_llm and vocab_ocr_lines:
            try:
                system, user = _text_cleanup_prompt(lesson, vocab_ocr_lines)
                structured_vocab = llm.complete_json(
                    system, user, temperature=0.0, max_tokens=4096
                )
                if not isinstance(structured_vocab, dict):
                    structured_vocab = {}
            except LLMError as exc:
                notes.append(f"第{lesson}课单词文本清洗失败：{exc}")
        vocab = _parse_vision_vocab(
            structured_vocab,
            lesson,
            vocab_page,
            source_file=vocab_source_file,
            source_kind=vocab_source_kind,
            book_page=info.get("workbook_book_page") if use_workbook else None,
        )

        # ---- grammar: main textbook lesson page ----
        grammar: list[dict[str, Any]] = []
        grammar_ocr_lines: list[str] = []
        main_page = main_pages[0] if main_pages else None
        if main_page:
            try:
                grammar_png, grammar_image = _render_png(main_pdf, main_page)
                grammar_ocr_lines = _ocr_lines(grammar_image)
                structured_grammar: dict[str, Any] = {}
                if vision:
                    try:
                        system, user = _vision_prompt_grammar(lesson, main_page)
                        structured_grammar = llm.complete_vision_json(
                            system, user, grammar_png, max_tokens=16000
                        )
                        if not isinstance(structured_grammar, dict):
                            structured_grammar = {}
                    except (VisionUnavailable, LLMError) as exc:
                        notes.append(f"第{lesson}课语法视觉 OCR 失败：{exc}")
                if not structured_grammar and text_llm and grammar_ocr_lines:
                    try:
                        system, user = _text_cleanup_prompt(lesson, grammar_ocr_lines)
                        structured_grammar = llm.complete_json(
                            system, user, temperature=0.0, max_tokens=4096
                        )
                        if not isinstance(structured_grammar, dict):
                            structured_grammar = {}
                    except LLMError as exc:
                        notes.append(f"第{lesson}课语法文本清洗失败：{exc}")
                grammar = _parse_vision_grammar(
                    structured_grammar, lesson, main_page
                )
            except Exception as exc:
                notes.append(f"第{lesson}课主教材页 PDF第{main_page}页渲染失败：{exc}")

        total_vocab += len(vocab)
        grammar_points.extend(grammar)
        grammar_raw_by_lesson[str(lesson)] = grammar_ocr_lines
        vocab_draft_lessons.append(
            {
                "lesson": lesson,
                "course_lesson": lesson - 36,
                "source_pages": [vocab_page],
                "grammar_page": main_page,
                "vocab": vocab,
                "raw_ocr": vocab_ocr_lines,
                "reviewed": False,
                "notes": (
                    "视觉 LLM 结构化结果，需人工审核"
                    if vision and structured_vocab
                    else "本地 OCR/文本清洗结果，需逐条人工审核"
                ),
            }
        )

    meta = {
        "semester": settings.active_semester.value,
        "source": "textbook_pdf_ocr",
        "generated_at": now_iso(),
        "reviewed": False,
        "vision_used": vision,
        "text_llm_used": text_llm,
        "source_files": [MAIN_PDF_NAME, WORKBOOK_PDF_NAME],
        "notes": notes
        + [
            "OCR 结果一律为 draft，不得直接用于生成。",
            "请用 /textbook review 逐课审核；审核后才会并入 vocab.json/grammar.json。",
            "44-50 课在审核前会被生成器明确拒绝。",
        ],
    }
    vocab_payload = {"meta": meta, "lessons": vocab_draft_lessons}
    grammar_payload = {
        "meta": meta,
        "points": grammar_points,
        "raw_ocr_by_lesson": grammar_raw_by_lesson,
    }
    settings.drafts_dir.mkdir(parents=True, exist_ok=True)
    vocab_draft.write_text(
        json.dumps(vocab_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    grammar_draft.write_text(
        json.dumps(grammar_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "status": "extracted",
        "lessons": len(vocab_draft_lessons),
        "vocab_entries": total_vocab,
        "grammar_points": len(grammar_points),
        "vision_used": vision,
        "files": [str(vocab_draft), str(grammar_draft)],
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 不是 JSON 对象")
    return data


def merge_approved_drafts(settings: Settings, *, approve_all: bool = False) -> dict[str, Any]:
    """Merge human-reviewed OCR drafts into vocab.json / grammar.json.

    ``approve_all`` is intended for a tool wrapper after the human has inspected
    the drafts; it is never called during init.
    """
    vocab_draft_path = settings.drafts_dir / "vocab_44_50.draft.json"
    grammar_draft_path = settings.drafts_dir / "grammar_44_50.draft.json"
    if not vocab_draft_path.exists():
        return {"status": "missing", "path": str(vocab_draft_path)}
    draft = _load_json(vocab_draft_path)
    if approve_all:
        draft["meta"]["reviewed"] = True
        for lesson in draft.get("lessons", []):
            lesson["reviewed"] = True
            for entry in lesson.get("vocab", []):
                entry["reviewed"] = True

    vocab_path = settings.vocab_path
    if not vocab_path.exists():
        return {"status": "missing", "path": str(vocab_path)}
    vocab_data = _load_json(vocab_path)
    lessons_by_number = {
        int(lesson.get("lesson")): lesson for lesson in vocab_data.get("lessons", [])
    }
    merged = 0
    for draft_lesson in draft.get("lessons", []):
        if not draft_lesson.get("reviewed"):
            continue
        lesson_no = int(draft_lesson.get("lesson"))
        lesson = lessons_by_number.get(lesson_no)
        if lesson is None:
            lesson = {
                "lesson": lesson_no,
                "course_lesson": lesson_no - 36,
                "title": None,
                "vocab": [],
            }
            vocab_data.setdefault("lessons", []).append(lesson)
            lessons_by_number[lesson_no] = lesson
        existing = {entry.get("entry_id"): entry for entry in lesson.get("vocab", [])}
        for entry in draft_lesson.get("vocab", []):
            if not entry.get("reviewed"):
                continue
            entry = dict(entry)
            entry["reviewed"] = True
            entry["source"] = dict(entry.get("source") or {})
            entry["source"]["kind"] = "pdf_reviewed"
            entry["source"]["reviewed_at"] = now_iso()
            existing[entry.get("entry_id")] = entry
            merged += 1
        lesson["vocab"] = sorted(existing.values(), key=lambda e: e.get("entry_id", ""))

    vocab_data.setdefault("meta", {})["updated_at"] = now_iso()
    vocab_data["meta"]["ocr_merged_entries"] = int(vocab_data["meta"].get("ocr_merged_entries", 0)) + merged
    lesson_numbers = sorted(
        int(lesson.get("lesson"))
        for lesson in vocab_data.get("lessons", [])
        if lesson.get("lesson") is not None
    )
    if lesson_numbers:
        vocab_data["meta"]["lesson_range"] = [min(lesson_numbers), max(lesson_numbers)]
    vocab_data["meta"]["count"] = sum(
        len(lesson.get("vocab", [])) for lesson in vocab_data.get("lessons", [])
    )
    vocab_path.write_text(json.dumps(vocab_data, ensure_ascii=False, indent=2), encoding="utf-8")

    grammar_merged = 0
    if grammar_draft_path.exists():
        gdraft = _load_json(grammar_draft_path)
        if approve_all:
            gdraft["meta"]["reviewed"] = True
            for point in gdraft.get("points", []):
                point["reviewed"] = True
                point["status"] = RuleStatus.APPROVED.value
        if gdraft.get("meta", {}).get("reviewed"):
            grammar_path = settings.grammar_path
            grammar_data = _load_json(grammar_path) if grammar_path.exists() else {"meta": {}, "points": []}
            existing_gp = {point.get("gp_id"): point for point in grammar_data.get("points", [])}
            for point in gdraft.get("points", []):
                if point.get("reviewed"):
                    existing_gp[point.get("gp_id")] = point
                    grammar_merged += 1
            grammar_data["points"] = list(existing_gp.values())
            grammar_data.setdefault("meta", {})["updated_at"] = now_iso()
            grammar_data["meta"]["approved_points"] = sum(
                1 for point in grammar_data["points"] if point.get("reviewed")
            )
            grammar_path.write_text(
                json.dumps(grammar_data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # Keep the reviewed flags in the drafts so /textbook review and the
    # coverage report reflect that these lessons were merged.
    if not draft.get("meta", {}).get("reviewed"):
        draft.setdefault("meta", {})["merged_at"] = now_iso()
    vocab_draft_path.write_text(
        json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if grammar_draft_path.exists():
        grammar_draft_path.write_text(
            json.dumps(gdraft, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {
        "status": "merged",
        "vocab_entries": merged,
        "grammar_points": grammar_merged,
        "vocab_path": str(vocab_path),
        "grammar_path": str(settings.grammar_path),
    }
