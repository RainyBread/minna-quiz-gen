"""Import and clean the legacy 37-43 corpus and the 2024 final exam.

The legacy project is treated as a read-only corpus.  Every imported item is
normalised into JLPTCraft's raw schema, with explicit quality flags and
quarantine status so downstream mining never silently trusts OCR/AI output.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from config import Settings
from core.models import (
    BLUEPRINT_2024,
    BLUEPRINT_JP1_NOTICE,
    Blueprint,
    GrammarFile,
    GrammarPoint,
    QualityStatus,
    QuestionType,
    RawBank,
    RawBankMeta,
    RawFlags,
    RawOption,
    RawQuestion,
    RuleStatus,
    SourceRef,
    TextbookVocab,
    VocabEntry,
    VocabLesson,
    contains_hiragana,
    contains_kanji,
    has_japanese,
    is_kana_only,
    is_katakana_only,
    is_reading_text,
    normalize_text,
    now_iso,
    sha256_text,
)

LEGACY_BANK = Path("public/japanese3-question-bank.json")
LEGACY_VOCAB_MD = Path("data/raw/japanese3/vocabulary-37-43.md")
LEGACY_JP3_BANK = Path("public/jp-3-question-bank.json")
LEGACY_2024_MD = Path("data/raw/japanese/2024年日语期末试卷.md")

STEM_PREFIX_TO_TYPE = {
    "假名 → 汉字": QuestionType.KANA_TO_KANJI,
    "汉字 → 假名": QuestionType.KANJI_TO_KANA,
    "假名 → 中文": QuestionType.TRANSLATION_CHOICE,
    "中文 → 假名": QuestionType.TRANSLATION_CHOICE,
}
SUBTYPE_TO_TYPE = {
    "kana-to-kanji": QuestionType.KANA_TO_KANJI,
    "kanji-to-kana": QuestionType.KANJI_TO_KANA,
}

EXAM_TYPE_RANGES: tuple[tuple[int, int, QuestionType], ...] = (
    (1, 30, QuestionType.KANJI_TO_KANA),
    (31, 40, QuestionType.KANA_TO_KANJI),
    (41, 45, QuestionType.VOCAB_CHOICE),
    (46, 55, QuestionType.EXPRESSION_CHOICE),
    (56, 80, QuestionType.GRAMMAR_FILL),
    (81, 85, QuestionType.SENTENCE_ORDERING),
    (86, 90, QuestionType.CLOZE),
    (91, 100, QuestionType.READING),
)

EXAM_SECTIONS = {
    QuestionType.KANJI_TO_KANA: "题组一：汉字读音选择",
    QuestionType.KANA_TO_KANJI: "题组二：画线词汉字选择",
    QuestionType.VOCAB_CHOICE: "题组三：词汇选择",
    QuestionType.EXPRESSION_CHOICE: "题组四：表达选择",
    QuestionType.GRAMMAR_FILL: "题组五：语法填空",
    QuestionType.SENTENCE_ORDERING: "题组六：句子排序",
    QuestionType.CLOZE: "题组七：完形填空",
    QuestionType.READING: "题组八：阅读理解",
}

OCR_MARKERS = (
    "【不清】",
    "OCR",
    "不确定",
    "（不清）",
    "(不清)",
    "题干前半",
    "图片",
    "裁切",
)
BLANK_RE = re.compile(r"（\s*　?\s*）")
EXAM_BLANK_RE = re.compile(r"【\d+】")
HEADWORD_RE = re.compile(r"^(.+?)（(.+?)）[—\-－](.+)$")
UNDERLINE_RE = re.compile(r"<u>(.*?)</u>", re.S)
BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)
TAG_RE = re.compile(r"<[^>]+>")
GROUP_RE = re.compile(r"第\s*(\d+)\s*课")
OPTION_RE = re.compile(r"^([A-D])[.．]\s*(.*)$")
ANSWER_RE = re.compile(r"正确答案[：:]\s*([A-D])")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)


def _existing_matches(path: Path, source_sha: str) -> bool:
    if not path.exists():
        return False
    try:
        data = _read_json(path)
    except Exception:
        return False
    meta = data.get("meta", {}) if isinstance(data, dict) else {}
    return meta.get("source_sha256") == source_sha


def _clean_text(text: str) -> str:
    return TAG_RE.sub("", text or "").replace("**", "").strip()


def _add_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _count_blanks(stem: str) -> int:
    return len(BLANK_RE.findall(stem)) + len(EXAM_BLANK_RE.findall(stem))


def _has_ocr_marker(text: str) -> bool:
    return any(marker in (text or "") for marker in OCR_MARKERS)


def _parse_headword(value: str) -> tuple[str, str, str]:
    match = HEADWORD_RE.match(value or "")
    if not match:
        return "", "", ""
    return match.group(1).strip(), match.group(2).strip(), match.group(3).strip()


def _answer_for_type(
    qtype: QuestionType, option_map: dict[str, str], answer_key: str | None
) -> str:
    if answer_key and answer_key in option_map:
        return option_map[answer_key]
    return ""


# ---------------------------------------------------------------------------
# Legacy 738-question bank
# ---------------------------------------------------------------------------


def import_legacy_vocab_bank(settings: Settings, force: bool = False) -> dict[str, Any]:
    source_path = settings.legacy_root / LEGACY_BANK
    if not source_path.exists():
        return {"status": "missing", "path": str(source_path)}
    source_sha = _file_sha(source_path)
    output_path = settings.raw_legacy_path
    if not force and _existing_matches(output_path, source_sha):
        return {"status": "skipped", "path": str(output_path), "reason": "source hash unchanged"}

    data = _read_json(source_path)
    if not isinstance(data, list):
        raise ValueError(f"{source_path} 不是预期的 JSON 数组")

    rel_source = _rel(source_path, settings.project_root)
    questions: list[RawQuestion] = []
    stats: dict[str, Any] = {
        "total": len(data),
        "imported": 0,
        "quarantined": 0,
        "by_type": {},
        "answer_key_mismatch": 0,
        "duplicate_options": 0,
        "ocr_unclear": 0,
        "target_no_kanji": 0,
        "multi_blank": 0,
    }

    for index, item in enumerate(data):
        qid = str(item.get("id") or f"legacy-{index + 1:04d}")
        stem = str(item.get("stem") or "").strip()
        prefix_match = re.match(r"^([^｜|]+)[｜|]\s*(.*)$", stem)
        prefix = prefix_match.group(1).strip() if prefix_match else ""
        basis = "stem_prefix"
        qtype = STEM_PREFIX_TO_TYPE.get(prefix)
        if qtype is None:
            qtype = SUBTYPE_TO_TYPE.get(str(item.get("subType") or ""))
            basis = "subtype"
        if qtype is None:
            qtype = QuestionType.TRANSLATION_CHOICE
            basis = "fallback"

        group_title = str(item.get("groupTitle") or "")
        group_match = GROUP_RE.search(group_title)
        textbook_lesson = int(group_match.group(1)) if group_match else None
        lesson = textbook_lesson - 36 if textbook_lesson else None

        options: list[RawOption] = []
        for opt in item.get("options", []) or []:
            key = str(opt.get("key") or "").strip().upper()
            text = _clean_text(str(opt.get("text") or ""))
            if key:
                options.append(RawOption(key=key, text=text))
        answer_key = str(item.get("answerKey") or "").strip().upper() or None
        option_map = {opt.key: opt.text for opt in options}
        answer_text = str(item.get("answerText") or "").strip()
        if answer_key and answer_key in option_map:
            expected = normalize_text(option_map[answer_key])
            actual = normalize_text(answer_text)
            answer_key_mismatch = bool(actual and expected != actual)
        else:
            answer_key_mismatch = True

        written, kana, meaning = _parse_headword(str(item.get("headword") or ""))
        target = written or (answer_text if qtype == QuestionType.KANA_TO_KANJI else "")
        target_has_kanji = contains_kanji(target)
        duplicate_texts = _duplicate_option_texts(options)

        flags = RawFlags()
        if answer_key_mismatch:
            flags.answer_key_mismatch = True
            stats["answer_key_mismatch"] += 1
        if duplicate_texts:
            flags.duplicate_option_texts = duplicate_texts
            stats["duplicate_options"] += 1
        missing_keys = [k for k in "ABCD" if k not in option_map]
        if missing_keys:
            flags.missing_option_keys = missing_keys
        ocr_texts = [
            f"{opt.key}:{opt.text}"
            for opt in options
            if _has_ocr_marker(opt.text) or _has_ocr_marker(stem)
        ]
        if ocr_texts:
            flags.ocr_unclear = ocr_texts
            stats["ocr_unclear"] += 1
        blanks = _count_blanks(stem)
        if blanks:
            flags.multi_blank_count = blanks
            stats["multi_blank"] += 1
        if qtype in {QuestionType.KANJI_TO_KANA, QuestionType.KANA_TO_KANJI} and not target_has_kanji:
            flags.script_mismatch = True
            flags.notes.append("target_no_kanji")
            stats["target_no_kanji"] += 1
        if qtype == QuestionType.KANJI_TO_KANA and any(
            not is_reading_text(opt.text) for opt in options
        ):
            flags.script_mismatch = True
            flags.notes.append("kanji_to_kana_options_not_kana")
        if qtype == QuestionType.KANA_TO_KANJI and not any(
            contains_kanji(opt.text) for opt in options
        ):
            flags.script_mismatch = True
            flags.notes.append("kana_to_kanji_options_no_kanji")
        if qtype != QuestionType.TRANSLATION_CHOICE and any(
            not has_japanese(opt.text) for opt in options
        ):
            flags.non_japanese_options = [
                f"{opt.key}:{opt.text}" for opt in options if not has_japanese(opt.text)
            ]

        status = (
            QualityStatus.QUARANTINED
            if flags.has_blocking_issues
            else QualityStatus.IMPORTED
        )
        if status == QualityStatus.QUARANTINED:
            stats["quarantined"] += 1
        else:
            stats["imported"] += 1
        stats["by_type"][qtype.value] = stats["by_type"].get(qtype.value, 0) + 1

        source = SourceRef(
            source_id=f"legacy-{qid}",
            file=rel_source,
            page=(item.get("source") or {}).get("pages", [None])[0]
            if isinstance(item.get("source"), dict)
            else None,
            section=group_title or None,
            question_no=item.get("numberInGroup"),
            content_hash=source_sha,
            legacy_id=qid,
            status=str(item.get("status") or ""),
            answer_provenance=str(item.get("answerProvenance") or ""),
            kind="legacy_vocab",
            target_has_kanji=target_has_kanji,
            type_basis=basis,
        )

        questions.append(
            RawQuestion(
                raw_id=f"legacy-{qid}",
                type=qtype,
                lesson=lesson,
                textbook_lesson=textbook_lesson,
                stem=stem,
                target=target or None,
                options=options,
                answer_key=answer_key,
                answer_text=answer_text or None,
                explanation=_clean_text(str(item.get("explanation") or "")),
                translation=str(item.get("translation") or ""),
                source=source,
                flags=flags,
                quality={
                    "legacy_status": item.get("status"),
                    "answer_provenance": item.get("answerProvenance"),
                    "subType": item.get("subType"),
                    "headword": item.get("headword"),
                    "meaning_zh": meaning,
                },
                status=status,
            )
        )

    bank = RawBank(
        meta=RawBankMeta(
            semester=settings.active_semester.value,
            source="legacy_vocab_bank",
            textbook="大家的日本语 初级2",
            lesson_range=[37, 43],
            quality=stats,
            generated_from=[rel_source],
            source_sha256=source_sha,
            generated_at=now_iso(),
            notes=[
                "738 道旧题库已按 stem 前缀重新分类；假名↔中文题标记为 translation_choice 记录型。",
                "target_no_kanji / duplicate_options / answer_key_mismatch 等已隔离，不参与干净证据。",
            ],
        ),
        questions=questions,
    )
    _write_json(output_path, bank.model_dump(mode="json"))
    stats["status"] = "imported"
    stats["path"] = str(output_path)
    return stats


def _duplicate_option_texts(options: Iterable[RawOption]) -> list[str]:
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for opt in options:
        key = normalize_text(opt.text)
        if not key:
            continue
        seen[key] = seen.get(key, 0) + 1
    for text, count in seen.items():
        if count > 1:
            duplicates.append(text)
    return duplicates


# ---------------------------------------------------------------------------
# 2024 final exam (raw Markdown)
# ---------------------------------------------------------------------------


def _exam_type_for_no(number: int) -> QuestionType:
    for start, end, qtype in EXAM_TYPE_RANGES:
        if start <= number <= end:
            return qtype
    return QuestionType.READING


def _parse_2024_blocks(text: str) -> list[dict[str, Any]]:
    """Split the exam Markdown into question blocks with attached passages."""
    lines = text.splitlines()
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    passage_lines: list[str] = []
    in_passage = False

    def finish() -> None:
        nonlocal current
        if current is not None:
            blocks.append(current)
            current = None

    for line in lines:
        question_match = re.match(r"^###\s*第(\d+)题\s*$", line)
        if question_match:
            finish()
            current = {
                "no": int(question_match.group(1)),
                "lines": [],
                "passage": "\n".join(passage_lines).strip(),
            }
            in_passage = False
            continue
        if re.match(r"^###\s*文章", line) or re.match(r"^##\s*题组七[：:]", line):
            finish()
            passage_lines = []
            in_passage = True
            continue
        if re.match(r"^##\s", line) or re.match(r"^###\s", line):
            finish()
            in_passage = False
            continue
        if current is not None:
            current["lines"].append(line)
        elif in_passage:
            passage_lines.append(line)
    finish()
    return blocks


def _parse_2024_question(
    block: dict[str, Any], source_sha: str, rel_source: str
) -> RawQuestion:
    number = int(block["no"])
    qtype = _exam_type_for_no(number)
    lines: list[str] = list(block.get("lines") or [])
    passage = str(block.get("passage") or "").strip()

    answer_split = None
    for index, line in enumerate(lines):
        if line.strip().startswith("#### 答案与解析"):
            answer_split = index
            break
    if answer_split is None:
        question_lines = lines
        answer_lines: list[str] = []
    else:
        question_lines = lines[:answer_split]
        answer_lines = lines[answer_split + 1 :]

    stem_lines: list[str] = []
    options: list[RawOption] = []
    current_opt: dict[str, Any] | None = None
    for line in question_lines:
        stripped = line.strip()
        if stripped.startswith("#### 题目"):
            continue
        opt_match = OPTION_RE.match(stripped)
        if opt_match:
            current_opt = {"key": opt_match.group(1), "text": opt_match.group(2).strip()}
            options.append(RawOption(key=current_opt["key"], text=current_opt["text"]))
            continue
        if current_opt is not None:
            if stripped and not stripped.startswith("#") and not stripped.startswith(">"):
                current_opt["text"] = f"{current_opt['text']} {stripped}".strip()
                options[-1] = RawOption(key=current_opt["key"], text=current_opt["text"])
            continue
        if stripped and not stripped.startswith(">") and not stripped.startswith("####"):
            stem_lines.append(line)

    raw_stem = "\n".join(stem_lines)
    underline_match = UNDERLINE_RE.search(raw_stem)
    bold_match = BOLD_RE.search(raw_stem)
    target_markup = (
        underline_match.group(1)
        if underline_match
        else (bold_match.group(1) if bold_match else "")
    )
    underlined = _clean_text(target_markup)
    stem = _clean_text(raw_stem)
    answer_block = "\n".join(answer_lines)
    answer_match = ANSWER_RE.search(answer_block)
    answer_key = answer_match.group(1).upper() if answer_match else None
    option_map = {opt.key: opt.text for opt in options}
    answer_text = _answer_for_type(qtype, option_map, answer_key)

    if qtype == QuestionType.KANA_TO_KANJI:
        target = answer_text or underlined
    else:
        target = underlined or answer_text
    target_has_kanji = contains_kanji(target)

    flags = RawFlags()
    ocr_items = [
        f"{key}:{text}"
        for key, text in option_map.items()
        if _has_ocr_marker(text)
    ]
    if ocr_items or _has_ocr_marker(stem):
        flags.ocr_unclear = ocr_items or ["stem"]
    duplicates = _duplicate_option_texts(options)
    if duplicates:
        flags.duplicate_option_texts = duplicates
    missing_keys = [key for key in "ABCD" if key not in option_map]
    if missing_keys:
        flags.missing_option_keys = missing_keys
    if answer_key is None or answer_key not in option_map:
        flags.answer_key_mismatch = True
    if answer_key in option_map and _has_ocr_marker(option_map[answer_key]):
        flags.ocr_unclear = list(dict.fromkeys(flags.ocr_unclear + [f"answer:{answer_key}"]))
    blanks = _count_blanks(stem)
    if blanks > 1:
        flags.multi_blank_count = blanks
    if qtype == QuestionType.KANJI_TO_KANA and not target_has_kanji:
        flags.script_mismatch = True
        flags.notes.append("target_no_kanji")
    if qtype == QuestionType.KANJI_TO_KANA and any(
        not is_reading_text(opt.text) for opt in options
    ):
        flags.script_mismatch = True
        flags.notes.append("kanji_to_kana_options_not_kana")
    if qtype == QuestionType.KANA_TO_KANJI and not any(
        contains_kanji(opt.text) for opt in options
    ):
        flags.script_mismatch = True
        flags.notes.append("kana_to_kanji_options_no_kanji")
    if qtype in {QuestionType.CLOZE, QuestionType.READING} and not passage:
        flags.notes.append("passage_missing")
    if number == 13:
        flags.notes.append("ocr_missing_q13")

    status = (
        QualityStatus.QUARANTINED
        if flags.has_blocking_issues
        else QualityStatus.IMPORTED
    )

    explanation = _clean_text(answer_block)
    return RawQuestion(
        raw_id=f"exam2024-q{number:03d}",
        type=qtype,
        lesson=None,
        textbook_lesson=None,
        stem=stem,
        target=target or None,
        options=options,
        answer_key=answer_key,
        answer_text=answer_text or None,
        explanation=explanation,
        translation="",
        passage=passage or None,
        source=SourceRef(
            source_id=f"exam2024-q{number:03d}",
            file=rel_source,
            section=EXAM_SECTIONS.get(qtype, ""),
            question_no=number,
            content_hash=source_sha,
            kind="2024_exam",
            exam_no=number,
            target_has_kanji=target_has_kanji,
            type_basis="section_range",
        ),
        flags=flags,
        quality={
            "exam_no": number,
            "raw_answer_block": answer_block[:2000],
            "has_passage": bool(passage),
        },
        status=status,
    )


def import_2024_exam(settings: Settings, force: bool = False) -> dict[str, Any]:
    source_path = settings.legacy_root / LEGACY_2024_MD
    if not source_path.exists():
        return {"status": "missing", "path": str(source_path)}
    source_sha = _file_sha(source_path)
    output_path = settings.raw_exams_path
    if not force and _existing_matches(output_path, source_sha):
        return {"status": "skipped", "path": str(output_path), "reason": "source hash unchanged"}

    text = source_path.read_text(encoding="utf-8", errors="ignore")
    rel_source = _rel(source_path, settings.project_root)
    blocks = _parse_2024_blocks(text)
    questions = [_parse_2024_question(block, source_sha, rel_source) for block in blocks]
    questions.sort(key=lambda q: q.source.exam_no or 0)

    stats: dict[str, Any] = {
        "total_blocks": len(questions),
        "imported": sum(1 for q in questions if q.status == QualityStatus.IMPORTED),
        "quarantined": sum(1 for q in questions if q.status == QualityStatus.QUARANTINED),
        "by_type": {},
        "ocr_unclear": sum(1 for q in questions if q.flags.ocr_unclear),
        "duplicate_options": sum(1 for q in questions if q.flags.duplicate_option_texts),
        "answer_missing": sum(1 for q in questions if not q.answer_key),
        "multi_blank": sum(1 for q in questions if q.flags.multi_blank_count > 1),
        "missing_q13": not any(q.source.exam_no == 13 and q.status == QualityStatus.IMPORTED for q in questions),
        "record_only": sum(1 for q in questions if not q.type.is_generatable),
    }
    for question in questions:
        stats["by_type"][question.type.value] = stats["by_type"].get(question.type.value, 0) + 1

    bank = RawBank(
        meta=RawBankMeta(
            semester=settings.active_semester.value,
            source="exam_2024",
            textbook="大家的日本语 初级2（37-50课期末）",
            lesson_range=[37, 50],
            quality=stats,
            generated_from=[rel_source],
            source_sha256=source_sha,
            generated_at=now_iso(),
            notes=[
                "2024 大二上期末真题，官方 100 题；第13题 OCR 不可用，实际可用题数见 quality.imported。",
                "题号区间决定题型；排序/完形/阅读仅记录配比，首版不生成。",
                "所有 OCR 不清、选项重复、答案键缺失的题已隔离，不进入干净证据。",
            ],
        ),
        questions=questions,
    )
    _write_json(output_path, bank.model_dump(mode="json"))
    stats["status"] = "imported"
    stats["path"] = str(output_path)
    return stats


# ---------------------------------------------------------------------------
# Textbook vocabulary (37-43, parsed from the legacy vocabulary Markdown)
# ---------------------------------------------------------------------------


def import_textbook_vocab(settings: Settings, force: bool = False) -> dict[str, Any]:
    source_path = settings.legacy_root / LEGACY_VOCAB_MD
    if not source_path.exists():
        return {"status": "missing", "path": str(source_path)}
    source_sha = _file_sha(source_path)
    output_path = settings.vocab_path
    if not force and _existing_matches(output_path, source_sha):
        return {"status": "skipped", "path": str(output_path), "reason": "source hash unchanged"}

    text = source_path.read_text(encoding="utf-8", errors="ignore")
    rel_source = _rel(source_path, settings.project_root)
    section_pattern = re.compile(r"^##\s*第(\d+)课\s*$", re.M)
    sections = list(section_pattern.finditer(text))
    by_lesson: dict[int, list[VocabEntry]] = {}
    bad_json = 0
    duplicate_written: dict[str, set[str]] = {}

    for index, match in enumerate(sections):
        lesson = int(match.group(1))
        body_start = match.end()
        body_end = sections[index + 1].start() if index + 1 < len(sections) else len(text)
        body = text[body_start:body_end]
        for entry_match in re.finditer(r"^词条：(\{.*\})$", body, re.M):
            try:
                data = json.loads(entry_match.group(1))
            except Exception:
                bad_json += 1
                continue
            written = str(data.get("written") or "").strip()
            kana = str(data.get("kana") or "").strip()
            if not written or not kana:
                bad_json += 1
                continue
            entry_lesson = int(data.get("lesson") or lesson)
            entry_id = str(data.get("entryId") or f"w{entry_lesson}-{int(data.get('number') or 0):03d}")
            confidence = str(data.get("confidence") or "medium")
            if confidence not in {"high", "medium", "low"}:
                confidence = "medium"
            duplicate_written.setdefault(written, set()).add(kana)
            entry = VocabEntry(
                entry_id=entry_id,
                lesson=entry_lesson,
                course_lesson=entry_lesson - 36,
                written=written,
                kana=kana,
                translation_zh=str(data.get("translation") or ""),
                pos=str(data.get("partOfSpeech") or ""),
                notes=str(data.get("notes") or ""),
                confidence=confidence,
                source={
                    "file": rel_source,
                    "page": data.get("sourcePage"),
                    "kind": "textbook_vocab_md",
                    "legacy_entry": data,
                },
                has_kanji=contains_kanji(written),
                is_loanword=is_katakana_only(written),
                unidic_verified=None,
                uncertainty=str(data.get("uncertainty") or ""),
                reviewed=True,
            )
            by_lesson.setdefault(entry_lesson, []).append(entry)

    lessons = [
        VocabLesson(
            lesson=lesson,
            course_lesson=lesson - 36,
            title=None,
            vocab=sorted(entries, key=lambda e: e.entry_id),
        )
        for lesson, entries in sorted(by_lesson.items())
    ]
    ambiguous = {
        written: sorted(readings)
        for written, readings in duplicate_written.items()
        if len(readings) > 1
    }
    payload = TextbookVocab(
        meta={
            "semester": settings.active_semester.value,
            "source": "vocabulary-37-43.md",
            "source_sha256": source_sha,
            "generated_from": [rel_source],
            "generated_at": now_iso(),
            "count": sum(len(lesson.vocab) for lesson in lessons),
            "lesson_range": [min(by_lesson), max(by_lesson)] if by_lesson else [],
            "bad_json": bad_json,
            "ambiguous_written_readings": ambiguous,
            "notes": [
                "37-43 课已审核词条，来自旧项目教材词条 OCR。",
                "written 相同但读音不同的条目保留，生成时由上下文/UniDic 决定；之后会作为硬冲突回归。",
            ],
        },
        lessons=lessons,
    )
    _write_json(output_path, payload.model_dump(mode="json"))
    return {
        "status": "imported",
        "path": str(output_path),
        "lessons": len(lessons),
        "entries": payload.meta["count"],
        "bad_json": bad_json,
        "ambiguous_written_readings": len(ambiguous),
    }


# ---------------------------------------------------------------------------
# Grammar reference
# ---------------------------------------------------------------------------

PATTERN_HINTS: dict[str, str] = {
    "jp-3-q001": "～てある",
    "jp-3-q002": "～たそうな",
    "jp-3-q003": "～ように（可能形）",
    "jp-3-q004": "ために／ように",
    "jp-3-q005": "か／かどうか",
    "jp-3-q006": "のに",
    "jp-3-q007": "から／ので",
    "jp-3-q008": "可能形",
    "jp-3-q009": "～ように（可能形）",
    "jp-3-q010": "なる／する",
    "jp-3-q011": "～てもらう",
    "jp-3-q012": "から／ので",
    "jp-3-q013": "～になる",
    "jp-3-q014": "～かどうか",
    "jp-3-q015": "なくす／なくなる",
    "jp-3-q016": "ので",
    "jp-3-q017": "～てくれる",
    "jp-3-q018": "の／こと",
    "jp-3-q019": "～てもらう／ていただく",
    "jp-3-q020": "～そうだ（様態）",
    "jp-3-q021": "～ように",
}

GRAMMAR_ALLOWED_RULES = [
    "particle_error",
    "conjugation_error",
    "tense_aspect_error",
    "voice_error",
    "collocation_error",
    "situational_error",
]


def import_grammar_reference(settings: Settings, force: bool = False) -> dict[str, Any]:
    output_path = settings.grammar_path
    source_paths = [
        settings.legacy_root / LEGACY_JP3_BANK,
        settings.legacy_root / LEGACY_2024_MD,
    ]
    existing = [path for path in source_paths if path.exists()]
    if not existing:
        return {"status": "missing", "paths": [str(p) for p in source_paths]}
    combined_sha = sha256_text(
        "\n".join(f"{_file_sha(path)}:{path.name}" for path in existing)
    )
    if not force and _existing_matches(output_path, combined_sha):
        return {"status": "skipped", "path": str(output_path), "reason": "source hash unchanged"}

    points: list[GrammarPoint] = []
    jp3_path = settings.legacy_root / LEGACY_JP3_BANK
    if jp3_path.exists():
        rel = _rel(jp3_path, settings.project_root)
        data = _read_json(jp3_path)
        singles = [q for q in data if q.get("questionType") == "single"]
        for question in singles:
            qid = str(question.get("id") or "")
            answer_text = str(question.get("answerText") or "").strip()
            stem = _clean_text(str(question.get("stem") or ""))
            explanation = _clean_text(str(question.get("explanation") or ""))
            meaning = str(question.get("translation") or "").strip()
            if not meaning:
                meaning = explanation.splitlines()[0][:120] if explanation else ""
            points.append(
                GrammarPoint(
                    gp_id=f"g-jp3-{qid.split('-')[-1]}" if qid else f"g-jp3-{len(points)+1:03d}",
                    lesson=None,
                    pattern_ja=PATTERN_HINTS.get(qid, ""),
                    meaning_zh=meaning,
                    formation="",
                    examples=[stem],
                    source_refs=[
                        {
                            "file": rel,
                            "question_id": qid,
                            "answer_key": question.get("answerKey"),
                            "answer_text": answer_text,
                        }
                    ],
                    approved_answer_forms=[answer_text] if answer_text else [],
                    allowed_distractor_rule_ids=GRAMMAR_ALLOWED_RULES,
                    confidence="medium",
                    status=RuleStatus.APPROVED,
                    reviewed=True,
                )
            )

    exam_2024_path = settings.legacy_root / LEGACY_2024_MD
    if exam_2024_path.exists():
        rel_exam = _rel(exam_2024_path, settings.project_root)
        exam_sha = _file_sha(exam_2024_path)
        text = exam_2024_path.read_text(encoding="utf-8", errors="ignore")
        for block in _parse_2024_blocks(text):
            number = int(block["no"])
            if not 56 <= number <= 80:
                continue
            question = _parse_2024_question(block, exam_sha, rel_exam)
            if question.status != QualityStatus.IMPORTED or not question.answer_text:
                continue
            if question.flags.multi_blank_count > 1 or question.flags.ocr_unclear:
                continue
            points.append(
                GrammarPoint(
                    gp_id=f"g-exam2024-q{number:03d}",
                    lesson=None,
                    pattern_ja="",
                    meaning_zh="",
                    formation="",
                    examples=[question.stem],
                    source_refs=[
                        {
                            "file": rel_exam,
                            "question_id": question.raw_id,
                            "answer_key": question.answer_key,
                            "answer_text": question.answer_text,
                        }
                    ],
                    approved_answer_forms=[question.answer_text],
                    allowed_distractor_rule_ids=GRAMMAR_ALLOWED_RULES,
                    confidence="low",
                    status=RuleStatus.DRAFT,
                    reviewed=False,
                )
            )

    payload = GrammarFile(
        meta={
            "semester": settings.active_semester.value,
            "source_sha256": combined_sha,
            "generated_from": [_rel(path, settings.project_root) for path in existing],
            "generated_at": now_iso(),
            "approved_points": sum(1 for p in points if p.reviewed),
            "draft_points": sum(1 for p in points if not p.reviewed),
            "notes": [
                "21 道 jp-3 干净单选题作为已审核语法参考（解析为 AI 生成，建议人工复核）。",
                "2024 语法填空作为 draft，需人工审核后才用于生成；不伪造课次。",
                "37-50 课语法课次未标注，生成时允许跨课检索并标注 warning。",
            ],
        },
        points=points,
    )
    _write_json(output_path, payload.model_dump(mode="json"))
    return {
        "status": "imported",
        "path": str(output_path),
        "approved": payload.meta["approved_points"],
        "draft": payload.meta["draft_points"],
    }


# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------


def import_exam_blueprint(settings: Settings, force: bool = False) -> dict[str, Any]:
    source_path = settings.legacy_root / LEGACY_2024_MD
    source_sha = _file_sha(source_path) if source_path.exists() else ""
    output_path = settings.blueprint_path
    if not force and _existing_matches(output_path, source_sha):
        return {"status": "skipped", "path": str(output_path)}
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
        note=(
            "主 blueprint 采用 2024 大二上期末真题的题组题号分布；"
            "第13题 OCR 不可用，语料统计按 99 题。综合日语1通知不是大二上正式考纲。"
        ),
    )
    payload = blueprint.model_dump(mode="json")
    payload["meta"] = {
        "semester": settings.active_semester.value,
        "source_sha256": source_sha,
        "generated_from": [_rel(source_path, settings.project_root)] if source_path.exists() else [],
        "generated_at": now_iso(),
        "reviewed": False,
    }
    _write_json(output_path, payload)
    return {"status": "imported", "path": str(output_path), "counts": BLUEPRINT_2024}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def ensure_corpus(settings: Settings, force: bool = False) -> dict[str, Any]:
    """Import every available local corpus and return a compact summary."""
    if not settings.raw_mock_path.exists():
        raise FileNotFoundError(
            f"缺少 5 道 mock：{settings.raw_mock_path}；请从项目种子文件恢复。"
        )
    summary: dict[str, Any] = {
        "mock": len((_read_json(settings.raw_mock_path) or {}).get("questions", [])),
        "legacy": import_legacy_vocab_bank(settings, force=force),
        "exam_2024": import_2024_exam(settings, force=force),
        "vocab": import_textbook_vocab(settings, force=force),
        "grammar": import_grammar_reference(settings, force=force),
        "blueprint": import_exam_blueprint(settings, force=force),
    }
    return summary
