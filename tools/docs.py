#!/usr/bin/env python
"""Documentation index and lint for JLPTCraft.

Only the Python standard library is used.  The tool is deliberately
deterministic: it never calls an LLM, never executes documented commands and
never scans for secrets (that is ``tools/check_repo_hygiene.py``'s job).

Commands::

    python tools/docs.py index            # regenerate docs/README.md + CLAUDE index block
    python tools/docs.py index --check    # fail if either generated block is stale
    python tools/docs.py lint             # structural docs checks
    python tools/docs.py lint --json      # machine-readable report
    python tools/docs.py lint --simulate-missing "20-模块/24-装配审计与降级.md"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CLAUDE = ROOT / "CLAUDE.md"
DOCS_README = DOCS / "README.md"

INDEX_BEGIN = "<!-- BEGIN docs-index -->"
INDEX_END = "<!-- END docs-index -->"

TIER_ORDER = ("L0", "L1", "L2", "L3", "记录")
TIER_DIRS = {
    "L0": DOCS / "00-总纲",
    "L1": DOCS / "10-契约",
    "L2": DOCS / "20-模块",
    "L3": DOCS / "30-操作",
    "记录": DOCS / "80-决策",
}
ARCHIVE_DIR = DOCS / "90-归档"

ALLOWED_KEYS = {
    "模块",
    "档次",
    "说明",
    "代码",
    "契约",
    "验收",
    "校验",
    "红线",
    "状态",
    "日期",
}
LIST_KEYS = {"代码", "红线"}
REQUIRED = {
    "L0": {"档次", "说明", "校验"},
    "L1": {"档次", "说明", "代码", "验收", "校验", "红线"},
    "L2": {"模块", "档次", "说明", "代码", "契约", "验收", "校验"},
    "L3": {"档次", "说明", "验收", "校验"},
    "记录": {"档次", "说明", "校验"},
}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CJK_RE = re.compile(r"[぀-ヿㇰ-ㇿ㐀-䶿一-鿿]")
LATIN_RE = re.compile(r"[A-Za-z]")
DRIFT_RE = re.compile(r"DRIFT-\d{2}")
DEFECT_RE = re.compile(r"\bE_[A-Z_]+\b")
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
ANCHOR_RE = re.compile(r"<a\s+(?:id|name)=\"([^\"]+)\"", re.I)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
FENCE_RE = re.compile(r"```.*?```", re.S)


@dataclass
class Doc:
    path: Path
    rel: str
    layer: str
    title: str
    meta: dict[str, Any]
    body: str
    text: str
    line_count: int

    @property
    def summary(self) -> str:
        return str(self.meta.get("说明") or self.title)

    @property
    def code(self) -> list[str]:
        value = self.meta.get("代码") or []
        if isinstance(value, str):
            return [value]
        return list(value)

    @property
    def contracts(self) -> str:
        return str(self.meta.get("契约") or "")

    @property
    def redlines(self) -> list[int]:
        raw = self.meta.get("红线") or []
        if isinstance(raw, str):
            raw = [raw]
        result: list[int] = []
        for item in raw:
            try:
                result.append(int(str(item).strip()))
            except ValueError:
                continue
        return result


@dataclass
class Finding:
    severity: str
    rule: str
    path: str
    line: int
    message: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    docs: list[Doc] = field(default_factory=list)
    routes: list[dict[str, Any]] = field(default_factory=list)
    links: int = 0
    simulated_refs: int = 0

    def fail(self, rule: str, path: str, line: int, message: str) -> None:
        self.findings.append(Finding("fail", rule, path, line, message))

    def warn(self, rule: str, path: str, line: int, message: str) -> None:
        self.findings.append(Finding("warn", rule, path, line, message))

    @property
    def failures(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == "fail"]

    @property
    def warnings(self) -> list[Finding]:
        return [item for item in self.findings if item.severity == "warn"]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _docs_rel(path: Path) -> str:
    return path.relative_to(DOCS).as_posix()


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def parse_front_matter(text: str) -> tuple[dict[str, Any], str, list[str]]:
    """Parse the constrained front-matter subset used by docs/."""
    errors: list[str] = []
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text, ["缺少 front-matter（首行必须是 ---）"]
    end = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            end = index
            break
    if end is None:
        return {}, text, ["front-matter 未闭合（缺少结束 ---）"]
    meta: dict[str, Any] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, sep, value = line.partition(":")
        key = key.strip()
        if not sep:
            errors.append(f"无法解析 front-matter 行：{line!r}")
            continue
        if key not in ALLOWED_KEYS:
            errors.append(f"未知 front-matter 字段：{key}")
            continue
        value = _strip_quotes(value)
        if key in LIST_KEYS:
            meta[key] = [item.strip() for item in value.split(",") if item.strip()]
        else:
            meta[key] = value
    body = "\n".join(lines[end + 1 :])
    return meta, body, errors


def discover_docs() -> tuple[list[Doc], list[str]]:
    errors: list[str] = []
    docs: list[Doc] = []
    for layer, directory in TIER_DIRS.items():
        if not directory.exists():
            errors.append(f"缺少目录：{_rel(directory)}")
            continue
        for path in sorted(directory.rglob("*.md")):
            if path.name == "README.md" and path.parent == directory:
                # Tier READMEs are not used; skip rather than treat as docs.
                continue
            text = _read(path)
            meta, body, parse_errors = parse_front_matter(text)
            for error in parse_errors:
                errors.append(f"{_docs_rel(path)}: {error}")
            if meta.get("档次") and meta["档次"] != layer:
                errors.append(
                    f"{_docs_rel(path)}: 档次 {meta['档次']!r} 与目录 {layer!r} 不一致"
                )
            title = ""
            for line in body.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break
            docs.append(
                Doc(
                    path=path,
                    rel=_docs_rel(path),
                    layer=layer,
                    title=title or path.stem,
                    meta=meta,
                    body=body,
                    text=text,
                    line_count=len(text.splitlines()),
                )
            )
    if ARCHIVE_DIR.exists():
        for path in sorted(ARCHIVE_DIR.rglob("*.md")):
            text = _read(path)
            meta, body, parse_errors = parse_front_matter(text)
            for error in parse_errors:
                errors.append(f"{_docs_rel(path)}: {error}")
            title = next(
                (line[2:].strip() for line in body.splitlines() if line.startswith("# ")),
                path.stem,
            )
            docs.append(
                Doc(
                    path=path,
                    rel=_docs_rel(path),
                    layer="记录",
                    title=title,
                    meta=meta,
                    body=body,
                    text=text,
                    line_count=len(text.splitlines()),
                )
            )
    return docs, errors


def is_exempt(doc: Doc) -> bool:
    return doc.rel.startswith("90-归档/") or doc.rel == "README.md"


def check_front_matter(report: Report, docs: Iterable[Doc]) -> None:
    for doc in docs:
        if is_exempt(doc):
            continue
        layer = str(doc.meta.get("档次") or doc.layer)
        if layer not in REQUIRED:
            report.fail("FM-001", doc.rel, 1, f"未知档次：{layer}")
            continue
        missing = sorted(REQUIRED[layer] - set(doc.meta))
        for key in missing:
            report.fail("FM-002", doc.rel, 1, f"缺少必填字段：{key}")
        updated = str(doc.meta.get("校验") or "")
        if not DATE_RE.fullmatch(updated):
            report.fail("FM-003", doc.rel, 1, f"校验 不是 YYYY-MM-DD：{updated!r}")
        if layer == "记录":
            status = str(doc.meta.get("状态") or "")
            if not status:
                report.fail("FM-004", doc.rel, 1, "ADR 缺少 状态")
        if doc.line_count > 250:
            report.fail("SIZE-001", doc.rel, 1, f"文档 {doc.line_count} 行 > 250")
        elif doc.line_count > 200:
            report.warn("SIZE-002", doc.rel, 1, f"文档 {doc.line_count} 行 > 200")


def _symbol_present(path: Path, symbol: str) -> bool:
    text = _read(path)
    name = symbol.split(".")[-1]
    patterns = (
        rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(name)}\b",
        rf"(?m)^\s*class\s+{re.escape(name)}\b",
        rf"(?m)^\s*{re.escape(name)}\s*(?::[^=\n]+)?=",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def check_code_paths(report: Report, docs: Iterable[Doc]) -> set[str]:
    covered: set[str] = set()
    for doc in docs:
        if is_exempt(doc):
            continue
        for entry in doc.code:
            raw = str(entry).strip().strip("`")
            if not raw:
                continue
            path_part, _, symbol = raw.partition("#")
            path_part = path_part.strip()
            if not path_part or path_part.startswith(("/", "\\")) or ".." in Path(path_part).parts:
                report.fail("PATH-001", doc.rel, 1, f"非法代码路径：{raw!r}")
                continue
            target = ROOT / path_part
            if not target.exists():
                report.fail("PATH-002", doc.rel, 1, f"代码路径不存在：{path_part}")
                continue
            covered.add(path_part.replace("\\", "/"))
            if symbol and target.is_file():
                if not _symbol_present(target, symbol.strip()):
                    report.fail(
                        "PATH-003",
                        doc.rel,
                        1,
                        f"符号不存在：{path_part}#{symbol.strip()}",
                    )
    return covered


def check_l2_sections(report: Report, docs: Iterable[Doc]) -> None:
    required = (
        "## 1. 职责与边界",
        "## 2. 输入输出契约",
        "## 3. 数据与依赖",
        "## 4. 本地不变量",
        "## 5. 失败与降级",
        "## 6. 扩展点",
        "## 7. 验收",
        "## 8. 已知缺陷",
    )
    for doc in docs:
        if doc.layer != "L2" or is_exempt(doc):
            continue
        for heading in required:
            if heading not in doc.body:
                report.fail("L2-001", doc.rel, 1, f"缺少小节：{heading}")


def check_coverage(report: Report, covered: set[str]) -> None:
    required: set[str] = set()
    core = ROOT / "core"
    if core.exists():
        for path in core.glob("*.py"):
            if path.name != "__init__.py":
                required.add(_rel(path))
    for name in ("main.py", "init.py", "quiz.py", "config.py"):
        if (ROOT / name).exists():
            required.add(name)
    for path in sorted(required - covered):
        report.fail("COVERAGE-001", path, 0, "未被任何文档的 代码: 字段覆盖")


def _resolve(base_dir: Path, target: str) -> Path:
    target = target.strip().strip("`")
    if target.startswith("docs/"):
        return ROOT / target
    return (base_dir / target).resolve()


def _anchor_exists(path: Path, anchor: str) -> bool:
    if not path.exists():
        return False
    return bool(ANCHOR_RE.search(_read(path)) and any(
        match.group(1) == anchor for match in ANCHOR_RE.finditer(_read(path))
    ))


def _strip_fences(text: str) -> str:
    return FENCE_RE.sub("", text)


def check_links(report: Report, docs: Iterable[Doc], simulate_missing: str) -> None:
    for doc in docs:
        body = _strip_fences(doc.body)
        for match in LINK_RE.finditer(body):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "tel:", "#")):
                if target.startswith("#"):
                    anchor = target[1:]
                    if not _anchor_exists(doc.path, anchor):
                        report.fail("LINK-002", doc.rel, 1, f"锚点不存在：#{anchor}")
                continue
            path_part, _, anchor = target.partition("#")
            if not path_part:
                if anchor and not _anchor_exists(doc.path, anchor):
                    report.fail("LINK-002", doc.rel, 1, f"锚点不存在：#{anchor}")
                continue
            resolved = _resolve(doc.path.parent, path_part)
            report.links += 1
            rel = _rel(resolved) if resolved.is_relative_to(ROOT) else str(resolved)
            simulated = simulate_missing and _docs_rel(resolved) == simulate_missing
            if simulated:
                report.simulated_refs += 1
            if not resolved.exists():
                if not simulated:
                    report.fail("LINK-001", doc.rel, 1, f"死链：{path_part}")
                continue
            if anchor and not _anchor_exists(resolved, anchor):
                report.fail("LINK-003", doc.rel, 1, f"目标锚点不存在：{path_part}#{anchor}")
            if simulated:
                report.fail("LINK-004", doc.rel, 1, f"模拟缺失仍有引用：{path_part}")
        # Contract references are relative to docs/, not to the file.
        if doc.contracts:
            target = DOCS / doc.contracts
            report.links += 1
            simulated = simulate_missing and doc.contracts == simulate_missing
            if simulated:
                report.simulated_refs += 1
            if not target.exists() and not simulated:
                report.fail("LINK-005", doc.rel, 1, f"契约不存在：{doc.contracts}")


def _claude_section(text: str, heading: str) -> str:
    match = re.search(rf"(?ms)^## {re.escape(heading)}\s*$", text)
    if not match:
        return ""
    rest = text[match.end() :]
    next_heading = re.search(r"(?m)^## ", rest)
    return rest[: next_heading.start()] if next_heading else rest


def check_claude(report: Report) -> None:
    if not CLAUDE.exists():
        report.fail("CLAUDE-001", "CLAUDE.md", 0, "CLAUDE.md 不存在")
        return
    text = _read(CLAUDE)
    line_count = len(text.splitlines())
    if line_count > 120:
        report.fail("CLAUDE-002", "CLAUDE.md", 1, f"{line_count} 行 > 120")
    for heading in ("项目定位", "项目红线", "文档规则", "任务路由", "文档索引"):
        if not re.search(rf"(?m)^## {re.escape(heading)}\s*$", text):
            report.fail("CLAUDE-003", "CLAUDE.md", 1, f"缺少小节：{heading}")
    fences = re.findall(r"```.*?```", text, re.S)
    for fence in fences:
        if len(fence.splitlines()) > 8:
            report.fail("CLAUDE-004", "CLAUDE.md", 1, "代码块超过 6 行正文")
    if INDEX_BEGIN not in text or INDEX_END not in text:
        report.fail("CLAUDE-005", "CLAUDE.md", 1, "缺少 docs-index 标记块")
    for match in LINK_RE.finditer(_strip_fences(text)):
        target = match.group(1).strip()
        if target.startswith(("http://", "https://", "mailto:", "tel:", "#")):
            continue
        path_part, _, anchor = target.partition("#")
        resolved = ROOT / path_part if path_part.startswith("docs/") else (CLAUDE.parent / path_part).resolve()
        if not resolved.exists():
            report.fail("CLAUDE-006", "CLAUDE.md", 1, f"死链：{path_part}")
        elif anchor and not _anchor_exists(resolved, anchor):
            report.fail("CLAUDE-007", "CLAUDE.md", 1, f"目标锚点不存在：{path_part}#{anchor}")


def _redline_lines(text: str) -> list[tuple[int, str]]:
    section = _claude_section(text, "项目红线")
    result: list[tuple[int, str]] = []
    for line in section.splitlines():
        match = re.match(r"^\s*(\d+)\.\s+(.*\S)\s*$", line)
        if match:
            result.append((int(match.group(1)), match.group(2)))
    return result


def check_redlines(report: Report, docs: list[Doc]) -> None:
    owners: dict[int, Doc] = {}
    for doc in docs:
        if doc.layer != "L1":
            continue
        for number in doc.redlines:
            if number < 1 or number > 12:
                report.fail("RED-001", doc.rel, 1, f"红线编号越界：{number}")
                continue
            if number in owners:
                report.fail(
                    "RED-002",
                    doc.rel,
                    1,
                    f"红线 {number} 已由 {owners[number].rel} 声明",
                )
                continue
            owners[number] = doc
            if f'<a id="redline-{number}"' not in doc.body:
                report.fail(
                    "RED-003", doc.rel, 1, f"缺少红线锚点：<a id=\"redline-{number}\">"
                )
    for number in range(1, 13):
        if number not in owners:
            report.fail("RED-004", "docs", 0, f"红线 {number} 没有唯一 owner")
    if not CLAUDE.exists():
        return
    text = _read(CLAUDE)
    lines = _redline_lines(text)
    if len(lines) != 12:
        report.fail("RED-005", "CLAUDE.md", 1, f"红线数量 {len(lines)} != 12")
    for number, line in lines:
        links = re.findall(r"\]\((docs/[^)]+)\)", line)
        if not links:
            report.fail("RED-006", "CLAUDE.md", 1, f"红线 {number} 缺少 docs 链接")
            continue
        target = links[0]
        path_part, _, anchor = target.partition("#")
        expected = owners.get(number)
        if expected is None:
            continue
        if _rel(expected.path) != path_part:
            report.fail(
                "RED-007",
                "CLAUDE.md",
                1,
                f"红线 {number} 指向 {path_part}，owner 是 {_rel(expected.path)}",
            )
        if anchor != f"redline-{number}":
            report.fail(
                "RED-008", "CLAUDE.md", 1, f"红线 {number} 锚点应为 redline-{number}"
            )


def _split_route_cell(cell: str) -> list[str]:
    value = cell.strip().strip("`")
    if not value or value in {"—", "-", "无"}:
        return []
    parts = re.split(r"[、,]", value)
    return [part.strip().strip("`") for part in parts if part.strip().strip("`")]


def check_routes(report: Report, docs_by_rel: dict[str, Doc], simulate_missing: str) -> None:
    if not CLAUDE.exists():
        return
    text = _read(CLAUDE)
    section = _claude_section(text, "任务路由")
    rows = [line for line in section.splitlines() if line.strip().startswith("|")]
    for line in rows:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        if cells[0] in {"任务", "---"} or set(cells[0]) <= {"-", ":"}:
            continue
        required = _split_route_cell(cells[1])
        optional = _split_route_cell(cells[2]) if len(cells) > 2 else []
        if len(required) > 3:
            report.fail("ROUTE-001", "CLAUDE.md", 1, f"{cells[0]} 必读 {len(required)} 篇 > 3")
        total = 0
        for rel in required + optional:
            target = DOCS / rel
            report.links += 1
            simulated = simulate_missing and rel == simulate_missing
            if simulated:
                report.simulated_refs += 1
            if not target.exists():
                if not simulated:
                    report.fail("ROUTE-002", "CLAUDE.md", 1, f"路由文档不存在：{rel}")
                continue
            doc = docs_by_rel.get(rel)
            if doc is not None:
                total += doc.line_count
        if total > 800:
            report.fail("ROUTE-003", "CLAUDE.md", 1, f"{cells[0]} 读取 {total} 行 > 800")
        report.routes.append(
            {
                "task": cells[0],
                "required": required,
                "optional": optional,
                "lines": total,
            }
        )


def _tier_counts(docs: list[Doc]) -> dict[str, int]:
    counts = {tier: 0 for tier in TIER_ORDER}
    for doc in docs:
        layer = str(doc.meta.get("档次") or doc.layer)
        counts[layer] = counts.get(layer, 0) + 1
    return counts


def render_docs_readme(docs: list[Doc]) -> str:
    lines = [
        "# 文档索引",
        "",
        "> 本文件由 `python tools/docs.py index` 生成，勿手改。入口见根目录 `CLAUDE.md`。",
        "",
        "## 分档",
        "",
        "| 档次 | 目录 | 用途 |",
        "|---|---|---|",
        "| L0 | `00-总纲/` | 定位、开发要诀、术语、文档规范 |",
        "| L1 | `10-契约/` | 跨模块契约、数据结构、传输格式、不变量 |",
        "| L2 | `20-模块/` | 单模块实现细则、扩展点、已知缺口 |",
        "| L3 | `30-操作/` | 上手、数据制备、命令、验收、排障 |",
        "| 记录 | `80-决策/`、`90-归档/` | ADR 与历史基线 |",
        "",
        "## 全部文档",
        "",
        "| 档次 | 文档 | 行数 | 说明 |",
        "|---|---|---:|---|",
    ]
    ordered = sorted(
        docs,
        key=lambda doc: (
            TIER_ORDER.index(str(doc.meta.get("档次") or doc.layer))
            if str(doc.meta.get("档次") or doc.layer) in TIER_ORDER
            else len(TIER_ORDER),
            doc.rel,
        ),
    )
    for doc in ordered:
        layer = str(doc.meta.get("档次") or doc.layer)
        lines.append(
            f"| {layer} | [{doc.rel}](./{doc.rel}) | {doc.line_count} | {doc.summary} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_claude_block(docs: list[Doc]) -> str:
    counts = _tier_counts(docs)
    parts = [
        f"{tier} {counts.get(tier, 0)} 篇"
        for tier in ("L0", "L1", "L2", "L3", "记录")
    ]
    return "\n".join(
        [
            INDEX_BEGIN,
            "- " + " · ".join(parts),
            "- 篇级索引：`docs/README.md`",
            INDEX_END,
        ]
    )


def replace_block(text: str, begin: str, end: str, replacement: str) -> str | None:
    start = text.find(begin)
    finish = text.find(end, start + len(begin)) if start >= 0 else -1
    if start < 0 or finish < 0:
        return None
    return text[:start] + replacement + text[finish + len(end) :]


def check_index(report: Report, docs: list[Doc]) -> None:
    expected_readme = render_docs_readme(docs)
    if not DOCS_README.exists():
        report.fail("IDX-001", "docs/README.md", 0, "docs/README.md 不存在")
    elif _read(DOCS_README) != expected_readme:
        report.fail("IDX-002", "docs/README.md", 1, "索引过期；运行 python tools/docs.py index")
    if not CLAUDE.exists():
        return
    text = _read(CLAUDE)
    expected_block = render_claude_block(docs)
    actual = None
    start = text.find(INDEX_BEGIN)
    finish = text.find(INDEX_END, start + len(INDEX_BEGIN)) if start >= 0 else -1
    if start >= 0 and finish >= 0:
        actual = text[start : finish + len(INDEX_END)]
    if actual != expected_block:
        report.fail("IDX-003", "CLAUDE.md", 1, "docs-index 区块过期；运行 python tools/docs.py index")


def _language_ratio(text: str) -> float:
    value = FENCE_RE.sub("", text)
    value = INLINE_CODE_RE.sub("", value)
    value = re.sub(r"https?://\S+", "", value)
    value = re.sub(r"\]\([^)]+\)", "]", value)
    value = re.sub(r"<[^>]+>", "", value)
    cjk = len(CJK_RE.findall(value))
    latin = len(LATIN_RE.findall(value))
    total = cjk + latin
    return (cjk / total) if total else 1.0


def check_language(report: Report, docs: Iterable[Doc]) -> None:
    for doc in docs:
        if is_exempt(doc):
            continue
        ratio = _language_ratio(doc.body)
        if ratio < 0.30:
            report.fail(
                "LANG-001",
                doc.rel,
                1,
                f"中文正文比例 {ratio:.2f} < 0.30（标识符/日语例词除外）",
            )


def check_commands(report: Report, docs: Iterable[Doc]) -> None:
    for doc in docs:
        if is_exempt(doc):
            continue
        for match in INLINE_CODE_RE.finditer(doc.body):
            value = match.group(1).strip()
            if not value.startswith("python "):
                continue
            tokens = value.split()
            if len(tokens) < 2:
                continue
            script = tokens[1].strip("'\"")
            if script in {"-m", "-c", "-V", "--version"}:
                if script == "-m" and len(tokens) < 3:
                    report.fail("CMD-001", doc.rel, 1, f"python -m 缺少模块：{value}")
                continue
            if script.startswith("-"):
                continue
            if not (script.endswith(".py") or "/" in script or "\\" in script):
                continue
            target = ROOT / script.replace("\\", "/")
            if not target.exists():
                report.fail("CMD-002", doc.rel, 1, f"命令脚本不存在：{script}")


def check_drift(report: Report, docs: list[Doc]) -> None:
    register = None
    for doc in docs:
        if doc.rel == "10-契约/15-不变量与缺陷码.md":
            register = doc
            break
    if register is None:
        report.fail("DRIFT-001", "docs", 0, "缺少 10-契约/15-不变量与缺陷码.md")
        return
    known = set(DRIFT_RE.findall(register.body))
    if not known:
        report.fail("DRIFT-002", register.rel, 1, "DRIFT 登记表为空")
    for doc in docs:
        if is_exempt(doc):
            continue
        refs = set(DRIFT_RE.findall(doc.body))
        unknown = sorted(refs - known)
        for ref in unknown:
            report.fail("DRIFT-003", doc.rel, 1, f"未登记的漂移：{ref}")
        if doc.layer == "L2" and not refs and "无已知缺口" not in doc.body:
            report.warn(
                "DRIFT-004",
                doc.rel,
                1,
                "L2 第 8 节未引用 DRIFT 编号，也未写“无已知缺口”",
            )


def check_defect_codes(report: Report, docs: list[Doc]) -> None:
    register = next(
        (doc for doc in docs if doc.rel == "10-契约/15-不变量与缺陷码.md"), None
    )
    if register is None:
        return
    codes = sorted(set(DEFECT_RE.findall(register.body)))
    sources: list[str] = []
    for path in sorted((ROOT / "core").glob("*.py")):
        sources.append(_read(path))
    for name in ("main.py", "init.py", "quiz.py", "config.py"):
        path = ROOT / name
        if path.exists():
            sources.append(_read(path))
    haystack = "\n".join(sources)
    for code in codes:
        if code not in haystack:
            report.fail("DEFECT-001", register.rel, 1, f"缺陷码在代码中不存在：{code}")


def build_report(simulate_missing: str = "") -> Report:
    report = Report()
    docs, errors = discover_docs()
    report.docs = docs
    for error in errors:
        report.fail("FM-000", "docs", 0, error)
    docs_by_rel = {doc.rel: doc for doc in docs}
    check_front_matter(report, docs)
    check_l2_sections(report, docs)
    covered = check_code_paths(report, docs)
    check_coverage(report, covered)
    check_links(report, docs, simulate_missing)
    check_claude(report)
    check_redlines(report, docs)
    check_routes(report, docs_by_rel, simulate_missing)
    check_index(report, docs)
    check_language(report, docs)
    check_commands(report, docs)
    check_drift(report, docs)
    check_defect_codes(report, docs)
    if simulate_missing and report.simulated_refs == 0:
        report.fail(
            "GRAPH-001",
            simulate_missing,
            0,
            "没有可检测的入站引用；删除该文档不会触发死链",
        )
    return report


def cmd_index(args: argparse.Namespace) -> int:
    docs, errors = discover_docs()
    if errors:
        for error in errors:
            print(f"docs index error: {error}", file=sys.stderr)
        return 2
    if not CLAUDE.exists():
        print("docs index error: CLAUDE.md 不存在", file=sys.stderr)
        return 2
    expected_readme = render_docs_readme(docs)
    expected_block = render_claude_block(docs)
    claude_text = _read(CLAUDE)
    updated = replace_block(claude_text, INDEX_BEGIN, INDEX_END, expected_block)
    if updated is None:
        print("docs index error: CLAUDE.md 缺少 docs-index 标记", file=sys.stderr)
        return 2
    if args.check:
        stale: list[str] = []
        if not DOCS_README.exists() or _read(DOCS_README) != expected_readme:
            stale.append("docs/README.md")
        if updated != claude_text:
            stale.append("CLAUDE.md")
        if stale:
            print("docs index stale: " + ", ".join(stale), file=sys.stderr)
            return 1
        print(f"docs index OK: 2 targets fresh ({len(docs)} docs)")
        return 0
    DOCS_README.write_text(expected_readme, encoding="utf-8", newline="\n")
    CLAUDE.write_text(updated, encoding="utf-8", newline="\n")
    print(f"docs index written: CLAUDE.md, docs/README.md ({len(docs)} docs)")
    return 0


def _print_finding(item: Finding) -> None:
    location = item.path + (f":{item.line}" if item.line else "")
    print(f"{item.severity.upper():4} [{item.rule}] {location}: {item.message}")


def cmd_lint(args: argparse.Namespace) -> int:
    report = build_report(simulate_missing=args.simulate_missing or "")
    if args.json:
        payload = {
            "ok": not report.failures,
            "docs": len(report.docs),
            "links": report.links,
            "simulated_refs": report.simulated_refs,
            "routes": report.routes,
            "fail": [item.__dict__ for item in report.failures],
            "warn": [item.__dict__ for item in report.warnings],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for item in report.failures:
            _print_finding(item)
        for item in report.warnings:
            _print_finding(item)
        state = "OK" if not report.failures else "FAIL"
        print(
            f"docs lint {state}: {len(report.docs)} docs, {report.links} links, "
            f"{len(report.failures)} fail, {len(report.warnings)} warn"
        )
        if args.simulate_missing:
            print(
                f"simulate-missing: {args.simulate_missing} would break "
                f"{report.simulated_refs} refs"
            )
    return 0 if not report.failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="JLPTCraft docs index/lint")
    sub = parser.add_subparsers(dest="command", required=True)

    index_parser = sub.add_parser("index", help="regenerate docs index")
    index_parser.add_argument("--check", action="store_true", help="only check freshness")
    index_parser.set_defaults(func=cmd_index)

    lint_parser = sub.add_parser("lint", help="run docs lint")
    lint_parser.add_argument("--json", action="store_true", help="JSON report")
    lint_parser.add_argument(
        "--simulate-missing",
        default="",
        help="docs-relative path to treat as missing",
    )
    lint_parser.set_defaults(func=cmd_lint)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
