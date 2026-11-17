#!/usr/bin/env python
"""JLPTCraft CLI: lightweight generation, MDict extension, storage, export."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import (
    ConfigurationError,
    SemesterViolation,
    Settings,
    ensure_directories,
    load_settings,
    mask_secret,
)
from core.context import AppContext, ContextError, load_context
from core.db import (
    DatabaseError,
    get_batch,
    init_db,
    list_batches,
    stats as db_stats,
)
from core.discovery import discover_unidic
from core.exporter import (
    archive_batch,
    export_existing_batch,
    load_batch_json,
)
from core.items import item_from_json
from core.render import (
    AnswerMode,
    render_answer_book,
    render_inline_book,
    render_item,
    render_question_book,
    resolve_answer_mode,
)
from core.extension import (
    ExtensionPool,
    live_extension_candidates,
    merge_entries,
)
from core.intent import (
    UNSUPPORTED_ALIASES,
    IntentError,
    QuizPlan,
    UnsupportedTypeError,
    parse_plan,
)
from core.item_bank import ItemBank
from core.llm import LLMClient, LLMError, LLMUnavailable, create_llm
from core.locks import LockTable
from core.mdict_index import MDictIndex
from core.miner import run_mining_and_review
from core.models import (
    MistakeRulesFile,
    QuestionType,
    QuestionTypesFile,
    RuleStatus,
    now_iso,
)
from core.runner import QuizRunner
from core.tagformat import first_tag

try:
    from rich.console import Console
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]


HELP = """
JLPTCraft 主菜单
自然语言示例：
  生成一套模拟卷
  生成3道第44课读音题
  围绕旅行生成10道拓展题
  只出课内题
命令：
  /help                       显示帮助
  /list [n]                   查看最近批次
  /show <batch_id>            查看批次题目
  /export <batch_id> [--no-answers]  导出 Markdown
  /rules [rule_id]            查看规则库/单条规则
  /approve <rule_id>          批准一条草稿规则
  /reject <rule_id>           弃用一条规则
  /remine                     重新挖掘规则并审核
  /stats                      SQLite 统计
  /env                        环境/资源/数据状态
  /dict <word>                查询全部 MDict 资源
  /extend [topic|build topic] 查看/构建拓展词
  /mode <in_syllabus|mdict_distractors|extend>  设置默认范围
  /seed <int>                 设置随机种子
  /workers <int>              设置 LLM 并发数
  /quit                       退出
"""


def _console() -> Any:
    return Console() if Console else None


def _print(message: str, style: str | None = None) -> None:
    console = _console()
    if console:
        console.print(message, style=style, markup=False, highlight=False)
    else:
        print(message)


def _error(message: str) -> None:
    _print(message, "red")


def _make_llm(settings: Settings, offline: bool) -> Any:
    return create_llm(settings, offline=offline)


def _make_mdict(settings: Settings) -> MDictIndex:
    try:
        return MDictIndex(settings)
    except Exception:
        return MDictIndex(settings, timeout=2)


def _print_header(
    settings: Settings,
    context: AppContext,
    retriever_unidic: Path | None,
    mdict: MDictIndex,
    extension_pool: ExtensionPool,
) -> None:
    lessons = context.vocab.approved_lessons()
    points = len(context.grammar.approved_points())
    forms = sum(1 for point in context.grammar.approved_points() if point.approved_answer_forms)
    _print(
        f"JLPTCraft · {settings.active_semester.value} · "
        f"LLM={'已配置' if settings.has_llm else '未配置'} · "
        f"UniDic={'自动发现' if retriever_unidic else '未发现'} · "
        f"MDict={mdict.count() if mdict.available else 0} 个",
        "cyan",
    )
    _print(
        f"数据：{len(lessons)} 课 / {len(context.vocab.all_entries())} 词；"
        f"语法 {points} 点 / {forms} 点有 forms；拓展词 {len(extension_pool.entries)} 条"
    )
    _print(f"目录：{settings.project_root}")
    _print("输入自然语言需求或 /help；Ctrl+C 退出。", "dim")


def _print_rendered_batch(batch: Any) -> None:
    rendered = []
    seed = int(batch.batch_meta.seed or 0)
    for index, data in enumerate(batch.items, start=1):
        try:
            rendered.append(render_item(item_from_json(data), seed=seed, number=index))
        except Exception:
            continue
    if not rendered:
        return
    mode = resolve_answer_mode(
        batch.batch_meta.answers_mode,
        is_mock=batch.batch_meta.mode == "mock",
    )
    if mode == AnswerMode.ONLY:
        _print(render_answer_book(rendered))
    elif mode == AnswerMode.INLINE:
        _print(render_inline_book(rendered))
    else:
        _print(render_question_book(rendered))


def _print_plan(plan: QuizPlan) -> None:
    mode = "模拟卷" if plan.mode == "mock" else "练习"
    lessons = ", ".join(str(item) for item in plan.lessons)
    mix = " / ".join(
        f"{'ai' if key == 'new' else key}:{value:.0%}"
        for key, value in plan.source_mix.items()
    )
    _print(
        f"计划：{mode} {plan.total} 题 | 用途 {plan.use_case} | 范围 {lessons} | "
        f"{plan.type_summary()} | {plan.scope_label()}"
        + (f" | 主题 {plan.topic}" if plan.topic else "")
        + f" | 来源配比 {mix}",
        "cyan",
    )
    for warning in plan.warnings:
        _print(f"提示：{warning}", "yellow")


def _ask(prompt: str, default: str = "") -> str:
    try:
        value = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return value or default


def _lesson_text(lessons: list[int]) -> str:
    if not lessons:
        return ""
    ordered = sorted(set(lessons))
    if ordered == list(range(ordered[0], ordered[-1] + 1)):
        return f"第{ordered[0]}-{ordered[-1]}课"
    return " ".join(f"第{item}课" for item in ordered)


def _strip_unsupported(text: str) -> str:
    for alias in UNSUPPORTED_ALIASES:
        text = text.replace(alias, "")
    return text.strip()


def _parse_with_clarification(
    text: str,
    context: AppContext,
    *,
    llm: Any,
    offline: bool,
    scope_override: str | None,
    topic_override: str | None,
    force_mock: bool,
    seed: int | None,
    use_case: str | None,
    interactive: bool,
    count_override: int | None = None,
) -> QuizPlan:
    kwargs = dict(
        llm=llm,
        offline=offline,
        scope_override=scope_override,
        topic_override=topic_override,
        force_mock=force_mock,
        seed=seed,
        use_case_override=use_case,
        count_override=count_override,
    )
    skip_vague_prompt = False
    try:
        plan = parse_plan(text, context, **kwargs)
    except UnsupportedTypeError as exc:
        if not interactive:
            raise
        _print(f"{exc}", "yellow")
        answer = _ask("改用读音/汉字/词汇/表达/语法五类配比？[Y/n] ", "y").lower()
        if answer in {"n", "no"}:
            text = _ask("请输入完整需求（例：第44-50课 80题 按真题配比）：", text)
        else:
            text = _strip_unsupported(text) or "生成一套模拟卷"
            skip_vague_prompt = True
        plan = parse_plan(text, context, **kwargs)
    except IntentError as exc:
        if not interactive:
            raise
        _print(f"还缺少可执行信息：{exc}", "yellow")
        text = _ask(
            "请补充：范围（哪些课）、题量、题型，例如“第44-50课 80题 模拟卷”：",
            "生成一套模拟卷",
        )
        plan = parse_plan(text, context, **kwargs)

    if (
        interactive
        and plan.mode == "mock"
        and not plan.explicit_count
        and not plan.explicit_types
        and not force_mock
    ):
        answer = _ask("检测到考卷请求：按 80 题五类（2024 配比）生成？[Y/n] ", "y").lower()
        if answer in {"n", "no"}:
            count = _ask("需要多少题？[默认 80] ", "80")
            types = _ask(
                "题型与数量（例：读音30 汉字10 词汇5 表达10 语法25；留空用真题配比）："
            )
            new_text = f"生成{count}题考卷" + (f" {types}" if types else "")
            if plan.explicit_lessons:
                new_text += f" {_lesson_text(plan.lessons)}"
            plan = parse_plan(new_text, context, **kwargs)

    if interactive and not skip_vague_prompt and plan.source == "default" and plan.mode != "mock":
        answer = _ask("未指定题型，按真题配比生成？[Y/n] ", "y").lower()
        if answer in {"n", "no"}:
            types = _ask("请输入题型与数量（例：读音3 语法2）：")
            if types:
                plan = parse_plan(
                    f"生成 {types} {_lesson_text(plan.lessons)}",
                    context,
                    **kwargs,
                )
    return plan


def _generate(
    settings: Settings,
    text: str,
    *,
    offline: bool = False,
    dry_run: bool = False,
    json_output: bool = False,
    no_stream: bool = False,
    hide_answers: bool = False,
    answers: str = "auto",
    reveal: bool = False,
    force_mock: bool = False,
    scope_override: str | None = None,
    topic_override: str | None = None,
    workers: int | None = None,
    seed: int | None = None,
    use_case: str | None = None,
    count_override: int | None = None,
    interactive: bool = True,
) -> Any:
    try:
        context = load_context(settings)
    except ContextError as exc:
        _error(str(exc))
        raise
    llm = _make_llm(settings, offline)
    mdict = _make_mdict(settings)
    extension_pool = ExtensionPool.load(settings.extension_path)
    item_bank = ItemBank.load(settings.item_bank_path)
    locks = LockTable.load(settings.locks_path)
    plan = _parse_with_clarification(
        text,
        context,
        llm=None if offline else llm,
        offline=offline,
        scope_override=scope_override,
        topic_override=topic_override,
        force_mock=force_mock,
        seed=seed,
        use_case=use_case,
        count_override=count_override,
        interactive=interactive,
    )

    has_llm = llm.available
    if not has_llm:
        if plan.mode == "mock":
            raise LLMUnavailable("模拟卷需要 LLM；请先在 .env 配置 LLM_API_KEY，或使用 --offline 做课内读音/汉字练习。")
        local_types = {
            key: value
            for key, value in plan.type_plan.items()
            if context.profiles is not None
            and key in context.profiles.types
            and context.profiles.type(key).is_local
        }
        if not local_types:
            raise LLMUnavailable("该请求需要 LLM；当前未配置 LLM_API_KEY。")
        plan.type_plan = local_types
        plan.count = sum(local_types.values())
        plan.scope = "in_syllabus"
        plan.warnings.append("未配置 LLM：已自动只保留本地题型（含插件本地题型）。")

    if plan.scope != "in_syllabus" and not mdict.available:
        plan.scope = "in_syllabus"
        plan.topic = ""
        plan.warnings.append("未发现 MDict：已回退到课内模式。")
    elif not mdict.available:
        plan.warnings.append("未发现 MDict：课内出题正常；拓展模式将降级。")
    if plan.scope == "extend" and mdict.available and plan.topic:
        if len(extension_pool.candidates(plan.topic)) < plan.count:
            live = live_extension_candidates(
                plan.topic, mdict, textbook=context.vocab, limit=40
            )
            extension_pool, added, _ = merge_entries(extension_pool, live)
            if added:
                plan.warnings.append(f"已从本地 MDict 实时补充 {added} 个拓展词。")
    if plan.scope == "extend" and not extension_pool.entries:
        if not (mdict.available and plan.topic):
            plan.scope = "in_syllabus"
            plan.warnings.append("拓展词池为空：已回退到课内模式。")

    _print_plan(plan)
    runner = QuizRunner(
        settings,
        context,
        llm=llm if has_llm else None,
        mdict=mdict if mdict.available else None,
        extension_pool=extension_pool,
        item_bank=item_bank,
        locks=locks,
        workers=workers,
    )
    result = runner.run(
        plan,
        stream=not json_output and not no_stream,
        dry_run=dry_run,
        hide_answers=hide_answers,
        answers=answers,
        reveal=reveal,
    )
    if json_output:
        _print(json.dumps(result.batch.model_dump(mode="json"), ensure_ascii=False, indent=2))
    elif no_stream and result.batch is not None:
        _print_rendered_batch(result.batch)

    if result.batch is not None:
        _print(
            f"完成：{len(result.batch.questions)}/{plan.total} 题；"
            f"失败 {len(result.failures)}；耗时 {result.duration_ms}ms",
            "green" if not result.failures else "yellow",
        )
        metrics = result.batch.batch_meta.metrics
        if metrics:
            _print(
                "本地指标："
                f"来源 {metrics.get('source_counts')} | "
                f"域 {metrics.get('domain_counts')} | "
                f"考点覆盖 {metrics.get('point_coverage')} | "
                f"题面重合 max/avg {metrics.get('stem_overlap_max')}/{metrics.get('stem_overlap_avg')} | "
                f"超纲选项率 {metrics.get('out_of_level_rate')} | "
                f"裸框架率 {metrics.get('bare_framework_rate')} | "
                f"同场无线索率 {metrics.get('same_field_no_clue_rate')} | "
                f"答案分布 {metrics.get('answer_distribution')}",
                "dim",
            )
        for failure in result.failures[:8]:
            _print(f"  - [{failure.seq}] {failure.qtype}: {failure.message}", "yellow")
        if result.warnings:
            for warning in result.warnings:
                _print(f"  ! {warning}", "yellow")
        if result.json_path:
            _print(f"已归档：{result.json_path}")
            _print(f"已导出：{result.md_path}")
            if result.md_path:
                answer_path = Path(result.md_path).with_name(
                    Path(result.md_path).stem + "-answers.md"
                )
                if answer_path.exists():
                    _print(f"答案册：{answer_path}")
            if result.duplicates:
                _print("重复题未入库：" + ", ".join(result.duplicates), "yellow")
        elif dry_run:
            _print("dry-run：未写入数据库/Markdown。", "yellow")
    return result


def _show_batch(settings: Settings, batch_id: str) -> None:
    batch_row = get_batch(settings, batch_id)
    if batch_row is None:
        _error(f"找不到批次 {batch_id}")
        return
    json_path = Path(batch_row.get("json_path") or "")
    if json_path.exists():
        batch = load_batch_json(json_path)
        _print(render_batch_console(batch))
        return
    _print(f"批次 {batch_id} 共 {len(batch_row.get('questions', []))} 题；JSON 归档缺失。")


def _export_batch(settings: Settings, batch_id: str, include_answers: bool = True) -> None:
    rows = list_batches(settings, limit=1000)
    row = next((item for item in rows if item["batch_id"] == batch_id), None)
    if row is None:
        _error(f"找不到批次 {batch_id}")
        return
    json_path = Path(row.get("json_path") or "")
    if not json_path.exists():
        _error(f"批次 JSON 不存在：{json_path}")
        return
    batch = load_batch_json(json_path)
    path = export_existing_batch(settings, batch, include_answers=include_answers)
    _print(f"已导出：{path}", "green")


def _list_batches(settings: Settings, limit: int = 20) -> None:
    rows = list_batches(settings, limit=limit)
    if not rows:
        _print("题库为空。", "yellow")
        return
    if Console is None or Table is None:
        for row in rows:
            _print(f"{row['batch_id']} | {row['created_at']} | {row['question_count']} 题 | {row['model']}")
        return
    table = Table(title=f"题库批次（{settings.active_semester.value}）")
    table.add_column("batch_id", no_wrap=True)
    table.add_column("时间")
    table.add_column("题数", justify="right")
    table.add_column("模型")
    table.add_column("请求")
    for row in rows:
        request = row.get("request", {})
        table.add_row(
            row["batch_id"],
            row["created_at"],
            str(row["question_count"]),
            row["model"] or "",
            str(request.get("raw") or request.get("request") or "")[:40],
        )
    Console().print(table)


def _print_rules(settings: Settings, rule_id: str | None = None) -> None:
    if not settings.common_mistakes_path.exists():
        _error("规则库缺失。")
        return
    rules = MistakeRulesFile.model_validate(
        json.loads(settings.common_mistakes_path.read_text(encoding="utf-8"))
    )
    if rule_id:
        rule = rules.rule_map().get(rule_id)
        if rule is None:
            _error(f"找不到规则 {rule_id}")
            return
        _print(f"{rule.id} | {rule.name_zh} | {rule.status.value} | checker={rule.checker}")
        _print(rule.description)
        _print(f"适用：{[t.value for t in rule.applies_to]}")
        _print(f"证据数：{rule.evidence_count}；置信度：{rule.confidence}")
        return
    if Console is None or Table is None:
        for rule in rules.rules:
            _print(f"{rule.id} | {rule.name_zh} | {rule.status.value} | {rule.evidence_count}")
        return
    table = Table(title="设误规则库（仅供展示/本地干扰项参考）")
    table.add_column("规则 ID", no_wrap=True)
    table.add_column("名称")
    table.add_column("状态", no_wrap=True)
    table.add_column("次数", justify="right")
    table.add_column("checker", no_wrap=True)
    for rule in sorted(rules.rules, key=lambda r: (-r.evidence_count, r.id)):
        table.add_row(
            rule.id,
            rule.name_zh,
            rule.status.value,
            str(rule.evidence_count),
            rule.checker or "语义",
        )
    Console().print(table)


def _set_rule_status(settings: Settings, rule_id: str, status: RuleStatus) -> bool:
    if not settings.common_mistakes_path.exists() or not settings.question_types_path.exists():
        _error("规则库缺失。")
        return False
    rules = MistakeRulesFile.model_validate(
        json.loads(settings.common_mistakes_path.read_text(encoding="utf-8"))
    )
    rule = rules.rule_map().get(rule_id)
    if rule is None:
        _error(f"找不到规则 {rule_id}")
        return False
    rule.status = status
    rule.locked = status == RuleStatus.LOCKED
    rules.meta["reviewed"] = True
    rules.meta["reviewed_at"] = now_iso()
    settings.common_mistakes_path.write_text(
        json.dumps(rules.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    types = QuestionTypesFile.model_validate(
        json.loads(settings.question_types_path.read_text(encoding="utf-8"))
    )
    for type_rule in types.types.values():
        if any(rule_id == current for current in type_rule.allowed_mistake_rules):
            if status not in {RuleStatus.APPROVED, RuleStatus.LOCKED}:
                type_rule.allowed_mistake_rules = [
                    current
                    for current in type_rule.allowed_mistake_rules
                    if current != rule_id
                ]
        elif status in {RuleStatus.APPROVED, RuleStatus.LOCKED}:
            if any(qtype in rule.applies_to for qtype in [type_rule.type]):
                type_rule.allowed_mistake_rules = sorted(
                    set(type_rule.allowed_mistake_rules) | {rule_id}
                )
    types.meta["reviewed"] = True
    settings.question_types_path.write_text(
        json.dumps(types.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _print(f"规则 {rule_id} 已设为 {status.value}。", "green")
    return True


def _print_env(settings: Settings, mdict: MDictIndex, extension_pool: ExtensionPool) -> None:
    unidic = discover_unidic(settings)
    lines = [
        f"学期：{settings.active_semester.value}",
        f"项目根：{settings.project_root}",
        f"数据目录：{settings.data_dir}",
        f"规则目录：{settings.rules_dir}",
        f"DB：{settings.db_path}",
        f"LLM：{settings.llm_model} @ {settings.llm_base_url} / key={mask_secret(settings.llm_api_key)}",
        f"并发：{settings.llm_concurrency} / max_tokens={settings.llm_max_tokens} / timeout={settings.llm_timeout}s",
        f"UniDic：{unidic or '未发现（使用教材读音）'}",
        f"MDict：{mdict.count()} 个 .mdx / 拓展池 {len(extension_pool.entries)} 条",
        f"拓展主题：{', '.join(extension_pool.topics()[:12]) or '无'}",
    ]
    _print("\n".join(lines))


def _dict_command(mdict: MDictIndex, word: str) -> None:
    if not mdict.available:
        _error("未发现 MDict 词典；请把 .mdx 放到本地 MDict 目录。")
        return
    hits = mdict.query_all(word)
    if not hits:
        _print(f"没有查到：{word}", "yellow")
        return
    for hit in hits:
        _print(f"[{hit.file.label}] {hit.text[:400]}")


def _extend_command(
    settings: Settings,
    context: AppContext,
    mdict: MDictIndex,
    pool: ExtensionPool,
    argument: str,
) -> ExtensionPool:
    argument = argument.strip()
    if not argument:
        topics = pool.topics()
        _print(f"拓展池：{len(pool.entries)} 条；主题：{', '.join(topics[:20]) or '无'}")
        return pool
    if argument.startswith("build "):
        topic = argument[6:].strip()
        if not mdict.available:
            _error("未发现 MDict，无法构建拓展词。")
            return pool
        new_entries = live_extension_candidates(
            topic, mdict, textbook=context.vocab, limit=40
        )
        merged, added, updated = merge_entries(pool, new_entries)
        merged.meta["updated_at"] = now_iso()
        merged.meta["version"] = int(merged.meta.get("version", 1)) + 1
        merged.save(settings.extension_path)
        _print(
            f"拓展池已更新：新增 {added}，合并 {updated}，总计 {len(merged.entries)} 条。",
            "green",
        )
        return merged
    topic = argument
    if not mdict.available:
        _error("未发现 MDict，拓展搜索不可用。")
        return pool
    entries = pool.candidates(topic)
    if not entries:
        entries = live_extension_candidates(topic, mdict, textbook=context.vocab, limit=20)
    if not entries:
        _print(f"没有找到与「{topic}」相关的拓展词。", "yellow")
        return pool
    for entry in entries[:20]:
        gloss = mdict.chinese_gloss(entry.written) or ""
        reading = entry.kana or mdict.reading(entry.written) or ""
        _print(f"- {entry.written}（{reading}） {gloss}  [{entry.confidence}]")
    return pool


def run_repl(settings: Settings, offline: bool = False) -> int:
    ensure_directories(settings)
    try:
        context = load_context(settings)
    except ContextError as exc:
        _error(str(exc))
        return 1
    mdict = _make_mdict(settings)
    extension_pool = ExtensionPool.load(settings.extension_path)
    unidic = discover_unidic(settings)
    _print_header(settings, context, unidic, mdict, extension_pool)

    default_scope: str | None = None
    default_topic = ""
    default_workers: int | None = None
    seed: int | None = None

    while True:
        try:
            line = input("jp> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            command = parts[0].lower()
            argument = parts[1].strip() if len(parts) > 1 else ""
            if command in {"/quit", "/exit", "/q"}:
                return 0
            if command == "/help":
                _print(HELP)
                continue
            if command == "/list":
                try:
                    _list_batches(settings, int(argument) if argument else 20)
                except ValueError:
                    _list_batches(settings)
                continue
            if command == "/show":
                if not argument:
                    _error("用法：/show <batch_id>")
                else:
                    _show_batch(settings, argument)
                continue
            if command == "/export":
                tokens = argument.split()
                if not tokens:
                    _error("用法：/export <batch_id> [--no-answers]")
                    continue
                _export_batch(
                    settings,
                    tokens[0],
                    include_answers="--no-answers" not in tokens,
                )
                continue
            if command == "/rules":
                if argument == "diff":
                    _print("规则差异见 data/rules/_draft/ 与 data/rules/history/。")
                else:
                    _print_rules(settings, argument or None)
                continue
            if command in {"/approve", "/reject"}:
                if not argument:
                    _error(f"用法：{command} <rule_id>")
                    continue
                status = RuleStatus.APPROVED if command == "/approve" else RuleStatus.DEPRECATED
                _set_rule_status(settings, argument, status)
                continue
            if command == "/remine":
                if run_mining_and_review(settings, force=True, assume_yes=False, offline=offline):
                    try:
                        context = load_context(settings)
                    except ContextError as exc:
                        _error(str(exc))
                continue
            if command == "/stats":
                init_db(settings)
                _print(json.dumps(db_stats(settings), ensure_ascii=False, indent=2))
                continue
            if command == "/env":
                _print_env(settings, mdict, extension_pool)
                continue
            if command == "/dict":
                if not argument:
                    _error("用法：/dict <word>")
                else:
                    _dict_command(mdict, argument)
                continue
            if command == "/extend":
                extension_pool = _extend_command(
                    settings, context, mdict, extension_pool, argument
                )
                continue
            if command == "/mode":
                if argument in {"in_syllabus", "mdict_distractors", "extend"}:
                    default_scope = argument
                    _print(f"默认范围：{argument}", "green")
                else:
                    _print(f"默认范围：{default_scope or '未设置（自动）'}")
                continue
            if command == "/seed":
                if argument:
                    try:
                        seed = int(argument)
                        _print(f"随机种子：{seed}")
                    except ValueError:
                        _error("用法：/seed <int>")
                else:
                    _print(f"随机种子：{seed if seed is not None else '随机'}")
                continue
            if command == "/workers":
                if argument:
                    try:
                        default_workers = max(1, int(argument))
                        _print(f"LLM 并发：{default_workers}")
                    except ValueError:
                        _error("用法：/workers <int>")
                else:
                    _print(f"LLM 并发：{default_workers or settings.llm_concurrency}")
                continue
            _error(f"未知命令：{command}；输入 /help 查看。")
            continue

        try:
            _generate(
                settings,
                line,
                offline=offline,
                scope_override=default_scope,
                topic_override=default_topic or None,
                workers=default_workers,
                seed=seed,
            )
            # Extension pool may have changed via generation/live search.
            extension_pool = ExtensionPool.load(settings.extension_path)
        except (IntentError, LLMError, DatabaseError, ContextError) as exc:
            _error(f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            _error(f"未预期错误：{type(exc).__name__}: {exc}")


def self_test(settings: Settings) -> int:
    import tempfile

    tmp = tempfile.TemporaryDirectory(prefix="jlpcraft-selftest-")
    try:
        tmp_settings = replace(
            settings,
            data_dir=Path(tmp.name) / "data",
            db_dir=Path(tmp.name) / "db",
            cache_dir=Path(tmp.name) / "cache",
            log_dir=Path(tmp.name) / "logs",
        )
        ensure_directories(tmp_settings)
        context = load_context(settings)
        plan = QuizPlan(
            raw="self-test",
            mode="practice",
            scope="in_syllabus",
            count=1,
            lessons=[37],
            type_plan={QuestionType.KANJI_TO_KANA.value: 1},
            seed=42,
            source="local",
        )
        runner = QuizRunner(tmp_settings, context, llm=None, workers=1)
        result = runner.run(plan, stream=False)
        if result.batch is None or not result.batch.questions:
            _error("self-test 未生成题目")
            return 1
        _print(
            f"self-test 通过：归档 {Path(result.json_path).name} / "
            f"{Path(result.md_path).name}，入库 {len(result.batch.questions)} 题。",
            "green",
        )
        if result.md_path:
            _print(Path(result.md_path).read_text(encoding="utf-8")[:800])
        return 0
    finally:
        tmp.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JLPTCraft CLI")
    parser.add_argument("request", nargs="?", help="自然语言需求；省略则进入 REPL")
    parser.add_argument("--mock", action="store_true", help="生成一套 80 题模拟卷")
    parser.add_argument(
        "--scope",
        choices=("in_syllabus", "mdict_distractors", "extend"),
        help="范围：课内/课内+拓展干扰/拓展学习",
    )
    parser.add_argument("--topic", help="拓展主题，例如 旅行")
    parser.add_argument(
        "--use-case",
        "--purpose",
        dest="use_case",
        choices=("first_learn", "unit_test", "mock", "review"),
        help="用途预设：首学/单元测/模拟卷/错题复习",
    )
    parser.add_argument(
        "--n",
        "--count",
        dest="count",
        type=int,
        help="题量覆盖（flag 层，零 LLM 意图解析）",
    )
    parser.add_argument("--workers", type=int, help="LLM 并发数")
    parser.add_argument("--seed", type=int, help="随机种子")
    parser.add_argument("--hide-answers", action="store_true", help="流式显示时不打印答案/解析")
    parser.add_argument(
        "--answers",
        choices=("auto", "separate", "inline", "only", "hidden"),
        default="auto",
        help="答案呈现：separate 双册/inline 合并/only 仅答案册/hidden 流式隐藏",
    )
    parser.add_argument("--reveal", action="store_true", help="hidden 模式下逐题揭示答案")
    parser.add_argument("--no-stream", action="store_true", help="全部完成后再输出")
    parser.add_argument("--dry-run", action="store_true", help="只生成，不写 DB/Markdown")
    parser.add_argument("--json", action="store_true", help="打印生成批次 JSON")
    parser.add_argument("--offline", action="store_true", help="禁用 LLM；仅本地读音/汉字题")
    parser.add_argument("--health", action="store_true", help="环境检查")
    parser.add_argument("--list", action="store_true", help="列出题库批次")
    parser.add_argument("--export", metavar="BATCH_ID", help="导出批次 Markdown")
    parser.add_argument("--no-answers", action="store_true", help="导出学生版（不含答案）")
    parser.add_argument("--self-test", action="store_true", help="离线自检")
    parser.add_argument(
        "--audit",
        nargs="?",
        const="",
        metavar="BATCH_ID",
        help="审计 Item/视图/请求日志；省略批次则审计最近一批",
    )
    return parser.parse_args(argv)


def audit_command(batch_id: str = "", json_output: bool = False) -> int:
    from core.audit import audit_items, audit_text

    settings = load_settings()
    ensure_directories(settings)
    report = audit_items(settings, batch_id)
    if json_output:
        _print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        _print(audit_text(report))
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        settings = load_settings()
    except (ConfigurationError, SemesterViolation) as exc:
        _error(f"配置错误：{exc}")
        return 2
    ensure_directories(settings)
    if args.health:
        from init import _render_check_report, check_environment

        _render_check_report(settings, check_environment(settings))
        return 0
    if args.self_test:
        return self_test(settings)
    if args.audit is not None:
        return audit_command(args.audit, args.json)
    if args.request and args.request.strip() == "audit":
        return audit_command("", args.json)
    if args.list:
        init_db(settings)
        _list_batches(settings)
        return 0
    if args.export:
        init_db(settings)
        _export_batch(settings, args.export, include_answers=not args.no_answers)
        return 0
    if not args.request and (args.mock or args.use_case or args.count is not None):
        args.request = (
            "生成一套模拟卷"
            if (args.mock or args.use_case == "mock")
            else "生成题"
        )
    if not args.request and not args.mock:
        return run_repl(settings, offline=args.offline)
    request = args.request or "生成一套模拟卷"
    try:
        _generate(
            settings,
            request,
            offline=args.offline,
            dry_run=args.dry_run,
            json_output=args.json,
            no_stream=args.no_stream,
            hide_answers=args.hide_answers,
            answers=args.answers,
            reveal=args.reveal,
            force_mock=args.mock or args.use_case == "mock",
            scope_override=args.scope,
            topic_override=args.topic,
            workers=args.workers,
            seed=args.seed,
            use_case=args.use_case,
            count_override=args.count,
            interactive=sys.stdin.isatty(),
        )
    except (IntentError, LLMError, DatabaseError, ContextError, UnsupportedTypeError) as exc:
        _error(f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
