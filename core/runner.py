"""Parallel one-question runner with immediate streaming output."""

from __future__ import annotations

import hashlib
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from config import Settings
from core.context import AppContext
from core.contracts import PluginContext, PluginError
from core.db import (
    exposed_points,
    insert_batch,
    record_exposures,
    used_target_counts,
)
from core.exporter import archive_batch
from core.extension import ExtensionPool
from core.fingerprint import question_content_hash
from core.intent import QuizPlan
from core.item_bank import ItemBank
from core.items import (
    Item,
    ItemBuildError,
    build_item_from_generated_question,
    item_to_json,
    project_generated_question,
    refresh_hashes,
    validate_item,
)
from core.locks import LockTable
from core.models import (
    BatchMeta,
    GeneratedBatch,
    GeneratedQuestion,
    QuestionType,
    TYPE_LABELS_ZH,
    normalize_text,
    now_iso,
)
from core.pools import TextbookPools
from core.registry import question_types
from core.quizgen import (
    QuestionBuildError,
    QuestionSpec,
    build_specs,
    generate_llm,
    generate_local,
    generate_local_grammar,
    generate_original,
    generate_variant,
    style_key,
)
from core.render import (
    AnswerMode,
    inline_view,
    key_view,
    render_item,
    resolve_answer_mode,
    stem_view,
)
from core.similarity import max_overlap


class FatalLLMError(RuntimeError):
    pass


class _NeedsLLM(RuntimeError):
    pass


@dataclass
class RunFailure:
    seq: int
    qtype: str
    message: str


@dataclass
class RunResult:
    batch: GeneratedBatch | None
    failures: list[RunFailure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    json_path: str = ""
    md_path: str = ""
    duplicates: list[str] = field(default_factory=list)
    duration_ms: int = 0
    items: list[Item] = field(default_factory=list)


def new_batch_id() -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"{random.SystemRandom().randint(0, 0xFFFF):04x}"
    return f"b2a-{stamp}-{suffix}"


class QuizRunner:
    def __init__(
        self,
        settings: Settings,
        context: AppContext,
        *,
        llm: Any | None = None,
        mdict: Any | None = None,
        extension_pool: ExtensionPool | None = None,
        item_bank: ItemBank | None = None,
        locks: LockTable | None = None,
        workers: int | None = None,
    ):
        self.settings = settings
        self.context = context
        self.llm = llm
        self.mdict = mdict
        self.extension_pool = extension_pool
        self.item_bank = item_bank or ItemBank([])
        self.locks = locks or LockTable({})
        self.workers = max(1, workers or settings.llm_concurrency)
        self._print_lock = threading.Lock()
        self._stop = threading.Event()
        self._consecutive_failures = 0
        self._exposed_points: set[str] = set()
        self._seed = 0
        self._answer_mode = AnswerMode.AUTO
        self._reveal = False
        self._console = None
        try:
            from rich.console import Console

            self._console = Console()
        except Exception:
            self._console = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def run(
        self,
        plan: QuizPlan,
        *,
        stream: bool = True,
        dry_run: bool = False,
        hide_answers: bool = False,
        answers: str = "auto",
        reveal: bool = False,
    ) -> RunResult:
        started = time.time()
        batch_id = new_batch_id()
        self._seed = plan.seed
        self._answer_mode = resolve_answer_mode(
            "hidden" if hide_answers else answers,
            is_mock=plan.mode == "mock",
        )
        self._reveal = reveal
        rng = random.Random(plan.seed)
        used_counts = used_target_counts(self.settings)
        pools = TextbookPools(self.context, used_counts)
        exposed = exposed_points(self.settings)
        self._exposed_points = exposed
        used: set[str] = set()
        specs, warnings = build_specs(
            plan,
            self.context,
            pools,
            self.extension_pool,
            self.mdict,
            rng,
            used,
            item_bank=self.item_bank,
            exposed_points=exposed,
            source_mix=plan.source_mix,
        )
        if not specs:
            raise QuestionBuildError("没有可生成的题目：请检查题量/课次/题型/资源。")

        questions: dict[int, GeneratedQuestion] = {}
        items: dict[int, Item] = {}
        failures: list[RunFailure] = []
        self._consecutive_failures = 0
        self._stop.clear()

        instant_specs = [
            spec
            for spec in specs
            if self._is_instant_spec(spec)
        ]
        instant_seqs = {spec.seq for spec in instant_specs}
        llm_specs = [
            spec
            for spec in specs
            if spec.seq not in instant_seqs and spec.source_kind == "new"
        ]

        executor = ThreadPoolExecutor(max_workers=self.workers)
        futures: dict[Future, QuestionSpec] = {}
        try:
            for spec in llm_specs:
                futures[executor.submit(self._run_llm, spec, pools, rng, batch_id)] = spec

            # Original/variant/local questions are instant; print them while LLM runs.
            for spec in instant_specs:
                try:
                    question, item = self._generate_instant(
                        spec, pools, rng, batch_id, exposed
                    )
                    if self._is_duplicate(question):
                        warnings.append(f"{question.question_id}: 与已有题库重复，已跳过。")
                        continue
                    questions[spec.seq] = question
                    items[spec.seq] = item
                    if stream:
                        self._print_question(item, spec, len(specs))
                except _NeedsLLM:
                    spec.source_kind = "new"
                    futures[
                        executor.submit(self._run_llm, spec, pools, rng, batch_id)
                    ] = spec
                except Exception as exc:
                    failures.append(
                        RunFailure(spec.seq, spec.qtype.value, f"{type(exc).__name__}: {exc}")
                    )

            for future in as_completed(futures):
                spec = futures[future]
                try:
                    question, item = future.result()
                    if self._is_duplicate(question):
                        warnings.append(f"{question.question_id}: 与已有题库重复，已跳过。")
                        continue
                    questions[spec.seq] = question
                    items[spec.seq] = item
                    self._consecutive_failures = 0
                    if stream:
                        self._print_question(item, spec, len(specs))
                except FatalLLMError as exc:
                    failures.append(RunFailure(spec.seq, spec.qtype.value, str(exc)))
                    self._stop.set()
                except Exception as exc:
                    failures.append(
                        RunFailure(spec.seq, spec.qtype.value, f"{type(exc).__name__}: {exc}")
                    )
                    if _is_transient_llm_error(str(exc)):
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= 3:
                            self._stop.set()
                    else:
                        # Structural misses are per-question, not a reason to
                        # cancel the rest of the batch.
                        self._consecutive_failures = 0
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        ordered = [questions[key] for key in sorted(questions)]
        ordered_items = [items[key] for key in sorted(items)]
        if not ordered:
            raise QuestionBuildError(
                "所有题目都生成失败。"
                + (" 首个错误：" + failures[0].message if failures else "")
            )
        batch = self._build_batch(
            plan, batch_id, ordered, ordered_items, warnings, started
        )
        json_path = md_path = ""
        duplicates: list[str] = []
        if not dry_run:
            json_path_obj, md_path_obj = archive_batch(self.settings, batch)
            result = insert_batch(
                self.settings,
                batch,
                json_path=json_path_obj,
                md_path=md_path_obj,
                report=None,
            )
            json_path = str(json_path_obj)
            md_path = str(md_path_obj)
            duplicates = list(result.get("duplicates", []))
            record_exposures(self.settings, batch)
        duration = int((time.time() - started) * 1000)
        batch.batch_meta.duration_ms = duration
        return RunResult(
            batch=batch,
            failures=failures,
            warnings=warnings,
            json_path=json_path,
            md_path=md_path,
            duplicates=duplicates,
            duration_ms=duration,
            items=ordered_items,
        )

    def _is_instant_spec(self, spec: QuestionSpec) -> bool:
        if spec.source_kind in {"original", "variant"} or spec.qtype.is_local:
            return True
        plugin = question_types.maybe_get(spec.type_name)
        return bool(
            spec.is_plugin_type
            and plugin is not None
            and getattr(plugin, "is_local", False)
        )

    def _plugin_context(
        self,
        pools: TextbookPools,
        rng: random.Random,
        batch_id: str,
        exposed: set[str],
    ) -> PluginContext:
        return PluginContext(
            settings=self.settings,
            context=self.context,
            pools=pools,
            rng=rng,
            llm=self.llm,
            locks=self.locks,
            mdict=self.mdict,
            extension_pool=self.extension_pool,
            item_bank=self.item_bank,
            batch_id=batch_id,
            seed=self._seed,
            exposed_points=set(exposed),
        )

    def _generate_instant(
        self,
        spec: QuestionSpec,
        pools: TextbookPools,
        rng: random.Random,
        batch_id: str,
        exposed: set[str],
    ) -> tuple[GeneratedQuestion, Item]:
        plugin = question_types.maybe_get(spec.type_name)
        if spec.is_plugin_type and plugin is not None:
            if not getattr(plugin, "is_local", False):
                raise _NeedsLLM("plugin type requires LLM")
            item = plugin.build_local(
                spec, self._plugin_context(pools, rng, batch_id, exposed)
            )
            if item is None:
                raise QuestionBuildError(f"插件题型 {spec.type_name} 未产出 Item")
            return self._finalize_item(item, spec)
        item = spec.original_item
        if spec.source_kind == "original" and item is not None:
            try:
                return self._finalize(
                    generate_original(spec, item, batch_id=batch_id), spec
                )
            except Exception:
                spec.source_kind = "variant"
        if spec.source_kind == "variant" and item is not None:
            try:
                return self._finalize(
                    generate_variant(
                        spec,
                        item,
                        pools,
                        rng,
                        batch_id=batch_id,
                        point_exposed=spec.point_key in exposed,
                    ),
                    spec,
                )
            except Exception:
                if spec.point_key not in exposed:
                    spec.source_kind = "original"
                    try:
                        return self._finalize(
                            generate_original(spec, item, batch_id=batch_id), spec
                        )
                    except Exception:
                        pass
                spec.source_kind = "new"
        if spec.qtype.is_local:
            spec.source_kind = "new"
            return self._finalize(
                generate_local(spec, pools, rng, batch_id=batch_id), spec
            )
        raise _NeedsLLM("需要 LLM 生成新题")

    # ------------------------------------------------------------------
    # LLM task
    # ------------------------------------------------------------------
    def _run_llm(
        self,
        spec: QuestionSpec,
        pools: TextbookPools,
        rng: random.Random,
        batch_id: str,
    ) -> tuple[GeneratedQuestion, Item]:
        if self._stop.is_set():
            raise QuestionBuildError("已熔断，取消后续 LLM 请求")
        if self.llm is None:
            raise FatalLLMError("未配置 LLM")
        spec.expected_text = spec.expected_text or spec.correct_text
        spec.correct_text = spec.expected_text
        spec.pending_defects = []
        spec.source_kind = "new"
        last_error: Exception | None = None
        for attempt in range(2):
            if self._stop.is_set():
                raise QuestionBuildError("已熔断，取消后续 LLM 请求")
            try:
                return self._finalize(
                    generate_llm(
                        spec,
                        self.llm,
                        self.settings,
                        pools,
                        rng,
                        extension_pool=self.extension_pool,
                        mdict=self.mdict,
                        locks=self.locks,
                        batch_id=batch_id,
                    ),
                    spec,
                )
            except Exception as exc:
                last_error = exc
                message = str(exc)
                if _is_fatal_llm_error(message):
                    raise FatalLLMError(message) from exc
                if attempt == 0:
                    time.sleep(0.4)
                    continue
        item = spec.original_item
        if item is not None:
            if spec.point_key not in self._exposed_points:
                try:
                    spec.source_kind = "original"
                    return self._finalize(
                        generate_original(spec, item, batch_id=batch_id), spec
                    )
                except Exception:
                    pass
            try:
                spec.source_kind = "variant"
                return self._finalize(
                    generate_variant(
                        spec,
                        item,
                        pools,
                        rng,
                        batch_id=batch_id,
                        point_exposed=spec.point_key in self._exposed_points,
                    ),
                    spec,
                )
            except Exception:
                pass
        if spec.qtype in {QuestionType.GRAMMAR_FILL, QuestionType.EXPRESSION_CHOICE}:
            try:
                spec.source_kind = "variant"
                return self._finalize(
                    generate_local_grammar(
                        spec, pools, rng, locks=self.locks, batch_id=batch_id
                    ),
                    spec,
                )
            except Exception:
                pass
        raise QuestionBuildError(str(last_error or "unknown error"))

    def _finalize_item(
        self, item: Item, spec: QuestionSpec
    ) -> tuple[GeneratedQuestion, Item]:
        defects = validate_item(item)
        if defects:
            raise ItemBuildError(defects[0].code, defects[0].message)
        projected = project_generated_question(
            item, batch_id=str(item.meta.provenance.get("batch_id") or "")
        )
        # Keep the sampled lesson when the Item has no textbook lesson.
        if not item.meta.lessons and spec.textbook_lesson:
            item.meta.lessons = [spec.textbook_lesson]
            refresh_hashes(item)
            projected.textbook_lesson = spec.textbook_lesson
            projected.lesson = spec.course_lesson
        return projected, item

    def _finalize(
        self, question: GeneratedQuestion, spec: QuestionSpec
    ) -> tuple[GeneratedQuestion, Item]:
        """Build and validate the canonical Item, then derive the legacy view."""
        item = build_item_from_generated_question(question, seed=self._seed)
        return self._finalize_item(item, spec)

    def _is_duplicate(self, question: GeneratedQuestion) -> bool:
        if question.source_kind in {"original", "variant"}:
            return False
        return self.context.novelty.is_duplicate_hash(question_content_hash(question))

    # ------------------------------------------------------------------
    # Batch assembly / display
    # ------------------------------------------------------------------
    def _build_batch(
        self,
        plan: QuizPlan,
        batch_id: str,
        questions: list[GeneratedQuestion],
        items: list[Item],
        warnings: list[str],
        started: float,
    ) -> GeneratedBatch:
        actual_plan: dict[str, int] = {}
        if items:
            for item in items:
                actual_plan[item.type] = actual_plan.get(item.type, 0) + 1
        else:
            for question in questions:
                actual_plan[question.type.value] = (
                    actual_plan.get(question.type.value, 0) + 1
                )
        model = (
            f"local+{self.settings.llm_model}"
            if any(not q.type.is_local for q in questions)
            else "local"
        )
        meta = BatchMeta(
            id=batch_id,
            semester=self.settings.active_semester,
            request=plan.raw,
            mode=plan.mode,
            scope=plan.scope,
            topic=plan.topic,
            type_plan=actual_plan,
            lessons=plan.lessons,
            model=model,
            rules_version=int(self.context.question_types.meta.get("version", 1)),
            ruleset_hash=self._ruleset_hash(),
            created_at=now_iso(),
            warnings=list(warnings),
            duration_ms=int((time.time() - started) * 1000),
            expected=plan.total,
            accepted=len(questions),
            seed=plan.seed,
            schema_version="1",
            answers_mode=self._answer_mode.value,
            metrics=self._metrics(questions, plan, items),
        )
        return GeneratedBatch(
            batch_meta=meta,
            questions=questions,
            items=[item_to_json(item) for item in items],
        )

    def _metrics(
        self,
        questions: list[GeneratedQuestion],
        plan: QuizPlan,
        items: list[Item] | None = None,
    ) -> dict[str, Any]:
        items = items or []
        source_counts: dict[str, int] = {}
        domain_counts: dict[str, int] = {}
        answer_distribution: dict[str, int] = {}
        for index, item in enumerate(items, start=1):
            source_counts[item.meta.source] = source_counts.get(item.meta.source, 0) + 1
            domain_counts[item.meta.domain] = domain_counts.get(item.meta.domain, 0) + 1
            try:
                rendered = render_item(item, seed=self._seed, number=index)
                answer_distribution[rendered.answer_key] = (
                    answer_distribution.get(rendered.answer_key, 0) + 1
                )
            except Exception:
                pass
        overlap_values: list[float] = []
        allowed = self._allowed_options()
        out_of_level = 0
        option_count = 0
        bare_framework = 0
        same_field_no_clue = 0
        for question in questions:
            references = list(question.provenance.get("reference_stems") or [])
            anchor = question.provenance.get("anchor_example")
            if anchor:
                references.append(str(anchor))
            if references:
                overlap_values.append(max_overlap(question.stem, references))
            clue = str(question.provenance.get("clue") or "")
            if not self.locks.is_clue_present(question.stem, clue):
                bare_framework += 1
            if question.type == QuestionType.VOCAB_CHOICE:
                target_fields = set(self.locks.fields_of(question.correct_option_text))
                option_fields = {
                    field
                    for option in question.options
                    if option.role.value == "distractor"
                    for field in self.locks.fields_of(option.text)
                }
                if target_fields and option_fields and target_fields & option_fields and not clue:
                    same_field_no_clue += 1
                for option in question.options:
                    option_count += 1
                    key = normalize_text(option.text)
                    if key not in allowed and style_key(option.text) != "particle":
                        out_of_level += 1
        point_keys = {
            item.meta.point for item in items if item.meta.point
        } or {question.point_key for question in questions if question.point_key}
        metrics = {
            "source_counts": source_counts,
            "domain_counts": domain_counts,
            "source_mix_target": {
                ("ai" if key == "new" else key): value
                for key, value in plan.source_mix.items()
            },
            "point_coverage": round(len(point_keys) / max(1, len(questions)), 3),
            "stem_overlap_max": round(max(overlap_values), 3) if overlap_values else 0.0,
            "stem_overlap_avg": round(
                sum(overlap_values) / len(overlap_values), 3
            ) if overlap_values else 0.0,
            "out_of_level_rate": round(out_of_level / max(1, option_count), 3),
            "bare_framework_rate": round(bare_framework / max(1, len(questions)), 3),
            "same_field_no_clue_rate": round(
                same_field_no_clue / max(1, len(questions)), 3
            ),
            "answer_distribution": answer_distribution,
            "difficulty_distribution": {
                "basic": sum(1 for q in questions if 0 < q.textbook_lesson <= 43),
                "standard": sum(1 for q in questions if q.textbook_lesson >= 44 or q.textbook_lesson == 0),
            },
        }
        return metrics

    def _allowed_options(self) -> set[str]:
        allowed: set[str] = set()
        for entry in self.context.vocab.all_entries():
            allowed.add(normalize_text(entry.written))
            allowed.add(normalize_text(entry.kana))
        for point in self.context.grammar.points:
            for form in point.approved_answer_forms:
                allowed.add(normalize_text(form))
        if self.extension_pool is not None:
            for entry in self.extension_pool.entries:
                allowed.add(normalize_text(entry.written))
        return allowed

    def _ruleset_hash(self) -> str:
        digest = hashlib.sha256()
        for path in (self.settings.question_types_path, self.settings.common_mistakes_path):
            if path.exists():
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def _print(self, text: str) -> None:
        with self._print_lock:
            if self._console is not None:
                self._console.print(text, markup=False, highlight=False)
            else:
                print(text, flush=True)

    def _print_question(
        self,
        item: Item,
        spec: QuestionSpec,
        total: int,
    ) -> None:
        rendered = render_item(item, seed=self._seed, number=spec.seq)
        if self._answer_mode == AnswerMode.ONLY:
            text = key_view(rendered)
        elif self._answer_mode == AnswerMode.INLINE:
            text = inline_view(rendered)
        else:
            text = stem_view(rendered)
            if self._reveal and self._answer_mode == AnswerMode.HIDDEN:
                text = text + "\n" + key_view(rendered)
        first = f"[{rendered.number}] "
        if text.startswith(first):
            text = f"[{spec.seq}/{total}] " + text[len(first) :]
        self._print(text)


def _is_fatal_llm_error(message: str) -> bool:
    text = (message or "").lower()
    return any(
        marker in text
        for marker in ("401", "403", "authentication", "invalid api key", "unauthorized")
    )


def _is_transient_llm_error(message: str) -> bool:
    text = (message or "").lower()
    return any(
        marker in text
        for marker in (
            "timeout",
            "timed out",
            "connection",
            "connect",
            "network",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
            "overloaded",
            "temporarily",
        )
    )
