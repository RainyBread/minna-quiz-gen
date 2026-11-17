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
    ) -> RunResult:
        started = time.time()
        batch_id = new_batch_id()
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
        failures: list[RunFailure] = []
        self._consecutive_failures = 0
        self._stop.clear()

        instant_specs = [
            spec
            for spec in specs
            if spec.source_kind in {"original", "variant"} or spec.qtype.is_local
        ]
        llm_specs = [
            spec
            for spec in specs
            if not spec.qtype.is_local and spec.source_kind == "new"
        ]

        executor = ThreadPoolExecutor(max_workers=self.workers)
        futures: dict[Future, QuestionSpec] = {}
        try:
            for spec in llm_specs:
                futures[executor.submit(self._run_llm, spec, pools, rng, batch_id)] = spec

            # Original/variant/local questions are instant; print them while LLM runs.
            for spec in instant_specs:
                try:
                    question = self._generate_instant(
                        spec, pools, rng, batch_id, exposed
                    )
                    if self._is_duplicate(question):
                        warnings.append(f"{question.question_id}: 与已有题库重复，已跳过。")
                        continue
                    questions[spec.seq] = question
                    if stream:
                        self._print_question(question, spec, len(specs), hide_answers)
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
                    question = future.result()
                    if self._is_duplicate(question):
                        warnings.append(f"{question.question_id}: 与已有题库重复，已跳过。")
                        continue
                    questions[spec.seq] = question
                    self._consecutive_failures = 0
                    if stream:
                        self._print_question(question, spec, len(specs), hide_answers)
                except FatalLLMError as exc:
                    failures.append(RunFailure(spec.seq, spec.qtype.value, str(exc)))
                    self._stop.set()
                except Exception as exc:
                    failures.append(
                        RunFailure(spec.seq, spec.qtype.value, f"{type(exc).__name__}: {exc}")
                    )
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= 3:
                        self._stop.set()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        ordered = [questions[key] for key in sorted(questions)]
        if not ordered:
            raise QuestionBuildError(
                "所有题目都生成失败。"
                + (" 首个错误：" + failures[0].message if failures else "")
            )
        batch = self._build_batch(plan, batch_id, ordered, warnings, started)
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
        )

    def _generate_instant(
        self,
        spec: QuestionSpec,
        pools: TextbookPools,
        rng: random.Random,
        batch_id: str,
        exposed: set[str],
    ) -> GeneratedQuestion:
        item = spec.original_item
        if spec.source_kind == "original" and item is not None:
            try:
                return generate_original(spec, item, batch_id=batch_id)
            except Exception:
                spec.source_kind = "variant"
        if spec.source_kind == "variant" and item is not None:
            try:
                return generate_variant(
                    spec,
                    item,
                    pools,
                    rng,
                    batch_id=batch_id,
                    point_exposed=spec.point_key in exposed,
                )
            except Exception:
                if spec.point_key not in exposed:
                    spec.source_kind = "original"
                    try:
                        return generate_original(spec, item, batch_id=batch_id)
                    except Exception:
                        pass
                spec.source_kind = "new"
        if spec.qtype.is_local:
            spec.source_kind = "new"
            return generate_local(spec, pools, rng, batch_id=batch_id)
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
    ) -> GeneratedQuestion:
        if self._stop.is_set():
            raise QuestionBuildError("已熔断，取消后续 LLM 请求")
        if self.llm is None:
            raise FatalLLMError("未配置 LLM")
        spec.source_kind = "new"
        last_error: Exception | None = None
        for attempt in range(2):
            if self._stop.is_set():
                raise QuestionBuildError("已熔断，取消后续 LLM 请求")
            try:
                return generate_llm(
                    spec,
                    self.llm,
                    self.settings,
                    pools,
                    rng,
                    extension_pool=self.extension_pool,
                    mdict=self.mdict,
                    locks=self.locks,
                    batch_id=batch_id,
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
                    return generate_original(spec, item, batch_id=batch_id)
                except Exception:
                    pass
            try:
                spec.source_kind = "variant"
                return generate_variant(
                    spec,
                    item,
                    pools,
                    rng,
                    batch_id=batch_id,
                    point_exposed=spec.point_key in self._exposed_points,
                )
            except Exception:
                pass
        if spec.qtype in {QuestionType.GRAMMAR_FILL, QuestionType.EXPRESSION_CHOICE}:
            try:
                spec.source_kind = "variant"
                return generate_local_grammar(spec, pools, rng, locks=self.locks, batch_id=batch_id)
            except Exception:
                pass
        raise QuestionBuildError(str(last_error or "unknown error"))

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
        warnings: list[str],
        started: float,
    ) -> GeneratedBatch:
        actual_plan: dict[str, int] = {}
        for question in questions:
            actual_plan[question.type.value] = actual_plan.get(question.type.value, 0) + 1
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
            metrics=self._metrics(questions, plan),
        )
        return GeneratedBatch(batch_meta=meta, questions=questions)

    def _metrics(self, questions: list[GeneratedQuestion], plan: QuizPlan) -> dict[str, Any]:
        source_counts: dict[str, int] = {}
        answer_distribution: dict[str, int] = {}
        overlap_values: list[float] = []
        allowed = self._allowed_options()
        out_of_level = 0
        option_count = 0
        bare_framework = 0
        same_field_no_clue = 0
        for question in questions:
            source_counts[question.source_kind] = source_counts.get(question.source_kind, 0) + 1
            answer_distribution[question.answer_key] = (
                answer_distribution.get(question.answer_key, 0) + 1
            )
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
        point_keys = {question.point_key for question in questions if question.point_key}
        metrics = {
            "source_counts": source_counts,
            "source_mix_target": dict(plan.source_mix),
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
        question: GeneratedQuestion,
        spec: QuestionSpec,
        total: int,
        hide_answers: bool,
    ) -> None:
        scope = {
            "in_syllabus": "课内",
            "mdict_distractors": "课内+拓展干扰",
            "extend": "拓展",
        }.get(question.scope, question.scope)
        lesson = f"L{question.textbook_lesson}" if question.textbook_lesson else "跨课"
        lines = [
            f"[{spec.seq}/{total}] {TYPE_LABELS_ZH.get(question.type.value, question.type.value)} · {lesson} · {scope}"
        ]
        if question.topic:
            lines[0] += f" · 主题：{question.topic}"
        lines.append(question.stem.replace("[[blank]]", "（　　）"))
        lines.extend(f"{option.key}. {option.text}" for option in question.options)
        if not hide_answers:
            lines.append(f"答案：{question.answer_key}. {question.correct_option_text}")
            if question.explanation_zh:
                lines.append(f"解析：{question.explanation_zh}")
        lines.append("")
        self._print("\n".join(lines))


def _is_fatal_llm_error(message: str) -> bool:
    text = (message or "").lower()
    return any(
        marker in text
        for marker in ("401", "403", "authentication", "invalid api key", "unauthorized")
    )
