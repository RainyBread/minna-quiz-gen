"""Built-in Markdown exporter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.items import Item
from core.registry import exporters
from core.render import (
    render_answer_book,
    render_item,
    render_question_book,
)


class MarkdownExporter:
    name = "markdown"

    def export(
        self,
        items: list[Item],
        out: Path,
        *,
        seed: int = 0,
        title: str = "JLPTCraft",
        **kwargs: Any,
    ) -> dict[str, Path]:
        rendered = [
            render_item(item, seed=seed, number=index)
            for index, item in enumerate(items, start=1)
        ]
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        questions = out / "questions.md"
        answers = out / "answers.md"
        questions.write_text(
            render_question_book(rendered, title=f"{title} 试题册"),
            encoding="utf-8",
        )
        answers.write_text(
            render_answer_book(rendered, title=f"{title} 答案册"),
            encoding="utf-8",
        )
        return {"questions": questions, "answers": answers}


exporters.register("markdown", MarkdownExporter())
