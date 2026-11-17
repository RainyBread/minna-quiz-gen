"""Demo exporter: proves a new exporter is one file + one registration line."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.items import Item
from core.registry import exporters


class DemoExporter:
    name = "demo"

    def export(
        self,
        items: list[Item],
        out: Path,
        *,
        seed: int = 0,
        title: str = "JLPTCraft",
        **kwargs: Any,
    ) -> Path:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        target = out / "demo.txt"
        lines = [
            f"{title} | items={len(items)} | seed={seed}",
            *[f"{item.id}\t{item.type}\t{item.key.value}" for item in items],
        ]
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target


exporters.register("demo", DemoExporter())
