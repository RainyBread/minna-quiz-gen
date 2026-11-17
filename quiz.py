#!/usr/bin/env python
"""Thin ``quiz`` entry point; ``python quiz.py audit [BATCH_ID]`` works."""

from __future__ import annotations

import sys

from main import audit_command, main


def _run() -> int:
    args = sys.argv[1:]
    if args and args[0] == "audit":
        rest = args[1:]
        json_output = "--json" in rest
        batch_id = next((item for item in rest if not item.startswith("-")), "")
        return audit_command(batch_id, json_output)
    return main(args)


if __name__ == "__main__":
    raise SystemExit(_run())
