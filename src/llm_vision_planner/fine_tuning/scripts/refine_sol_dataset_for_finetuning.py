#!/usr/bin/env python3
"""Redact non-safety numeric values from Sol reasoning without rewriting it."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REASONING_FIELDS = (
    "teacher_reason",
    "teacher_selection_reason",
    "teacher_trajectory_reasoning",
    "teacher_trajectory_reasoning_raw",
)

# Match standalone numeric values, including signed coordinates and decimals.
# Digits inside identifiers such as route-002, person-1, and chair-2 are preserved.
NUMERIC_VALUE = re.compile(
    r"(?<![A-Za-z0-9_-])[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?![A-Za-z0-9_-])"
)
REDACTION = "[scene-specific value]"


def clear_numeric_values(text: str) -> str:
    """Redact standalone values before ``Safety:`` and preserve the safety section."""
    reasoning, separator, safety = text.partition("Safety:")
    return NUMERIC_VALUE.sub(REDACTION, reasoning) + separator + safety


def refine_row(row: dict[str, object]) -> tuple[bool, int]:
    if "sol" not in str(row.get("teacher_model", "")).lower():
        return False, 0

    replacements = 0
    for field in REASONING_FIELDS:
        value = row.get(field)
        if not isinstance(value, str):
            continue
        reasoning, _, _ = value.partition("Safety:")
        replacements += len(NUMERIC_VALUE.findall(reasoning))
        row[field] = clear_numeric_values(value)
    return True, replacements


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must differ; the raw dataset is never overwritten")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    total = 0
    refined = 0
    replacements = 0
    try:
        with args.input.open(encoding="utf-8") as source, temporary.open(
            "w", encoding="utf-8"
        ) as destination:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
                total += 1
                is_sol, row_replacements = refine_row(row)
                refined += int(is_sol)
                replacements += row_replacements
                destination.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    print(
        f"Wrote {total} records to {args.output}; refined {refined} Sol records; "
        f"redacted {replacements} numeric values"
    )


if __name__ == "__main__":
    main()
