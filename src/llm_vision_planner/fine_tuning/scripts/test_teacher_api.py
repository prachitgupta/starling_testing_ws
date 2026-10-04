#!/usr/bin/env python3
"""Run one real HRRT teacher request and project its token cost."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict


SYSTEM_PROMPT = (
    "Select exactly one supplied route_id that best satisfies the operator preference. "
    "Never create coordinates or a new route. Return concise reasoning."
)


@dataclass(frozen=True)
class Pricing:
    input_per_million: float
    cached_input_per_million: float
    cache_write_per_million: float
    output_per_million: float


# Standard short-context API prices in USD per million tokens, checked 2026-09-30.
# Use the command-line price overrides when testing a new or differently priced model.
PRICES = {
    "gpt-6-astra": Pricing(10.00, 1.00, 12.50, 50.00),
    "gpt-6.1-sol": Pricing(2.00, 0.10, 2.50, 10.00),
    "gpt-6-sol": Pricing(2.00, 0.20, 2.50, 10.00),
    "gpt-6-luna": Pricing(0.10, 0.01, 0.125, 0.50),
    "gpt-5.4-mini": Pricing(0.75, 0.075, 0.75, 4.50),
    "gpt-5.4-nano": Pricing(0.20, 0.02, 0.20, 1.25),
    "gpt-5.4": Pricing(2.50, 0.25, 2.50, 15.00),
}


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected_route_id: str
    reason: str


def compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def load_sample(path: Path, sample_index: int) -> dict[str, Any]:
    if sample_index < 0:
        raise ValueError("--sample-index must be non-negative")
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index == sample_index:
                row = json.loads(line)
                return {
                    "sample_id": row.get("sample_id", f"row-{index}"),
                    "preference": row["preference"]["text"],
                    "route_cards": row["route_cards"],
                }
    raise IndexError(f"{path} does not contain sample index {sample_index}")


def model_pricing(args: argparse.Namespace) -> Pricing | None:
    overrides = (
        args.input_price,
        args.cached_input_price,
        args.cache_write_price,
        args.output_price,
    )
    if any(value is not None for value in overrides):
        if not all(value is not None for value in overrides):
            raise ValueError("provide all four price override arguments, or none")
        return Pricing(*overrides)
    for name in sorted(PRICES, key=len, reverse=True):
        if args.model == name or args.model.startswith(name + "-"):
            return PRICES[name]
    return None


def usage_value(obj: object, name: str) -> int:
    return int(getattr(obj, name, 0) or 0)


def calculate_cost(usage: object, pricing: Pricing) -> float:
    input_tokens = usage_value(usage, "input_tokens")
    output_tokens = usage_value(usage, "output_tokens")
    input_details = getattr(usage, "input_tokens_details", None)
    cached_tokens = usage_value(input_details, "cached_tokens")
    cache_write_tokens = usage_value(input_details, "cache_write_tokens")
    uncached_tokens = max(0, input_tokens - cached_tokens - cache_write_tokens)
    return (
        uncached_tokens * pricing.input_per_million
        + cached_tokens * pricing.cached_input_per_million
        + cache_write_tokens * pricing.cache_write_per_million
        + output_tokens * pricing.output_per_million
    ) / 1_000_000


def run(args: argparse.Namespace) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("set OPENAI_API_KEY before running this script")
    if args.runs < 1 or args.project_calls < 1:
        raise ValueError("--runs and --project-calls must be positive")

    sample = load_sample(args.dataset, args.sample_index)
    user_prompt = compact(
        {"preference": sample["preference"], "route_cards": sample["route_cards"]}
    )
    if args.show_prompt:
        print("SYSTEM PROMPT")
        print(SYSTEM_PROMPT)
        print("\nUSER PROMPT")
        print(user_prompt)

    request: dict[str, Any] = {
        "model": args.model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "text_format": Selection,
        "max_output_tokens": args.max_output_tokens,
    }
    if args.reasoning_effort != "default":
        request["reasoning"] = {"effort": args.reasoning_effort}

    client = OpenAI()
    responses = []
    for run_index in range(args.runs):
        response = client.responses.parse(**request)
        if response.output_parsed is None or response.usage is None:
            raise RuntimeError("API returned no parsed selection or token usage")
        responses.append(response)
        usage = response.usage
        reasoning_tokens = usage_value(
            getattr(usage, "output_tokens_details", None), "reasoning_tokens"
        )
        print(f"\nRUN {run_index + 1}")
        print(f"sample: {sample['sample_id']}")
        print(f"selection: {response.output_parsed.model_dump_json()}")
        print(
            "tokens: "
            f"input={usage_value(usage, 'input_tokens')}, "
            f"output={usage_value(usage, 'output_tokens')}, "
            f"reasoning={reasoning_tokens}, "
            f"total={usage_value(usage, 'total_tokens')}"
        )

    pricing = model_pricing(args)
    average_input = mean(usage_value(item.usage, "input_tokens") for item in responses)
    average_output = mean(usage_value(item.usage, "output_tokens") for item in responses)
    average_reasoning = mean(
        usage_value(getattr(item.usage, "output_tokens_details", None), "reasoning_tokens")
        for item in responses
    )
    print("\nAVERAGE OBSERVED USAGE")
    print(
        f"input={average_input:.1f}, output={average_output:.1f}, "
        f"reasoning={average_reasoning:.1f} tokens/call"
    )
    if pricing is None:
        print(
            f"No built-in price for {args.model!r}; rerun with all four --*-price overrides."
        )
        return

    costs = [calculate_cost(item.usage, pricing) for item in responses]
    average_cost = mean(costs)
    projected_standard = average_cost * args.project_calls
    print("\nESTIMATED COST")
    print(f"observed average: ${average_cost:.6f} per call")
    print(f"{args.project_calls} Standard API calls: ${projected_standard:.2f}")
    print(f"{args.project_calls} Batch API calls:    ${projected_standard * 0.5:.2f}")
    print("The API dashboard remains the authoritative source for the final billed amount.")


def build_parser() -> argparse.ArgumentParser:
    default_dataset = Path(__file__).resolve().parents[1] / "datasets" / "hrrt_teacher_raw.jsonl"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument(
        "--reasoning-effort",
        choices=("default", "none", "low", "medium", "high", "xhigh", "max"),
        default="default",
        help="Use the model default, or explicitly choose a reasoning effort.",
    )
    parser.add_argument("--dataset", type=Path, default=default_dataset)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--project-calls", type=int, default=2000)
    parser.add_argument("--max-output-tokens", type=int, default=2000)
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument("--input-price", type=float)
    parser.add_argument("--cached-input-price", type=float)
    parser.add_argument("--cache-write-price", type=float)
    parser.add_argument("--output-price", type=float)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
