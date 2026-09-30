#!/usr/bin/env python3
"""Create scene-isolated human-review assignments for HRRT teacher labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from typing import Dict, List, Mapping


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: List[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
    temporary.replace(path)


def scene_partition(scene_ids: List[int], validation_ratio: float, test_ratio: float, seed: int):
    shuffled = list(scene_ids)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_ratio))
    test_count = max(1, round(len(shuffled) * test_ratio))
    if validation_count + test_count >= len(shuffled):
        raise ValueError("Need more scenes for non-empty train, validation, and test partitions")
    validation = set(shuffled[:validation_count])
    test = set(shuffled[validation_count : validation_count + test_count])
    train = set(shuffled[validation_count + test_count :])
    return train, validation, test


def build_manifest(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    by_scene: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_scene[int(row["scene_id"])].append(row)
    train_scenes, validation_scenes, test_scenes = scene_partition(
        sorted(by_scene), args.validation_scene_ratio, args.test_scene_ratio, args.seed
    )

    manifest: List[Dict[str, object]] = []
    for scene_id in sorted(validation_scenes | test_scenes):
        split = "validation" if scene_id in validation_scenes else "test"
        for row in sorted(by_scene[scene_id], key=lambda item: str(item["sample_id"])):
            manifest.append(
                {
                    "sample_id": row["sample_id"],
                    "scene_id": scene_id,
                    "review_split": split,
                    "preference_type": row["preference"]["preference_type"],
                }
            )

    train_candidates = [row for scene in sorted(train_scenes) for row in by_scene[scene]]
    strata: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in train_candidates:
        environment = row["environment"]
        key = (
            f"profile={environment.get('scene_profile', 'unknown')}|"
            f"obstacles={len(environment['obstacles'])}|"
            f"preference={row['preference']['preference_type']}|"
            f"routes={len(row['routes'])}"
        )
        strata[key].append(row)
    target = round(len(train_candidates) * args.train_audit_ratio)
    rng = random.Random(args.seed + 1)
    selected: Dict[str, Dict[str, object]] = {}
    while len(selected) < target and any(strata.values()):
        for key in sorted(strata):
            bucket = strata[key]
            if bucket and len(selected) < target:
                row = bucket.pop(rng.randrange(len(bucket)))
                selected[str(row["sample_id"])] = row
    for row in sorted(selected.values(), key=lambda item: str(item["sample_id"])):
        manifest.append(
            {
                "sample_id": row["sample_id"],
                "scene_id": row["scene_id"],
                "review_split": "train_correction",
                "preference_type": row["preference"]["preference_type"],
            }
        )

    manifest.sort(key=lambda item: (str(item["review_split"]), int(item["scene_id"]), str(item["sample_id"])))
    write_jsonl(args.output, manifest)
    split_counts = {split: sum(row["review_split"] == split for row in manifest) for split in ("train_correction", "validation", "test")}
    print(json.dumps({"records": len(manifest), "split_counts": split_counts}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("fine_tuning/datasets/hrrt_teacher_raw.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("fine_tuning/datasets/hrrt_human_audit.jsonl"))
    parser.add_argument("--train-audit-ratio", type=float, default=0.10)
    parser.add_argument("--validation-scene-ratio", type=float, default=0.10)
    parser.add_argument("--test-scene-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    build_manifest(build_parser().parse_args())
