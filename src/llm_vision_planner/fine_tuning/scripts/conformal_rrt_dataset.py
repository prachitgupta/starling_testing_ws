#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from pathlib import Path

from dataset_generator import (
    DEFAULT_CLEARANCE_M,
    DEFAULT_DATASET_DIR,
    DEFAULT_WORKSPACE,
    load_prompt_generator,
    prompt_from_current_generator,
    sample_environment,
)
from rrt import plan_rrt


DEFAULT_OUTPUT = DEFAULT_DATASET_DIR / "conformal_rrt_calibration_dataset.csv"
DEFAULT_VLLM_BASE_URL = "http://172.22.224.93:8000/v1"
DEFAULT_LLAMA_MODEL_NAME = "rrt_planner"


def path_length(path):
    return sum(math.hypot(path[i]["x"] - path[i - 1]["x"], path[i]["y"] - path[i - 1]["y"]) for i in range(1, len(path)))


def interpolate_path(path, count):
    if count <= 1:
        return [dict(path[0])]
    total = path_length(path)
    if total <= 1e-9:
        return [dict(path[0]) for _ in range(count)]

    targets = [total * i / (count - 1) for i in range(count)]
    samples = []
    seg_index = 1
    seg_start_distance = 0.0
    for target in targets:
        while seg_index < len(path) - 1:
            segment = math.hypot(path[seg_index]["x"] - path[seg_index - 1]["x"], path[seg_index]["y"] - path[seg_index - 1]["y"])
            if seg_start_distance + segment >= target:
                break
            seg_start_distance += segment
            seg_index += 1

        a = path[seg_index - 1]
        b = path[seg_index]
        segment = max(math.hypot(b["x"] - a["x"], b["y"] - a["y"]), 1e-9)
        ratio = min(1.0, max(0.0, (target - seg_start_distance) / segment))
        samples.append(
            {
                "x": round(a["x"] + (b["x"] - a["x"]) * ratio, 4),
                "y": round(a["y"] + (b["y"] - a["y"]) * ratio, 4),
                "z": round(float(a.get("z", b.get("z", -0.25))), 4),
            }
        )
    return samples


def velocities(path, dt):
    return [
        (
            (path[i]["x"] - path[i - 1]["x"]) / dt,
            (path[i]["y"] - path[i - 1]["y"]) / dt,
        )
        for i in range(1, len(path))
    ]


def energy(delta_xy, m_diag):
    return m_diag[0] * delta_xy[0] * delta_xy[0] + m_diag[1] * delta_xy[1] * delta_xy[1]


def score_formulation(label_path, candidate_path, m_diag, alpha, dt, epsilon):
    count = max(len(label_path), len(candidate_path), 2)
    label = interpolate_path(label_path, count)
    candidate = interpolate_path(candidate_path, count)
    label_vel = velocities(label, dt)
    candidate_vel = velocities(candidate, dt)

    dyn_score = 0.0
    con_score = 0.0
    max_energy = 0.0
    for i, (v_label, v_candidate) in enumerate(zip(label_vel, candidate_vel), start=1):
        delta = (candidate[i]["x"] - label[i]["x"], candidate[i]["y"] - label[i]["y"])
        dyn_score = max(dyn_score, math.hypot(v_candidate[0] - v_label[0], v_candidate[1] - v_label[1]))
        e_hat = energy(delta, m_diag)
        max_energy = max(max_energy, e_hat)
        residual = (
            2.0 * (delta[0] * m_diag[0] * v_candidate[0] + delta[1] * m_diag[1] * v_candidate[1])
            - 2.0 * (delta[0] * m_diag[0] * v_label[0] + delta[1] * m_diag[1] * v_label[1])
            + 2.0 * alpha * e_hat
        )
        con_score = max(con_score, residual)

    alpha_bar = alpha - 0.5 * epsilon
    return {
        "dyn": round(dyn_score, 6),
        "con": round(max(0.0, con_score), 6),
        "max_energy": round(max_energy, 6),
        "alpha_bar": round(alpha_bar, 6),
    }


def conformal_quantile(values, delta):
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil((1.0 - delta) * (len(ordered) + 1)))
    return ordered[min(rank, len(ordered)) - 1]


def metric_is_acceptable(m_diag, alpha, epsilon, m_min, m_max):
    return all(m_min <= value <= m_max for value in m_diag) and alpha > 0.0 and 0.0 < epsilon < 2.0 * alpha


def sample_rrt_rows(count, seed):
    random.seed(seed)
    rows = []
    workspace = dict(DEFAULT_WORKSPACE)
    attempts = 0
    while len(rows) < count and attempts < count * 30:
        attempts += 1
        start, goal, obstacles = sample_environment(workspace, None, DEFAULT_CLEARANCE_M)
        try:
            label = plan_rrt(start, goal, obstacles, workspace=workspace, clearance_m=DEFAULT_CLEARANCE_M, seed=seed + attempts)
        except (RuntimeError, ValueError):
            continue
        rows.append({"start": start, "goal": goal, "workspace": workspace, "obstacles": obstacles, "rrt_label": label})
    return rows


def extract_json_object(text):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Llama response did not contain a JSON object.")
    return json.loads(text[start : end + 1])


def request_llama_waypoints(client, model_name, prompt, temperature, max_retries):
    last_error = None
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            content = response.choices[0].message.content
            payload = extract_json_object(content)
            waypoints = payload.get("waypoints")
            if not isinstance(waypoints, list) or len(waypoints) < 2:
                raise ValueError("Llama response JSON did not include at least two waypoints.")
            parsed = [
                {"x": float(point["x"]), "y": float(point["y"]), "z": float(point["z"])}
                for point in waypoints
                if all(key in point for key in ("x", "y", "z"))
            ]
            if len(parsed) < 2:
                raise ValueError("Llama response waypoints were missing x, y, or z fields.")
            return parsed
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Llama waypoint request failed after {max_retries} attempts: {last_error}")


def build_dataset(args):
    m_diag = [float(value) for value in args.m_diag.split(",")]
    if len(m_diag) != 2:
        raise ValueError("--m-diag must contain two comma-separated values, e.g. 1.0,1.0")
    acceptable_m = metric_is_acceptable(m_diag, args.alpha, args.epsilon, args.m_min, args.m_max)
    if not acceptable_m:
        raise ValueError("Contraction metric values are outside the accepted range or epsilon >= 2 * alpha.")

    rows = sample_rrt_rows(args.samples, args.seed)
    if len(rows) < args.samples:
        raise RuntimeError(f"Only built {len(rows)} rows; requested {args.samples}.")

    from openai import OpenAI

    client = OpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
    prompt_module = load_prompt_generator()
    scored = []
    for index, row in enumerate(rows[: args.samples]):
        prompt, _ = prompt_from_current_generator(
            prompt_module,
            row["start"],
            row["goal"],
            row["workspace"],
            row["obstacles"],
        )
        candidate = request_llama_waypoints(
            client,
            args.llama_model_name,
            prompt,
            args.temperature,
            args.llm_retries,
        )
        scores = score_formulation(row["rrt_label"], candidate, m_diag, args.alpha, args.dt, args.epsilon)
        scored.append((row, candidate, scores, prompt))
        print(f"Scored sample {index + 1}/{args.samples}: s_dyn={scores['dyn']}, s_con={scores['con']}")

    split = max(1, int(len(scored) * args.calibration_fraction))
    q_dyn = conformal_quantile([item[2]["dyn"] for item in scored[:split]], args.delta_dyn)
    q_con = conformal_quantile([item[2]["con"] for item in scored[:split]], args.delta_con)
    alpha_bar = args.alpha - 0.5 * args.epsilon
    bound_offset = ((q_dyn * q_dyn) * max(m_diag) / args.epsilon + q_con) / (2.0 * alpha_bar)

    fieldnames = [
        "sample_id",
        "start",
        "goal",
        "workspace",
        "obstacles",
        "rrt_waypoints",
        "llm_waypoints",
        "m_diag",
        "alpha",
        "epsilon",
        "alpha_bar",
        "s_dyn",
        "s_con",
        "max_energy",
        "q_dyn",
        "q_con",
        "bound_offset",
        "safety_buffer_m",
        "acceptable_M",
        "accepted",
        "llama_model_name",
        "vllm_base_url",
        "prompt",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, (row, candidate, scores, prompt) in enumerate(scored):
            writer.writerow(
                {
                    "sample_id": index,
                    "start": json.dumps(row["start"], separators=(",", ":")),
                    "goal": json.dumps(row["goal"], separators=(",", ":")),
                    "workspace": json.dumps(row["workspace"], separators=(",", ":")),
                    "obstacles": json.dumps(row["obstacles"], separators=(",", ":")),
                    "rrt_waypoints": json.dumps(row["rrt_label"], separators=(",", ":")),
                    "llm_waypoints": json.dumps(candidate, separators=(",", ":")),
                    "m_diag": json.dumps(m_diag, separators=(",", ":")),
                    "alpha": args.alpha,
                    "epsilon": args.epsilon,
                    "alpha_bar": scores["alpha_bar"],
                    "s_dyn": scores["dyn"],
                    "s_con": scores["con"],
                    "max_energy": scores["max_energy"],
                    "q_dyn": round(q_dyn, 6),
                    "q_con": round(q_con, 6),
                    "bound_offset": round(bound_offset, 6),
                    "safety_buffer_m": args.safety_buffer_m,
                    "acceptable_M": acceptable_m,
                    "accepted": scores["dyn"] <= q_dyn and scores["con"] <= q_con and bound_offset <= args.safety_buffer_m,
                    "llama_model_name": args.llama_model_name,
                    "vllm_base_url": args.vllm_base_url,
                    "prompt": prompt,
                }
            )
    return args.output


def main():
    parser = argparse.ArgumentParser(
        description="Build fresh prompt-compatible conformal calibration CSV data with RRT labels and Llama predictions."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--m-diag", default="1.0,1.0", help="Diagonal contraction metric values for M.")
    parser.add_argument("--m-min", type=float, default=0.25)
    parser.add_argument("--m-max", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--epsilon", type=float, default=0.4)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--delta-dyn", type=float, default=0.1)
    parser.add_argument("--delta-con", type=float, default=0.1)
    parser.add_argument("--calibration-fraction", type=float, default=0.8)
    parser.add_argument("--safety-buffer-m", type=float, default=DEFAULT_CLEARANCE_M)
    parser.add_argument("--llama-model-name", default=DEFAULT_LLAMA_MODEL_NAME)
    parser.add_argument("--vllm-base-url", default=DEFAULT_VLLM_BASE_URL)
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm-retries", type=int, default=2)
    args = parser.parse_args()

    output = build_dataset(args)
    print(f"Wrote prompt-compatible RRT conformal calibration dataset to {output}")


if __name__ == "__main__":
    main()
