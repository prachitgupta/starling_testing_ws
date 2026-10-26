#!/usr/bin/env python3
"""Create presentation-ready plots from a transformer verification run."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


COLORS = {
    "train": "#2563eb",
    "validation": "#dc2626",
    "expert": "#16a34a",
    "prediction": "#f97316",
    "accent": "#7c3aed",
    "muted": "#64748b",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite_series(values):
    return [float(value) if math.isfinite(float(value)) else np.nan for value in values]


def plot_training(run_root: Path, output_dir: Path) -> Path:
    history = read_json(run_root / "model" / "training_history.json")
    summary = read_json(run_root / "model" / "training_summary.json")
    epochs = [int(row["epoch"]) for row in history]
    train = [float(row["train_euclidean_imitation_loss_m"]) for row in history]
    validation = [float(row["euclidean_imitation_loss_m"]) for row in history]
    pass_rate = [float(row["verification_pass_rate"]) for row in history]
    radius = finite_series(row["safety_radius_m"] for row in history)

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    axes[0].plot(epochs, train, color=COLORS["train"], linewidth=2.2, label="Training imitation loss")
    axes[0].plot(epochs, validation, color=COLORS["validation"], linewidth=2.2, label="Validation imitation loss")
    axes[0].set(title="Waypoint imitation loss", xlabel="Epoch", ylabel="Mean Euclidean error (m)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, pass_rate, color=COLORS["accent"], linewidth=2.2, label="Verification pass rate")
    axes[1].set(xlabel="Epoch", ylabel="Verification pass rate", ylim=(0.0, 1.05))
    axes[1].grid(alpha=0.25)
    radius_axis = axes[1].twinx()
    radius_axis.plot(epochs, radius, color=COLORS["muted"], linewidth=2.0, label="Finite safety radius")
    threshold = float(summary["safety_radius_threshold_m"])
    radius_axis.axhline(threshold, color="#111827", linestyle="--", linewidth=1.5, label=f"Radius target ({threshold:.2f} m)")
    radius_axis.set_ylabel("Safety radius (m)")
    lines = axes[1].get_lines() + radius_axis.get_lines()
    axes[1].legend(lines, [line.get_label() for line in lines], loc="best")
    output = output_dir / "training_curves.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_diagnostics(run_root: Path, output_dir: Path) -> Path:
    preflight = read_json(run_root / "reports" / "preflight.json")
    postflight = read_json(run_root / "reports" / "postflight.json")
    tiny = preflight["tiny_overfit"]
    gradients = postflight["gradient_norms"]
    sensitivity = postflight["sensitivity"]
    validation, test = postflight["validation"], postflight["test"]

    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].bar(
        ["Initial", "Final"],
        [tiny["initial_loss_m"], tiny["final_loss_m"]],
        color=[COLORS["muted"], COLORS["train"]],
    )
    axes[0, 0].set(title="Tiny-set overfit test", ylabel="Euclidean loss (m)")
    axes[0, 0].grid(axis="y", alpha=0.25)

    names = list(gradients)
    axes[0, 1].barh(names, [gradients[name] for name in names], color=COLORS["accent"])
    axes[0, 1].set_xscale("log")
    axes[0, 1].set(title="Gradient flow by component", xlabel="Gradient L2 norm (log scale)")
    axes[0, 1].grid(axis="x", alpha=0.25)

    axes[1, 0].bar(
        ["Text changed", "Scene changed"],
        [sensitivity["text_change_normalized_waypoint_delta"], sensitivity["scene_change_normalized_waypoint_delta"]],
        color=[COLORS["train"], COLORS["expert"]],
    )
    axes[1, 0].set(title="Input sensitivity", ylabel="Mean normalized waypoint change")
    axes[1, 0].grid(axis="y", alpha=0.25)

    labels = ["Mean", "Median", "P95", "Maximum"]
    keys = ["mean_euclidean_imitation_loss_m", "median_waypoint_error_m", "p95_waypoint_error_m", "maximum_waypoint_error_m"]
    positions = np.arange(len(labels))
    width = 0.36
    axes[1, 1].bar(positions - width / 2, [validation[key] for key in keys], width, label="Validation", color=COLORS["validation"])
    axes[1, 1].bar(positions + width / 2, [test[key] for key in keys], width, label="Test", color=COLORS["train"])
    axes[1, 1].set(title="Waypoint error distribution", ylabel="Error (m)", xticks=positions, xticklabels=labels)
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.25)

    output = output_dir / "component_diagnostics.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def plot_routes(run_root: Path, output_dir: Path) -> Path:
    report = read_json(run_root / "reports" / "postflight.json")
    examples = report.get("route_examples", [])
    if not examples:
        raise ValueError("postflight report contains no route_examples")
    columns = 3
    rows = math.ceil(len(examples) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(15, 4.5 * rows), squeeze=False, constrained_layout=True)
    for axis, example in zip(axes.flat, examples):
        environment = example["environment"]
        for obstacle in environment.get("obstacles", []):
            minimum, maximum = obstacle["min_corner"], obstacle["max_corner"]
            axis.add_patch(
                plt.Rectangle(
                    (minimum[0], minimum[1]),
                    maximum[0] - minimum[0],
                    maximum[1] - minimum[1],
                    facecolor="#94a3b8",
                    edgecolor="#334155",
                    alpha=0.65,
                )
            )
        expert = np.asarray(example["expert_xy"], dtype=float)
        predicted = np.asarray(example["predicted_xy"], dtype=float)
        axis.plot(expert[:, 0], expert[:, 1], "o-", color=COLORS["expert"], linewidth=2.2, label="HRRT target")
        axis.plot(predicted[:, 0], predicted[:, 1], "o--", color=COLORS["prediction"], linewidth=2.2, label="Transformer")
        workspace = environment["workspace"]
        axis.set(xlim=workspace["x"], ylim=workspace["y"], aspect="equal", title=f"{example['sample_id']}\n{example['preference'][:55]}")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(examples) :]:
        axis.set_visible(False)
    axes.flat[0].legend(loc="best")
    output = output_dir / "route_examples.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def write_dashboard(run_root: Path, output_dir: Path, images: list[Path]) -> Path:
    postflight = read_json(run_root / "reports" / "postflight.json")
    checkpoint = html.escape(postflight["selected_checkpoint"])
    image_tags = "".join(
        f"<section><h2>{html.escape(image.stem.replace('_', ' ').title())}</h2><img src='{html.escape(image.name)}'></section>"
        for image in images
    )
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>Transformer verification</title>
<style>body{{font-family:Inter,system-ui,sans-serif;background:#f8fafc;color:#0f172a;max-width:1500px;margin:auto;padding:28px}}section{{background:white;border:1px solid #cbd5e1;border-radius:14px;padding:18px;margin:18px 0;box-shadow:0 8px 28px #0f172a12}}img{{display:block;width:100%;height:auto}}code{{background:#e2e8f0;padding:3px 6px;border-radius:5px}}</style></head><body>
<h1>HRRT waypoint-transformer verification</h1><p>Selected checkpoint: <code>{checkpoint}</code></p>{image_tags}</body></html>"""
    output = output_dir / "index.html"
    output.write_text(document, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output_dir = (args.output_dir or run_root / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images = [
        plot_training(run_root, output_dir),
        plot_diagnostics(run_root, output_dir),
        plot_routes(run_root, output_dir),
    ]
    dashboard = write_dashboard(run_root, output_dir, images)
    print(json.dumps({"plots": [str(path) for path in images], "dashboard": str(dashboard)}, indent=2))


if __name__ == "__main__":
    main()
