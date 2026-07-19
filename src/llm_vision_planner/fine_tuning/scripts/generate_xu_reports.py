#!/usr/bin/env python3
"""Create the numerical four-method report and the WN&CNet review."""

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np

from conformal_rrt_dataset import interpolate_sample
from lqr import B_DOUBLE_INTEGRATOR, solve_care


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "fine_tuning" / "datasets"
REPORT = ROOT.parent / "papers" / "report"
METHODS = [
    ("Different clocks", "calibration_min_snap_different_clocks.csv"),
    ("Shared clock", "calibration_min_snap_shared_clock.csv"),
    ("Frenet", "calibration_min_snap_frenet.csv"),
    ("Guided shared", "calibration_min_snap_guided_fit_shared.csv"),
]


def load():
    result = {}
    for label, filename in METHODS:
        with (DATA / filename).open(newline="", encoding="utf-8") as stream:
            result[label] = list(csv.DictReader(stream))
    return result


def choose_sample(rows):
    best_id, best_score = None, -1.0
    for index in range(len(next(iter(rows.values())))):
        score = 0.0
        for method_rows in rows.values():
            row = method_rows[index]
            score += math.hypot(float(row["s_u"]) / max(float(row["q_u"]), 1e-12), float(row["s_x"]) / max(float(row["q_x"]), 1e-12))
        if score > best_score:
            best_id, best_score = index, score
    return best_id


def arc(samples):
    points = np.asarray([sample["x"][:2] for sample in samples], dtype=float)
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))


def errors(rrt, llm, step=0.05):
    horizon = min(float(rrt[-1]["t"]), float(llm[-1]["t"]))
    times = np.arange(0.0, horizon + step / 2.0, step)
    sx, su = [], []
    for time in times:
        rx, ru = interpolate_sample(rrt, float(time))
        lx, lu = interpolate_sample(llm, float(time))
        sx.append(float(np.linalg.norm(np.asarray(lx) - np.asarray(rx))))
        su.append(float(np.linalg.norm(np.asarray(lu) - np.asarray(ru))))
    return times, sx, su


def plots(rows, sample_id, prefix="xu_method"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig2, error_axes = plt.subplots(2, 2, figsize=(10, 8))
    fig3, timing = plt.subplots(figsize=(8, 5))
    for axis, error_axis, (label, _) in zip(axes.flat, error_axes.flat, METHODS):
        row = rows[label][sample_id]
        rrt, llm = json.loads(row["rrt_trajectory"])["samples"], json.loads(row["llm_trajectory"])["samples"]
        axis.plot([v["x"][0] for v in rrt], [v["x"][1] for v in rrt], label="RRT expert")
        axis.plot([v["x"][0] for v in llm], [v["x"][1] for v in llm], "--", label="LLM")
        axis.set_title(label); axis.set_aspect("equal", adjustable="box"); axis.grid(alpha=.25); axis.legend()
        times, sx, su = errors(rrt, llm)
        error_axis.plot(times, sx, label=r"$\|\Delta x\|$")
        error_axis.plot(times, su, label=r"$\|\Delta u\|$")
        error_axis.set_title(label); error_axis.set_xlabel("time [s]"); error_axis.grid(alpha=.25); error_axis.legend()
        timing.plot([float(v["t"]) for v in rrt], arc(rrt), label=f"{label}: RRT")
        timing.plot([float(v["t"]) for v in llm], arc(llm), "--", label=f"{label}: LLM")
    axes[1, 0].set_xlabel("x [m]"); axes[1, 1].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]"); axes[1, 0].set_ylabel("y [m]")
    fig.tight_layout(); fig2.tight_layout()
    timing.set_xlabel("time [s]"); timing.set_ylabel("cumulative path length [m]"); timing.grid(alpha=.25); timing.legend(fontsize=7, ncol=2); fig3.tight_layout()
    paths = [REPORT / f"{prefix}_paths.png", REPORT / f"{prefix}_errors.png", REPORT / f"{prefix}_progress.png"]
    for figure, path in zip((fig, fig2, fig3), paths):
        figure.savefig(path, dpi=180); plt.close(figure)


def tube_factor(q_u, q_x):
    p, k, alpha = solve_care()
    eigenvalues = np.linalg.eigvalsh(p)
    return float(np.sqrt(np.max(eigenvalues)) * (np.linalg.norm(B_DOUBLE_INTEGRATOR, 2) * q_u + np.linalg.norm(B_DOUBLE_INTEGRATOR @ k, 2) * q_x) / (alpha * np.sqrt(np.min(eigenvalues))))


def comparison_tex(rows, sample_id, plot_prefix="xu_method", selection_text=None):
    base = rows["Different clocks"][sample_id]
    rrt_wp = json.loads(base["rrt_verified_waypoints"])
    llm_wp = json.loads(base["llm_verified_waypoints"])
    table = []
    for label, _ in METHODS:
        row = rows[label][sample_id]
        qu, qx = float(row["q_u"]), float(row["q_x"])
        table.append(f"{label} & {float(row['s_u']):.4f} & {float(row['s_x']):.4f} & {qu:.4f} & {qx:.4f} & {tube_factor(qu, qx):.4f}" + r" \\")
    waypoint_text = lambda values: ", ".join(f"({float(v['x']):.2f},{float(v['y']):.2f})" for v in values)
    selection_text = selection_text or "The deterministic difficult case maximizes the sum, across methods, of $\\sqrt{{(s_u/q_u)^2+(s_x/q_x)^2}}$."
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}\usepackage{{amsmath,booktabs,graphicx,url}}
\title{{Waypoint-to-State/Control Scheduling Comparison}}\author{{}}\date{{}}
\begin{{document}}\maketitle
\section*{{Methods}}
All methods use the same 200 actual LLM predictions and verified RRT references. Independent minimum snap compares each path on its natural clock. Shared clock assigns the expert horizon to the LLM path. Frenet scoring retains minimum-snap references but projects ordered LLM samples onto expert arc length before comparing the full state and control. Guided fit replaces seventh-order minimum snap with linear motion and symmetric constant-acceleration parabolic blends (blend fraction 0.2), then applies the shared clock. The blend construction follows \emph{{Computationally Efficient Algorithm to Generate a Waypoints-Based Trajectory for a Quadrotor UAV}}; the state/control mapping is consistent with the differential-flatness discussion in \emph{{A Comparative Study of Nonlinear MPC and Differential-Flatness-Based Control for Quadrotor Agile Flight}}.
\section*{{Selected actual prediction}}
{selection_text} It is sample {base['sample_id']}.

RRT verified waypoints: {waypoint_text(rrt_wp)}.\\
LLM verified waypoints: {waypoint_text(llm_wp)}.
\begin{{center}}\begin{{tabular}}{{lrrrrr}}\toprule
Method & $s_u$ & $s_x$ & $q_u$ & $q_x$ & asymptotic radius term [m]\\\midrule
{chr(10).join(table)}
\bottomrule\end{{tabular}}\end{{center}}
The last column substitutes each dataset's quantiles into the existing contraction-tube disturbance term. It isolates the quantile-dependent contribution; it is not a measured flight tracking radius. Independent clocks compare different phases and can inflate derivative-derived control error. Shared scheduling aligns the total execution horizon. Frenet alignment removes wall-clock phase from matching. Guided fit limits high-order polynomial oscillation but introduces explicit acceleration changes at its blends.
\begin{{figure}}[ht]\centering\includegraphics[width=.92\linewidth]{{{plot_prefix}_paths.png}}\caption{{Expert and actual predicted paths.}}\end{{figure}}
\begin{{figure}}[ht]\centering\includegraphics[width=.92\linewidth]{{{plot_prefix}_progress.png}}\caption{{Wall-clock schedule versus spatial progress.}}\end{{figure}}
\begin{{figure}}[ht]\centering\includegraphics[width=.92\linewidth]{{{plot_prefix}_errors.png}}\caption{{Time-indexed diagnostic state/control errors. Frenet calibration itself uses progress-aligned errors.}}\end{{figure}}
\clearpage\section*{{Interpretation}}
The numerical comparison separates three causes of large conformal radii: clock phase, spatial disagreement, and trajectory differentiation. Shared scheduling addresses phase at the horizon level; Frenet projection removes phase during scoring; guided parabolic blends avoid seventh-order differentiation. None changes the verified waypoint predictions or the contraction controller.
\end{{document}}
"""


def nn_tex():
    return r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}\usepackage{amsmath,url}
\title{Neural Waypoint-to-Control Generation: WN\&CNets}\author{}\date{}
\begin{document}\maketitle
\section*{What the method is}
The local paper \emph{Imitation Learning-Based Online Time-Optimal Control with Multiple-Waypoint Constraints for Quadrotors} trains waypoint-constrained navigation and control networks (WN\&CNets) on state--control pairs produced offline by complementary-progress-constraint (CPC) time-optimal optimization. Its input contains relative positions to two waypoints, velocity, roll--pitch--yaw attitude, and thrust. Its output is the next angular-rate command and total-thrust command. Delay-aware training maps a delayed observed state to the future command.
\section*{Why it can reduce amplification}
The deployed network directly predicts control at each update. It therefore does not obtain feedforward control by repeatedly differentiating a fitted seventh-order position polynomial. This can remove polynomial conditioning, high derivative peaks, segment-duration sensitivity, and wall-clock phase amplification from the waypoint-to-control conversion. Smoothness is learned from CPC demonstrations rather than guaranteed analytically, so prediction errors, distribution shift, delay mismatch, and control discontinuities can still increase $s_u$ or $s_x$.
\section*{Compatibility and limitations}
The paper uses full quadrotor state and thrust/angular-rate commands, whereas the current repository calibrates a planar damped double integrator with $x=[p_x,p_y,v_x,v_y]$ and $u=[u_x,u_y]$. A compatible adoption therefore requires a separately generated expert dataset, an explicitly defined network mapping for the repository's state/control interface, trained weights, normalization metadata, and validation under the same refinement, verification, and conformal scoring pipeline. The paper also uses MINCO during multi-waypoint transitions, so it does not eliminate polynomial trajectories from every part of its full system.

No neural model, training script, or calibration data is implemented as part of this comparison.
\end{document}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", type=int, default=None)
    parser.add_argument("--output-prefix", default="xu_method_comparison")
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    rows = load()
    sample_id = choose_sample(rows) if args.sample_id is None else args.sample_id
    if sample_id < 0 or sample_id >= len(next(iter(rows.values()))):
        raise ValueError("--sample-id is outside the dataset")
    plot_prefix = "xu_method" if args.output_prefix == "xu_method_comparison" else args.output_prefix
    plots(rows, sample_id, plot_prefix)
    selection_text = None
    if args.sample_id is not None:
        selection_text = "This case was selected as a moderate example with nonzero $s_u$ and $s_x$ under all four methods, rather than as an extreme-error case."
    (REPORT / f"{args.output_prefix}.tex").write_text(
        comparison_tex(rows, sample_id, plot_prefix=plot_prefix, selection_text=selection_text), encoding="utf-8"
    )
    if args.output_prefix == "xu_method_comparison":
        (REPORT / "nn_waypoint_control_review.tex").write_text(nn_tex(), encoding="utf-8")
    print(f"Selected sample_id={rows['Different clocks'][sample_id]['sample_id']}")


if __name__ == "__main__":
    main()
