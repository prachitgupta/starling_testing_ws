#!/usr/bin/env python3
"""Generate an actual-data report about minimum-snap time scaling."""

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt
import numpy as np

from conformal_rrt_dataset import score_trajectories, shared_llm_durations
from min_snap import DAMPING, derivative_row, generate_trajectory, segment_durations, solve_axis_min_snap


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "fine_tuning" / "datasets" / "calibration_min_snap_shared_clock.csv"
REPORT_DIR = ROOT.parent / "papers" / "report"
SAMPLE_ID = 105
SCALES = [1.0, 0.75, 0.5, 0.25]


def load_waypoints():
    with DATASET.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    row = rows[SAMPLE_ID]
    return json.loads(row["rrt_verified_waypoints"]), json.loads(row["llm_verified_waypoints"])


def exact_metrics(waypoints, durations):
    cx = solve_axis_min_snap([float(point["x"]) for point in waypoints], durations)
    cy = solve_axis_min_snap([float(point["y"]) for point in waypoints], durations)
    maxima = {name: 0.0 for name in ("position", "velocity", "acceleration", "jerk", "snap", "control")}
    for segment, duration in enumerate(durations):
        for time in np.linspace(0.0, duration, 301):
            values = []
            for derivative in range(5):
                values.append(np.array([derivative_row(time, derivative).dot(cx[segment]), derivative_row(time, derivative).dot(cy[segment])]))
            maxima["position"] = max(maxima["position"], float(np.linalg.norm(values[0])))
            maxima["velocity"] = max(maxima["velocity"], float(np.linalg.norm(values[1])))
            maxima["acceleration"] = max(maxima["acceleration"], float(np.linalg.norm(values[2])))
            maxima["jerk"] = max(maxima["jerk"], float(np.linalg.norm(values[3])))
            maxima["snap"] = max(maxima["snap"], float(np.linalg.norm(values[4])))
            maxima["control"] = max(maxima["control"], float(np.linalg.norm(values[1] + values[2] / DAMPING)))
    coefficient_maxima = {
        power: float(max(np.max(np.abs(cx[:, power])), np.max(np.abs(cy[:, power]))))
        for power in range(4, 8)
    }
    return maxima, coefficient_maxima


def build_cases(rrt_waypoints, llm_waypoints):
    base_rrt = segment_durations(rrt_waypoints)
    base_llm = shared_llm_durations(base_rrt, llm_waypoints)
    cases = []
    for scale in SCALES:
        rrt_durations = [duration * scale for duration in base_rrt]
        llm_durations = [duration * scale for duration in base_llm]
        rrt = generate_trajectory(rrt_waypoints, dt=0.01, durations=rrt_durations)
        llm = generate_trajectory(llm_waypoints, dt=0.01, durations=llm_durations)
        scores = score_trajectories(rrt, llm, dt=0.01)
        maxima, coefficients = exact_metrics(llm_waypoints, llm_durations)
        cases.append(
            {
                "scale": scale,
                "rrt": rrt,
                "llm": llm,
                "scores": scores,
                "maxima": maxima,
                "coefficients": coefficients,
                "horizon": sum(llm_durations),
            }
        )
    return cases


def make_plots(cases):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for case in cases:
        samples = case["llm"]["samples"]
        axes[0, 0].plot([sample["x"][0] for sample in samples], [sample["x"][1] for sample in samples], label=f"scale={case['scale']}")
    axes[0, 0].set_title("Same spatial path under uniform re-timing")
    axes[0, 0].set_xlabel("x [m]"); axes[0, 0].set_ylabel("y [m]"); axes[0, 0].axis("equal"); axes[0, 0].grid(alpha=.25); axes[0, 0].legend()

    scales = np.asarray([case["scale"] for case in cases])
    axes[0, 1].loglog(scales, [case["scores"]["s_x"] for case in cases], "o-", label=r"$s_x$")
    axes[0, 1].loglog(scales, [case["scores"]["s_u"] for case in cases], "s-", label=r"$s_u$")
    axes[0, 1].invert_xaxis(); axes[0, 1].set_title("Actual RRT--LLM score amplification"); axes[0, 1].set_xlabel("duration scale"); axes[0, 1].grid(True, which="both", alpha=.25); axes[0, 1].legend()

    for name in ("velocity", "acceleration", "jerk", "snap", "control"):
        axes[1, 0].loglog(scales, [case["maxima"][name] for case in cases], "o-", label=name)
    axes[1, 0].invert_xaxis(); axes[1, 0].set_title("Maximum derivative and control magnitudes"); axes[1, 0].set_xlabel("duration scale"); axes[1, 0].grid(True, which="both", alpha=.25); axes[1, 0].legend(fontsize=8)

    for power in range(4, 8):
        axes[1, 1].loglog(scales, [case["coefficients"][power] for case in cases], "o-", label=rf"$|c_{power}|$")
    axes[1, 1].invert_xaxis(); axes[1, 1].set_title("Physical-time high-order coefficients"); axes[1, 1].set_xlabel("duration scale"); axes[1, 1].grid(True, which="both", alpha=.25); axes[1, 1].legend()
    fig.tight_layout(); fig.savefig(REPORT_DIR / "polynomial_time_scaling.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    for case in cases:
        samples = case["llm"]["samples"]
        horizon = float(samples[-1]["t"])
        axis.plot([float(sample["t"]) / horizon for sample in samples], [np.linalg.norm(sample["u"]) for sample in samples], label=f"scale={case['scale']}")
    axis.set_xlabel("normalized trajectory progress $t/T$"); axis.set_ylabel(r"$\|u_d\|$"); axis.set_title("Control amplification for the same waypoint geometry"); axis.grid(alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(REPORT_DIR / "polynomial_control_scaling.png", dpi=180); plt.close(fig)


def latex(rrt_waypoints, llm_waypoints, cases):
    distances = [np.linalg.norm(np.array([b["x"], b["y"]]) - np.array([a["x"], a["y"]])) for a, b in zip(llm_waypoints, llm_waypoints[1:])]
    rows = []
    for case in cases:
        m, s = case["maxima"], case["scores"]
        rows.append(
            f"{case['scale']:.2f} & {case['horizon']:.3f} & {s['s_x']:.3f} & {s['s_u']:.3f} & {m['velocity']:.3f} & {m['acceleration']:.3f} & {m['snap']:.3f} & {m['control']:.3f} " + r"\\"
        )
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}\usepackage{{amsmath,booktabs,graphicx,url}}
\title{{Why Minimum-Snap $x_d$ and $u_d$ Can Grow for Nearby Waypoints}}\author{{}}\date{{}}
\begin{{document}}\maketitle
\section*{{Actual-data experiment}}
This report uses actual verified RRT and LLM waypoints from calibration sample {SAMPLE_ID}; it does not synthesize a new prediction. The six LLM segment lengths range from {min(distances):.3f} m to {max(distances):.3f} m. The experiment keeps those waypoint locations fixed and uniformly multiplies all segment durations by $\rho\in\{{1,0.75,0.5,0.25\}}$. Thus any increase in state derivatives or control is caused by time parameterization, not greater waypoint distance.

\begin{{center}}\small\begin{{tabular}}{{rrrrrrrr}}\toprule
$\rho$ & horizon [s] & $s_x$ & $s_u$ & max $\|v\|$ & max $\|a\|$ & max $\|p^{{(4)}}\|$ & max $\|u\|$\\\midrule
{chr(10).join(rows)}
\bottomrule\end{{tabular}}\end{{center}}

\section*{{Time normalization and coefficient growth}}
Write one seventh-degree segment using normalized time $\sigma=t/T$:
\[
p(t)=\sum_{{k=0}}^7 a_k\sigma^k
=\sum_{{k=0}}^7 \underbrace{{\frac{{a_k}}{{T^k}}}}_{{c_k}}t^k.
\]
The physical-time coefficient is therefore $c_k=a_k/T^k$. In particular, the fourth through seventh coefficients contain $T^{{-4}},T^{{-5}},T^{{-6}},T^{{-7}}$. This is coefficient growth under a change of coordinates; it can also worsen numerical conditioning when $T$ is small.

Every derivative adds another inverse power of duration:
\[
\frac{{d^r p}}{{dt^r}}=T^{{-r}}\frac{{d^r p}}{{d\sigma^r}}.
\]
Velocity scales as $T^{{-1}}$, acceleration as $T^{{-2}}$, jerk as $T^{{-3}}$, and snap as $T^{{-4}}$. The minimum-snap objective itself scales as
\[
\int_0^T \|p^{{(4)}}(t)\|^2dt
=T^{{-7}}\int_0^1\left\|\frac{{d^4p}}{{d\sigma^4}}\right\|^2d\sigma.
\]
Minimum snap minimizes this large quantity subject to waypoint and continuity constraints; it cannot make the required derivatives small when the assigned time is too short.

\section*{{How oscillation is amplified}}
A small between-waypoint ripple $e(t)=A\sin(\omega t)$ has derivatives $A\omega^r$. Its position amplitude remains $A$, but its acceleration is $A\omega^2$ and snap is $A\omega^4$. Shorter segment time raises the effective frequency. High-order coupled fits can also overshoot between constraints even when adjacent waypoints are close; differentiation makes such visually small lobes significant in $v$, $a$, and snap. The reviewed linear/parabolic-blend paper explicitly motivates its lower-order construction by the computational burden and Runge-type behavior of higher-order interpolation.

For this repository,
\[
x_d=[p_x,p_y,v_x,v_y]^\top,\qquad
u_d=v_d+\frac{{a_d}}{{1.1}}.
\]
Consequently, position does not acquire a direct $T^{{-4}}$ factor. The velocity part of $x_d$ scales as $T^{{-1}}$, while $u_d$ contains $T^{{-1}}$ and $T^{{-2}}$ terms and is normally acceleration-dominated for short durations. The $T^{{-4}}$ factor belongs to snap and to fourth-order differentiation, not directly to this planar control definition. For a full differentially flat quadrotor, higher derivatives can enter attitude, angular-rate, and feedforward mappings, so amplification can propagate further; this relationship is discussed in the reviewed DFBC/NMPC paper.

\begin{{figure}}[ht]\centering\includegraphics[width=.96\linewidth]{{polynomial_time_scaling.png}}\caption{{Actual sample {SAMPLE_ID}: identical waypoint geometry, score growth, derivative growth, and physical-time coefficient growth under re-timing.}}\end{{figure}}
\begin{{figure}}[ht]\centering\includegraphics[width=.94\linewidth]{{polynomial_control_scaling.png}}\caption{{The same LLM waypoint path produces increasingly large control peaks as its duration is shortened.}}\end{{figure}}

\section*{{Implication for conformal radius}}
The calibration scores take maxima over time. A brief derivative peak therefore becomes the sample's $s_u$ or $s_x$ and can enter a high conformal quantile. The contraction radius weights $q_x$ and $q_u$, so a small number of short-time or oscillatory polynomial fits can enlarge the radius for every subsequent trajectory. Shared scheduling removes one source of time mismatch, Frenet matching removes wall-clock phase during comparison, and guided parabolic blends avoid high-order polynomial differentiation; none substitutes for enforcing feasible segment times and explicit derivative/input limits.
\end{{document}}
"""


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rrt_waypoints, llm_waypoints = load_waypoints()
    cases = build_cases(rrt_waypoints, llm_waypoints)
    make_plots(cases)
    output = REPORT_DIR / "polynomial_time_scaling_report.tex"
    output.write_text(latex(rrt_waypoints, llm_waypoints, cases), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
