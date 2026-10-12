#!/usr/bin/env python3
"""Interactive, offline calibration verifier for one refined capture.

This gateway deliberately does not subscribe to ROS topics.  It reuses the
residual-calibration CSV schema and the same HRRT/minimum-control machinery as
the batch postprocessor, while exposing a small browser UI for L1/L2, route
selection, and trajectory/tube inspection.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.parse import parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hrrt_star import normalize_obstacles, plan_hrrt_star  # noqa: E402
from min_control_qp import generate_shared_pair  # noqa: E402
from postprocess_residual_calibration import (  # noqa: E402
    environments,
    residual_score,
    safe_path,
    vllm_predicted_waypoints,
)


def compact(value):
    return json.dumps(value, separators=(",", ":"))


def load_captures(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["capture_id"], []).append(row)
    if not grouped:
        raise ValueError(f"no captures found in {path}")
    return [(capture_id, grouped[capture_id]) for capture_id in sorted(grouped)]


def choose_capture(captures, row_number: int):
    if row_number < 1 or row_number > len(captures):
        raise ValueError(f"--row must be between 1 and {len(captures)}")
    return captures[row_number - 1]


def scene_setup(rows, args):
    ground_truth, perceived = environments(rows)
    first = rows[0]
    start = {
        "x": float(first.get("observer_x") or args.start_x),
        "y": float(first.get("observer_y") or args.start_y),
        "z": float(first.get("observer_z") or args.fixed_z),
    }
    goal = {"x": args.goal_x, "y": args.goal_y, "z": args.fixed_z}
    workspace = {
        "x": [args.workspace_x_min, args.workspace_x_max],
        "y": [args.workspace_y_min, args.workspace_y_max],
        "z": args.fixed_z,
    }
    return ground_truth, perceived, start, goal, workspace


def route_cards(routes):
    return [
        {
            "route_id": str(route["route_id"]),
            "path_length_m": round(float(route["path_length_m"]), 3),
            "minimum_clearance_m": {
                str(key): round(float(value), 3)
                for key, value in route["minimum_clearance_m"].items()
            },
            "waypoints": route["waypoints"],
        }
        for route in routes
    ]


def plan_family(start, goal, obstacles, workspace, args):
    return plan_hrrt_star(
        start,
        goal,
        obstacles,
        workspace,
        clearance_m=args.clearance_m,
        max_iterations=args.hrrt_iterations,
        max_candidates=args.max_candidates,
        max_waypoints=args.max_waypoints,
        seed=args.seed,
    )


def planner_waypoints(start, goal, planning_obstacles, workspace, l1, args):
    if args.planner_provider == "mock":
        family = plan_family(start, goal, planning_obstacles, workspace, args)
        return min(family["routes"], key=lambda route: (route["path_length_m"], route["route_id"]))["waypoints"]
    planner_args = SimpleNamespace(
        vllm_base_url=args.vllm_base_url,
        vllm_api_key=args.vllm_api_key,
        llama_model=args.llama_model,
        clearance_m=args.clearance_m,
    )
    return vllm_predicted_waypoints(
        start, goal, planning_obstacles, workspace, l1, planner_args
    )


def score_pair(expert_waypoints, predicted_waypoints, args):
    expert_trajectory, predicted_trajectory = generate_shared_pair(
        expert_waypoints,
        predicted_waypoints,
        workspace={
            "x": [args.workspace_x_min, args.workspace_x_max],
            "y": [args.workspace_y_min, args.workspace_y_max],
            "z": args.fixed_z,
        },
        dt=args.dt,
        damping=0.0,
    )
    maximum, p_matrix, k_matrix = residual_score(
        expert_trajectory["samples"], predicted_trajectory["samples"]
    )
    score = float(maximum[0])
    return {
        "score": score,
        "max_time_s": float(maximum[1]),
        "residual": [float(value) for value in maximum[2]],
        "expert_samples": expert_trajectory["samples"],
        "predicted_samples": predicted_trajectory["samples"],
        "p_matrix": p_matrix.tolist(),
        "k_matrix": k_matrix.tolist(),
    }


def public_result(capture_id, rows, mode, l1, l2, expert, predicted, result, environments_used):
    return {
        "capture_id": capture_id,
        "session_id": rows[0].get("session_id", ""),
        "mode": mode,
        "l1": l1,
        "l2": l2,
        "expert_waypoints": expert["waypoints"],
        "expert_route_id": expert["route_id"],
        "predicted_waypoints": predicted,
        "score": result["score"],
        "score_max_time_s": result["max_time_s"],
        "residual": result["residual"],
        "tube_quantile": result["score"],
        "expert_samples": result["expert_samples"],
        "predicted_samples": result["predicted_samples"],
        "ground_truth_obstacles": environments_used[0],
        "planning_obstacles": environments_used[1],
    }


STYLE = """
body{font-family:system-ui,sans-serif;background:#07111d;color:#e7f4ff;margin:0;padding:20px}
main{max-width:1400px;margin:auto}.grid{display:grid;grid-template-columns:360px 1fr;gap:16px}
.card{background:#102437;border:1px solid #29445b;border-radius:14px;padding:16px;margin-bottom:14px}
label{display:block;color:#a9c3d6;font-size:12px;margin:10px 0 4px}input,select,textarea,button{width:100%;box-sizing:border-box;padding:9px;border-radius:8px;border:1px solid #41617a;background:#081725;color:#eef8ff}textarea{min-height:62px}button{background:#2788c7;font-weight:700;cursor:pointer;margin-top:12px}.secondary{background:#21445b}
pre{white-space:pre-wrap;overflow:auto}.metric{font-size:19px;color:#59e0aa;font-weight:800}.hint{color:#a9c3d6;font-size:12px;line-height:1.4}
svg{width:100%;height:auto;background:#eff7fb;border-radius:10px}.legend{font-size:12px;color:#b9d0e0}
@media(max-width:850px){.grid{grid-template-columns:1fr}body{padding:8px}}
"""


def html_page(state, args, message=""):
    capture_options = "".join(
        f'<option value="{i}">row {i}: capture {html.escape(cid)}</option>'
        for i, (cid, _) in enumerate(state["captures"], 1)
    )
    selected = state["row_number"]
    routes = state.get("routes", [])
    route_options = "".join(
        f'<option value="{html.escape(route["route_id"])}">{html.escape(route["route_id"])} — '
        f'{route["path_length_m"]:.2f} m</option>' for route in routes
    )
    result = state.get("result")
    results = state.get("results", {})
    result_json = compact(result) if result else ""
    plot = plot_svg(result) if result else "<p class='hint'>Generate HRRT routes, then select an expert route to score.</p>"
    score_block = (
        f"<div class='metric'>{result['score']:.5f}</div>"
        f"<div class='hint'>Tube quantile q = score = {result['score']:.5f} m; max residual time = {result['score_max_time_s']:.2f} s</div>"
        if result else ""
    )
    if results:
        score_block += "<h3>Both uncertainty modes</h3>" + "".join(
            f"<p><b>{html.escape(name)}</b>: <span class='metric'>{item['score']:.5f} m</span> "
            f"(q = {item['tube_quantile']:.5f} m)</p>"
            for name, item in results.items()
        )
    return f"""<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'><title>Calibration verifier</title><style>{STYLE}</style></head><body><main>
<h1>Calibration trajectory verifier</h1><p class='hint'>Offline replacement for live mission-gateway perception. Select a refined calibration capture, compare LLM and HRRT trajectories, and inspect the residual tube.</p>
<div class='grid'><section>
<form class='card' method='post' action='/prepare'><h2>1. Environment and language</h2>
<label>Refined calibration row</label><select name='row'>{capture_options}</select>
<label>Mode</label><select name='mode'><option value='env_uncertainty'>With environment uncertainty (LLM uses perceived E-hat)</option><option value='planner_only'>Planner uncertainty only (both use ground-truth E)</option></select>
<label>L1 — goal instruction</label><textarea name='l1'>Go to the goal while avoiding all obstacles.</textarea>
<label>L2 — expert preference</label><textarea name='l2'>Prefer the route with the largest clearance.</textarea>
<button>Generate HRRT route family</button></form>
<form class='card' method='post' action='/score'><h2>2. Human expert selection</h2>
<input type='hidden' name='row' value='{selected}'><input type='hidden' name='mode' value='{html.escape(state.get('mode','env_uncertainty'))}'><input type='hidden' name='l1' value='{html.escape(state.get('l1',''))}'><input type='hidden' name='l2' value='{html.escape(state.get('l2',''))}'>
<label>Selected HRRT expert route</label><select name='route'>{route_options}</select>
<button {'disabled' if not routes else ''}>Score and plot trajectories</button></form>
<div class='card'><h2>Result</h2>{score_block}<p class='hint'>{html.escape(message)}</p><pre>{html.escape(result_json)}</pre></div>
</section><section class='card'><h2>Trajectory and safety-tube view</h2>{plot}<p class='legend'>Blue: LLM trajectory and q-tube · green: selected HRRT expert and q-tube · red: ground-truth obstacle · orange: perceived obstacle.</p></section></div></main></body></html>"""


def plot_svg(result):
    points = []
    for key in ("expert_waypoints", "predicted_waypoints"):
        points += [(float(p["x"]), float(p["y"])) for p in result[key]]
    for obstacle in result["ground_truth_obstacles"] + result["planning_obstacles"]:
        points += [(float(obstacle["min_corner"][0]), float(obstacle["min_corner"][1])), (float(obstacle["max_corner"][0]), float(obstacle["max_corner"][1]))]
    if not points:
        return ""
    min_x, max_x = min(p[0] for p in points) - 0.5, max(p[0] for p in points) + 0.5
    min_y, max_y = min(p[1] for p in points) - 0.5, max(p[1] for p in points) + 0.5
    width, height = 900, 600
    def xy(point):
        return ((point[0] - min_x) / (max_x - min_x) * width, height - (point[1] - min_y) / (max_y - min_y) * height)
    def polyline(waypoints, color, width_px=4):
        coords = " ".join(f"{xy((float(p['x']), float(p['y'])))[0]:.1f},{xy((float(p['x']), float(p['y'])))[1]:.1f}" for p in waypoints)
        return f"<polyline points='{coords}' fill='none' stroke='{color}' stroke-width='{width_px}'/>"
    scale = min(width / (max_x - min_x), height / (max_y - min_y))
    def tube(samples, color):
        radius = float(result["tube_quantile"]) * scale
        stride = max(1, len(samples) // 28)
        return "".join(
            f"<circle cx='{xy((float(sample['x'][0]), float(sample['x'][1])))[0]:.1f}' "
            f"cy='{xy((float(sample['x'][0]), float(sample['x'][1])))[1]:.1f}' r='{radius:.1f}' "
            f"fill='{color}' fill-opacity='.08' stroke='{color}' stroke-opacity='.18'/>"
            for sample in samples[::stride]
        )
    shapes = []
    for obstacle in result["ground_truth_obstacles"]:
        x0, y0 = xy((float(obstacle["min_corner"][0]), float(obstacle["max_corner"][1])))
        x1, y1 = xy((float(obstacle["max_corner"][0]), float(obstacle["min_corner"][1])))
        shapes.append(f"<rect x='{x0:.1f}' y='{y0:.1f}' width='{x1-x0:.1f}' height='{y1-y0:.1f}' fill='#ef444455' stroke='#b91c1c'/>")
    for obstacle in result["planning_obstacles"]:
        x0, y0 = xy((float(obstacle["min_corner"][0]), float(obstacle["max_corner"][1])))
        x1, y1 = xy((float(obstacle["max_corner"][0]), float(obstacle["min_corner"][1])))
        shapes.append(f"<rect x='{x0:.1f}' y='{y0:.1f}' width='{x1-x0:.1f}' height='{y1-y0:.1f}' fill='#f59e0b22' stroke='#d97706' stroke-dasharray='6 4'/>")
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='LLM and HRRT trajectories'>"
        f"{''.join(shapes)}{tube(result['expert_samples'], '#16a34a')}"
        f"{tube(result['predicted_samples'], '#2563eb')}"
        f"{polyline(result['expert_waypoints'], '#16a34a')}"
        f"{polyline(result['predicted_waypoints'], '#2563eb')}"
        f"<text x='15' y='25' fill='#111827'>q={result['tube_quantile']:.3f} m</text></svg>"
    )


class GatewayHandler(BaseHTTPRequestHandler):
    state = None
    args = None
    def do_GET(self):
        body = html_page(self.state, self.args)
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body.encode())
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        form = {key: values[-1] for key, values in parse_qs(self.rfile.read(length).decode()).items()}
        try:
            row_number = int(form.get("row", self.state["row_number"]))
            capture_id, rows = choose_capture(self.state["captures"], row_number)
            gt, perceived, start, goal, workspace = scene_setup(rows, self.args)
            mode = form.get("mode", "env_uncertainty")
            planning = perceived if mode == "env_uncertainty" else gt
            if self.path == "/prepare":
                family = plan_family(start, goal, gt, workspace, self.args)
                self.state.update({"row_number": row_number, "capture_id": capture_id, "rows": rows, "mode": mode, "l1": form.get("l1", ""), "l2": form.get("l2", ""), "routes": route_cards(family["routes"]), "route_data": family["routes"], "result": None, "results": {}})
                message = f"Generated {len(family['routes'])} HRRT routes for capture {capture_id}."
            elif self.path == "/score":
                if not self.state.get("route_data"):
                    raise ValueError("generate the HRRT route family first")
                expert = next(route for route in self.state["route_data"] if route["route_id"] == form["route"])
                results = {}
                for mode_name, mode_obstacles in (("env_uncertainty", perceived), ("planner_only", gt)):
                    predicted = planner_waypoints(start, goal, mode_obstacles, workspace, form.get("l1", ""), self.args)
                    if not safe_path(
                        predicted,
                        mode_obstacles,
                        workspace,
                        self.args.clearance_m,
                    ):
                        raise ValueError(f"LLM trajectory failed collision checking in {mode_name}")
                    scored = score_pair(expert["waypoints"], predicted, self.args)
                    results[mode_name] = public_result(
                        capture_id, rows, mode_name, form.get("l1", ""), form.get("l2", ""),
                        expert, predicted, scored, (gt, mode_obstacles)
                    )
                self.state.update({"result": results[mode], "results": results, "mode": mode, "l1": form.get("l1", ""), "l2": form.get("l2", "")})
                message = f"Scored both modes against HRRT route {expert['route_id']} using ideal double-integrator residual."
            else:
                raise ValueError("unknown action")
        except Exception as exc:
            message = f"Error: {exc}"
        body = html_page(self.state, self.args, message)
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body.encode())
    def log_message(self, *_args):
        return


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Refined calibration raw CSV.")
    parser.add_argument("--row", type=int, default=1, help="1-based capture row to select initially.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--planner-provider", choices=("mock", "vllm"), default="mock")
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--llama-model", default="hrrt_planner")
    parser.add_argument("--goal-x", type=float, default=0.0); parser.add_argument("--goal-y", type=float, default=1.5)
    parser.add_argument("--start-x", type=float, default=-3.5); parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--fixed-z", type=float, default=-0.5)
    parser.add_argument("--workspace-x-min", type=float, default=-3.0); parser.add_argument("--workspace-x-max", type=float, default=3.0)
    parser.add_argument("--workspace-y-min", type=float, default=-3.0); parser.add_argument("--workspace-y-max", type=float, default=3.0)
    parser.add_argument("--clearance-m", type=float, default=0.40); parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=17); parser.add_argument("--hrrt-iterations", type=int, default=300)
    parser.add_argument("--max-candidates", type=int, default=8); parser.add_argument("--max-waypoints", type=int, default=8)
    return parser


def main():
    args = build_parser().parse_args()
    captures = load_captures(args.dataset)
    choose_capture(captures, args.row)
    state = {"captures": captures, "row_number": args.row, "routes": [], "route_data": [], "result": None, "results": {}}
    GatewayHandler.state = state; GatewayHandler.args = args
    server = ThreadingHTTPServer((args.host, args.port), GatewayHandler)
    print(f"Calibration verifier listening at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
