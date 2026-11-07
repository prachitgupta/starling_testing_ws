"""Read-only model/pipeline audit; writes results only beside this script."""
import hashlib
import json
import re
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "fine_tuning/scripts"))
import calibration_verifier_gateway as g
from lqr import certified_metric_alpha
from postprocess_residual_calibration import ideal_metric

OUT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "fine_tuning/outputs/final-transformer-vertical-teacher_only_80_10_10/final_transformer_vertical_teacher_only_80_10_10_best/checkpoint"
DATASET = ROOT / "fine_tuning/datasets/calibration_residual_raw_person_outline.csv"
args = g.build_parser().parse_args([
    "--dataset", str(DATASET), "--planner-provider", "transformer",
    "--transformer-checkpoint", str(CHECKPOINT), "--intent-provider", "mock",
])
torch.set_num_threads(4)
args.transformer_planner = g.WaypointTransformerPlanner(CHECKPOINT, "auto")
captures = g.load_captures(DATASET)
preference = "go to the goal in shortest time possible"

def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")

def predictions(row_number, goal):
    cid, rows = captures[row_number - 1]
    gt, perceived, start, workspace = g.scene_setup(rows, args)
    modes = {}
    for mode, obstacles in g.transformer_environment_pairs(args, gt, perceived):
        raw = g.planner_waypoints(start, goal, obstacles, workspace, preference, args)
        refined = g.refine_waypoints(raw, workspace, obstacles, args.clearance_m, args.fixed_z)
        ok, reason, metrics = g.verify_waypoints(refined, start, goal, obstacles, workspace, args.clearance_m)
        modes[mode] = dict(raw=raw, refined=refined, passed=ok, reason=reason, metrics=metrics)
    return cid, rows, gt, perceived, start, workspace, modes

# Reproduce the currently visible failed scene, including a feedback retry.
live_goal = dict(x=-1.0, y=-0.6, z=-0.5)
cid, rows, gt, perceived, start, workspace, modes = predictions(15, live_goal)
retry = g.planner_waypoints(start, live_goal, perceived, workspace, preference, args,
    feedback="Previous plan failed verification: monotonic_goal_progress. Regenerate with larger clearance.",
    prompt_text="Different prompt containing the verifier feedback table.")
retry_difference = float(np.max(np.abs(np.array([[p['x'],p['y']] for p in retry]) - np.array([[p['x'],p['y']] for p in modes['env_uncertainty']['raw']]))))
dump("live_failed_scene.json", dict(row=15, capture_id=cid, start=start, goal=live_goal,
    preference=preference, modes=modes, retry=retry, retry_max_coordinate_difference=retry_difference))
print("LIVE_RETRY", retry_difference, {k:v['reason'] for k,v in modes.items()}, flush=True)

# Choose an illustrative passing case; log every attempted case. This search is
# for a worked example only and is not a calibration or coverage estimate.
attempts = []
selected = None
for row_number in [15] + [i for i in range(1, len(captures)+1) if i != 15]:
    for gx, gy in [(-1.0, -0.6), (-0.45, 2.4), (-2.0, 2.0), (1.5, 2.0)]:
        goal = dict(x=gx, y=gy, z=-0.5)
        try:
            cid, rows, gt, perceived, start, workspace, modes = predictions(row_number, goal)
            attempt = dict(row=row_number, goal=goal, modes={k:v['reason'] for k,v in modes.items()})
            attempts.append(attempt)
            if not all(v['passed'] for v in modes.values()):
                continue
            family = g.plan_family(start, goal, gt, workspace, args)
            routes = family['routes']
            expert = min(routes, key=lambda r: (r['path_length_m'], r['route_id']))
            refined_expert = g.refine_waypoints(expert['waypoints'], workspace, gt, args.clearance_m, args.fixed_z)
            ok, reason, _ = g.verify_waypoints(refined_expert, start, goal, gt, workspace, args.clearance_m)
            attempt['expert_reason'] = reason
            if not ok:
                continue
            selected = (row_number, cid, rows, gt, perceived, start, workspace, goal, family, expert)
            break
        except (ValueError, RuntimeError) as exc:
            attempts.append(dict(row=row_number, goal=goal, error=str(exc)))
    if selected:
        break
    print("SEARCH", row_number, "attempts", len(attempts), flush=True)
dump("example_search.json", attempts)
if selected is None:
    raise RuntimeError("No passing worked example found; see logged search.")
row_number, cid, rows, gt, perceived, start, workspace, goal, family, expert = selected

class FixedShortestSelector:
    def select(self, text, cards, conversation):
        return SimpleNamespace(status="SELECTED", selected_route_id=expert['route_id'],
            reason="Audit explicitly selects the shortest generated HRRT route for the stated shortest-time preference.")

state = dict(captures=captures, row_number=row_number, route_data=family['routes'],
    routes=g.route_cards(family['routes']), goal=goal, mode='env_uncertainty',
    route_selector=FixedShortestSelector(), lock=threading.Lock(), job_running=True,
    l1="Manual goal for reproducible numerical audit", l2=preference)
g.run_scoring_job(state, args, dict(row=row_number, mode='env_uncertainty', l1=state['l1'], l2=preference))
if state['progress'].get('error'):
    dump('pipeline_error.json', state['progress'])
    raise RuntimeError(state['progress']['message'])

b, p, k = ideal_metric()
a = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]], dtype=float)
c = np.eye(4)[:2]
alpha = certified_metric_alpha(a,b,k,p)
gain = float(np.sqrt(np.linalg.eigvalsh(c @ np.linalg.inv(p) @ c.T).max()))
details = {}
for mode, result in state['results'].items():
    i = max(range(len(result['expert_samples'])), key=lambda j: (
        lambda r: float(r @ p @ r))(b @ (np.asarray(result['predicted_samples'][j]['u'])-np.asarray(result['expert_samples'][j]['u']) + k @ (np.asarray(result['predicted_samples'][j]['x'])-np.asarray(result['expert_samples'][j]['x'])))))
    es, ps = result['expert_samples'][i], result['predicted_samples'][i]
    dx = np.asarray(ps['x'])-np.asarray(es['x'])
    du = np.asarray(ps['u'])-np.asarray(es['u'])
    kd = k @ dx
    v = du+kd
    residual = b @ v
    svg = g.plot_svg(result)
    (OUT / (mode + '.svg')).write_text(svg)
    (OUT / (mode + '.html')).write_text('<!doctype html><html><meta charset="utf-8"><title>Gateway radius audit</title><body><h1>Reproduced gateway result: '+mode+'</h1><p>Actual HRRT and Transformer; explicit shortest-route selection and manual goal. This is an illustrative sample, not a calibrated coverage claim.</p>'+svg+'</body></html>')
    positions = np.array([np.asarray(t['x'])[:2]-np.asarray(e['x'])[:2] for e,t in zip(result['expert_samples'],result['predicted_samples'])])
    details[mode] = dict(score=result['score'], time_s=es['t'], expert_sample=es, predicted_sample=ps,
        delta_state=dx.tolist(), delta_control=du.tolist(), K_delta_state=kd.tolist(),
        control_residual=v.tolist(), residual=residual.tolist(), squared_metric=float(residual@p@residual),
        website_label=f"{result['tube_quantile']:.5f} m", svg_circle_radius=round(result['score']*min(900/8,650/6),1),
        svg_actual_circle_radius=re.search(r"r='([^']+)' fill='#16a34a' fill-opacity",svg).group(1),
        same_time_position_mean_m=float(np.linalg.norm(positions,axis=1).mean()),
        same_time_position_max_m=float(np.linalg.norm(positions,axis=1).max()),
        projected_radius_if_score_were_q=gain*result['score']/alpha,
        number_of_samples=len(result['expert_samples']), horizon_s=result['expert_samples'][-1]['t'])
    assert abs(float(np.sqrt(residual@p@residual))-result['score']) < 1e-10
    assert abs(es['t']-result['score_max_time_s']) < 1e-10
dump('gateway_results.json', state['results'])
dump('numerical_audit.json', dict(checkpoint=str(CHECKPOINT), dataset=str(DATASET), row=row_number,
    capture_id=cid, start=start, goal=goal, preference=preference,
    expert_selection='Explicit shortest generated route; no OpenAI selection call',
    route_cards=g.route_cards(family['routes']), selected_route_id=expert['route_id'],
    P=p.tolist(),K=k.tolist(),B=b.tolist(),alpha=alpha,projection_gain=gain,
    radius_conversion_gain=gain/alpha, modes=details,
    checkpoint_sha256=hashlib.file_digest(CHECKPOINT.joinpath('model.pt').open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else None))
print('RESULT',json.dumps(dict(row=row_number,capture_id=cid,start=start,goal=goal,P=p.tolist(),K=k.tolist(),alpha=alpha,projection_gain=gain,modes=details),indent=2),flush=True)
