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
import threading
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_AUDIO_BYTES = 10 * 1024 * 1024
AUDIO_EXTENSIONS = {
    "audio/flac": "flac", "audio/m4a": "m4a", "audio/mp4": "mp4",
    "audio/mpeg": "mp3", "audio/ogg": "ogg", "audio/wav": "wav",
    "audio/webm": "webm", "audio/x-m4a": "m4a", "audio/x-wav": "wav",
}


def audio_file_metadata(content_type):
    mime_type = str(content_type).split(";", 1)[0].strip().lower()
    extension = AUDIO_EXTENSIONS.get(mime_type)
    if extension is None:
        raise ValueError(f"unsupported audio type: {mime_type or 'missing'}")
    return mime_type, extension

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
MISSION_SCRIPT_DIR = SCRIPT_DIR.parents[1] / "scripts"
if str(MISSION_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION_SCRIPT_DIR))

from hrrt_star import normalize_obstacles, plan_hrrt_star  # noqa: E402
from min_control_qp import generate_shared_pair  # noqa: E402
from postprocess_residual_calibration import (  # noqa: E402
    environments,
    residual_score,
    safe_path,
    vllm_predicted_waypoints,
)
from interactive_mission_gateway import (  # noqa: E402
    MockIntentParser,
    MockRouteSelector,
    OpenAIIntentParser,
    OpenAIRouteSelector,
    normalize_goal_relations,
    obstacle_catalog,
    range_constrained_goal,
    route_cards as mission_route_cards,
)
from prompt_generator import INSTRUCTIONS, PromptGenerator  # noqa: E402
from refinment import PathRefinement  # noqa: E402
from verifier import PathVerifier  # noqa: E402


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
    workspace = {
        "x": [args.workspace_x_min, args.workspace_x_max],
        "y": [args.workspace_y_min, args.workspace_y_max],
        "z": args.fixed_z,
    }
    return ground_truth, perceived, start, workspace


def derive_goal(l1, perceived, goal_obstacles, start, workspace, intent_parser, args):
    intent = intent_parser.parse(l1, obstacle_catalog(perceived), [])
    if intent.status != "READY":
        raise ValueError(intent.clarifying_question or f"intent parser returned {intent.status}")
    if intent.intent_type != "NAVIGATION" or intent.navigation_action not in ("HOVER", "GO_TO"):
        raise ValueError("L1 must describe object-relative HOVER or GO_TO navigation")
    relations = normalize_goal_relations(
        intent.relations,
        goal_obstacles,
        default_standoff=args.default_standoff_m,
        clearance=args.clearance_m,
        guard_band=0.0,
        default_range_half_width=args.default_goal_range_half_width_m,
        exact_distance_tolerance=args.exact_goal_distance_tolerance_m,
    )
    goal, relation_results = range_constrained_goal(
        start,
        relations,
        goal_obstacles,
        workspace,
        args.fixed_z,
        args.clearance_m,
        args.goal_sample_resolution_m,
    )
    return intent, relations, relation_results, goal


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


def planner_waypoints(start, goal, planning_obstacles, workspace, l1, args, feedback="", prompt_text=""):
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
        start, goal, planning_obstacles, workspace, l1, planner_args,
        feedback=feedback, validate=False, prompt_text=prompt_text,
    )


def configured_verifier(workspace, clearance):
    verifier = PathVerifier.__new__(PathVerifier)
    verifier.safety_margin_m = float(clearance)
    verifier.interpolation_spacing_m = 1.0
    verifier.cruise_speed_mps = 0.3
    verifier.nominal_dt_s = verifier.interpolation_spacing_m / verifier.cruise_speed_mps
    verifier.max_velocity_mps = 0.5
    verifier.max_acceleration_mps2 = 0.5
    verifier.goal_tolerance_m = 0.05
    verifier.start_tolerance_m = 0.10
    verifier.progress_tolerance_m = 1e-3
    return verifier


def refine_waypoints(waypoints, workspace, obstacles, clearance, fixed_z):
    refiner = PathRefinement.__new__(PathRefinement)
    refiner.interpolation_spacing_m = 1.0
    refiner.safety_margin_m = float(clearance)
    refiner.nudge_epsilon_m = 0.02
    refiner.fixed_z = float(fixed_z)
    return refiner.interpolate_waypoints(waypoints, workspace, obstacles)


def verify_waypoints(waypoints, start, goal, obstacles, workspace, clearance):
    """Use the production ROS2 PathVerifier metrics without starting ROS nodes."""
    verifier = configured_verifier(workspace, clearance)
    metrics = verifier.compute_metrics({
        "waypoints": waypoints,
        "start": start,
        "goal": goal,
        "obstacles": obstacles,
        "workspace": workspace,
    })
    failed = metrics["failed_constraints"]
    reason = "passed" if not failed else ", ".join(failed)
    return bool(metrics["passed"]), reason, metrics


def planner_prompt(start, goal, obstacles, workspace, clearance, l1="", feedback=""):
    prompt_builder = PromptGenerator.__new__(PromptGenerator)
    prompt_builder.workspace_x = (float(workspace["x"][0]), float(workspace["x"][1]))
    prompt_builder.workspace_y = (float(workspace["y"][0]), float(workspace["y"][1]))
    prompt_builder.fixed_z = float(workspace["z"])
    prompt_builder.clearance_m = float(clearance)
    prompt = "\n".join([INSTRUCTIONS, f"Operator instruction: {l1}", prompt_builder.build_nl_env(start, goal, obstacles), prompt_builder.constraints()])
    if feedback:
        prompt += "\n" + feedback
    return prompt


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


def public_result(capture_id, rows, mode, l1, l2, expert, raw_predicted, predicted, result, environments_used, start, goal, perceived, workspace, verification):
    return {
        "capture_id": capture_id,
        "session_id": rows[0].get("session_id", ""),
        "mode": mode,
        "l1": l1,
        "l2": l2,
        "expert_waypoints": expert["waypoints"],
        "expert_route_id": expert["route_id"],
        "predicted_waypoints": predicted,
        "raw_predicted_waypoints": raw_predicted,
        "verification": verification,
        "start": start,
        "goal": goal,
        "score": result["score"],
        "score_max_time_s": result["max_time_s"],
        "residual": result["residual"],
        "tube_quantile": result["score"],
        "expert_samples": result["expert_samples"],
        "predicted_samples": result["predicted_samples"],
        "ground_truth_obstacles": environments_used[0],
        "planning_obstacles": environments_used[1],
        "perceived_obstacles": perceived,
        "workspace": workspace,
    }


STYLE = """
body{font-family:system-ui,sans-serif;background:#07111d;color:#e7f4ff;margin:0;padding:20px}
main{max-width:1400px;margin:auto}.grid{display:grid;grid-template-columns:360px 1fr;gap:16px}
.card{background:#102437;border:1px solid #29445b;border-radius:14px;padding:16px;margin-bottom:14px}
label{display:block;color:#a9c3d6;font-size:12px;margin:10px 0 4px}input,select,textarea,button{width:100%;box-sizing:border-box;padding:9px;border-radius:8px;border:1px solid #41617a;background:#081725;color:#eef8ff}textarea{min-height:62px}button{background:#2788c7;font-weight:700;cursor:pointer;margin-top:12px}.secondary{background:#21445b}
pre{white-space:pre-wrap;overflow:auto}.metric{font-size:19px;color:#59e0aa;font-weight:800}.hint{color:#a9c3d6;font-size:12px;line-height:1.4}
.pipeline-log{max-height:190px;overflow:auto;padding-left:22px;font-size:12px;color:#cbd5e1}.pipeline-log li{margin:5px 0}.pipeline-log li:last-child{color:#59e0aa;font-weight:700}
.error{color:#fecaca;background:#7f1d1d55;border:1px solid #ef4444;border-radius:8px;padding:10px;font-weight:700}
svg{width:100%;height:auto;background:#eff7fb;border-radius:10px}.legend{font-size:12px;color:#b9d0e0}
.plot-key{display:flex;flex-wrap:wrap;gap:8px 14px;margin:10px 0;padding:10px;background:#081725;border-radius:8px;font-size:12px}.key-item{display:flex;align-items:center;gap:6px}.key-line{width:30px;border-top:4px solid}.key-dashed{border-top-style:dashed}.score-lock{border:2px solid #59e0aa;background:#0b2b2b;padding:12px;border-radius:10px;margin-bottom:10px}
.speech-row{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center}.speech-row button{width:48px;margin-top:0}.listening{background:#b91c1c}
.manual-goal{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
@media(max-width:850px){.grid{grid-template-columns:1fr}body{padding:8px}}
"""

SPEECH_SCRIPT = """
<script>
const types=['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/mp4'],audioType=window.MediaRecorder?types.find(t=>MediaRecorder.isTypeSupported(t))||'':'';
function setupSpeech(buttonId,fileId,targetId,statusId){const button=document.getElementById(buttonId),fileInput=document.getElementById(fileId),target=document.getElementById(targetId),status=document.getElementById(statusId);let recorder=null,stream=null,timer=null;
async function transcribe(audio,type){if(!audio.size)return;button.disabled=true;status.textContent='Transcribing…';try{const response=await fetch('/api/transcribe',{method:'POST',headers:{'Content-Type':type,'X-Speech-Language':navigator.language||'en-US'},body:audio}),result=await response.json();if(!response.ok)throw new Error(result.error||'transcription failed');target.value=result.text.trim();target.focus();status.textContent='Transcript ready. Review it before generating routes.'}catch(error){status.textContent=`Speech transcription failed: ${error.message}`}finally{button.disabled=false}}
function stopRecording(){if(timer)clearTimeout(timer);if(stream)stream.getTracks().forEach(track=>track.stop());stream=null;button.classList.remove('listening');button.textContent='●'}
fileInput.onchange=async()=>{const file=fileInput.files[0];if(file)await transcribe(file,file.type);fileInput.value=''};
if(!navigator.mediaDevices?.getUserMedia||!window.MediaRecorder)button.onclick=()=>fileInput.click();else button.onclick=async()=>{if(recorder?.state==='recording'){recorder.stop();return}try{stream=await navigator.mediaDevices.getUserMedia({audio:true});const chunks=[];recorder=new MediaRecorder(stream,audioType?{mimeType:audioType}:undefined);recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};recorder.onstop=async()=>{const type=recorder.mimeType||audioType;stopRecording();await transcribe(new Blob(chunks,{type}),type)};recorder.start();button.classList.add('listening');button.textContent='■';status.textContent='Listening…';timer=setTimeout(()=>{if(recorder?.state==='recording')recorder.stop()},20000)}catch(error){stopRecording();status.textContent=`Microphone unavailable: ${error.message}`}}}
setupSpeech('speech-l1','speech-file-l1','l1','speech-status-l1');setupSpeech('speech-l2','speech-file-l2','l2','speech-status-l2');
const scoreForm=document.getElementById('score-form'),scoreButton=document.getElementById('score-button'),pipelineStatus=document.getElementById('pipeline-status'),plotContainer=document.getElementById('plot-container'),plotLegend=document.getElementById('plot-legend'),scoreResults=document.getElementById('score-results'),resultJson=document.getElementById('result-json');
function showScores(progress){scoreResults.replaceChildren();const box=document.createElement('div');box.className='score-lock';const title=document.createElement('div');title.className='metric';title.textContent=`Final scores — ${progress.selected_mode} displayed`;box.appendChild(title);for(const [name,item] of Object.entries(progress.scores||{})){const line=document.createElement('p');line.textContent=`${name}: score = ${Number(item.score).toFixed(5)} m · tube q = ${Number(item.tube_quantile).toFixed(5)} m · max residual at ${Number(item.max_time_s).toFixed(2)} s`;box.appendChild(line)}scoreResults.appendChild(box);if(progress.result)resultJson.textContent=JSON.stringify(progress.result,null,2)}
function showHistory(history){const log=document.getElementById('pipeline-log');log.replaceChildren();for(const event of history||[]){const item=document.createElement('li');item.textContent=`${event.stage}: ${event.message}`;log.appendChild(item)}log.scrollTop=log.scrollHeight}
async function pollPipeline(){const response=await fetch('/api/progress',{cache:'no-store'}),progress=await response.json();pipelineStatus.textContent=`${progress.stage}: ${progress.message}`;showHistory(progress.history);if(progress.plot)plotContainer.innerHTML=progress.plot;if(progress.legend)plotLegend.textContent=progress.legend;if(progress.done){if(progress.error){pipelineStatus.classList.add('error');scoreButton.disabled=false}else{showScores(progress);scoreButton.disabled=true;scoreButton.textContent='Final result locked — change row to run again'}return}setTimeout(pollPipeline,350)}
scoreForm.addEventListener('submit',async event=>{event.preventDefault();scoreButton.disabled=true;scoreResults.replaceChildren();resultJson.textContent='';pipelineStatus.classList.remove('error');pipelineStatus.textContent='Starting LLM route selection…';const response=await fetch('/api/score',{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams(new FormData(scoreForm))});const payload=await response.json();if(!response.ok){pipelineStatus.textContent=payload.error||'Unable to start scoring';pipelineStatus.classList.add('error');scoreButton.disabled=false;return}pollPipeline()});
</script>
"""


def html_page(state, args, message=""):
    capture_options = "".join(
        f'<option value="{i}"{" selected" if i == state["row_number"] else ""}>row {i}: capture {html.escape(cid)}</option>'
        for i, (cid, _) in enumerate(state["captures"], 1)
    )
    selected = state["row_number"]
    routes = state.get("routes", [])
    result = state.get("result")
    results = state.get("results", {})
    result_json = compact(result) if result else ""
    l1_value = html.escape(state.get("l1") or "Go near the person.")
    l2_value = html.escape(state.get("l2") or "Prefer the route with the largest clearance.")
    view_error = ""
    try:
        preview = scene_preview(state, args)
        plot = plot_svg(result) if result else plot_scene(preview)
    except (KeyError, TypeError, ValueError) as error:
        result = None
        results = {}
        result_json = ""
        plot = ""
        view_error = f"Error: {error}"
        message = message or view_error
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
    stage_hint = "" if view_error else ("Final view: verified HRRT and refined LLM trajectories with QP-optimized samples and score tubes." if result else ("Select an HRRT route above; all generated candidates are shown in the plot." if routes else "Dataset scene loaded: red is ground truth, orange is perceived, and the blue marker is the drone pose."))
    legend = "" if view_error else f"{stage_hint} Red: ground-truth obstacle · orange dashed box: perceived obstacle · green: HRRT · orange dashed path: raw LLM · blue: refined LLM · blue marker: drone · gold marker: goal."
    message_block = f"<p class='error'>{html.escape(message)}</p>" if message.startswith("Error:") else f"<p class='hint'>{html.escape(message)}</p>"
    goal = state.get("goal")
    if goal:
        goal_card = f"""<div class='card'><h2>2. Goal approval</h2><p>Proposed goal: <b>({goal['x']:.2f}, {goal['y']:.2f}, {goal['z']:.2f})</b></p><form method='post' action='/generate'><input type='hidden' name='row' value='{selected}'><input type='hidden' name='mode' value='{html.escape(state.get('mode','env_uncertainty'))}'><input type='hidden' name='l1' value='{html.escape(state.get('l1',''))}'><button name='decision' value='accept'>Accept proposed goal and generate HRRT</button><button class='secondary' formaction='/reject'>Reject proposed goal</button><label>Manual goal (x, y, z)</label><div class='manual-goal'><input name='goal_x' value='{goal['x']:.2f}'><input name='goal_y' value='{goal['y']:.2f}'><input name='goal_z' value='{goal['z']:.2f}'></div><button name='decision' value='manual'>Use manual goal and generate HRRT</button></form></div>"""
    else:
        goal_card = "<div class='card'><h2>2. Goal approval</h2><p class='hint'>Generate a goal proposal first. HRRT generation is disabled until the proposed goal is accepted or replaced manually.</p></div>"
    return f"""<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'><title>Calibration verifier</title><style>{STYLE}</style></head><body><main>
<h1>Calibration trajectory verifier</h1><p class='hint'>Offline replacement for live mission-gateway perception. Select a refined calibration capture, compare LLM and HRRT trajectories, and inspect the residual tube.</p>
<div class='grid'><section>
<form class='card' method='post' action='/propose'><h2>1. Environment and language</h2>
<label>Refined calibration row</label><select name='row' onchange="window.location='/?row='+this.value">{capture_options}</select>
<label>Mode</label><select name='mode'><option value='env_uncertainty'>With environment uncertainty (LLM uses perceived E-hat)</option><option value='planner_only'>Planner uncertainty only (both use ground-truth E)</option></select>
<label>L1 — object-relative goal instruction</label><div class='speech-row'><textarea id='l1' name='l1'>{l1_value}</textarea><button id='speech-l1' type='button' title='Record L1 prompt' aria-label='Record L1 prompt'>●</button></div><input id='speech-file-l1' type='file' accept='audio/webm,audio/ogg,audio/mp4,audio/x-m4a,audio/mpeg,audio/wav' capture hidden><p id='speech-status-l1' class='hint'>Speak or type L1, then propose a goal.</p>
<button>Propose goal</button></form>
{goal_card}
<form id='score-form' class='card' method='post' action='/score'><h2>3. LLM expert selection and scoring</h2>
<input type='hidden' name='row' value='{selected}'><input type='hidden' name='mode' value='{html.escape(state.get('mode','env_uncertainty'))}'><input type='hidden' name='l1' value='{html.escape(state.get('l1',''))}'>
<label>L2 — route preference for the OpenAI selector</label><div class='speech-row'><textarea id='l2' name='l2'>{l2_value}</textarea><button id='speech-l2' type='button' title='Record L2 preference' aria-label='Record L2 preference'>●</button></div><input id='speech-file-l2' type='file' accept='audio/webm,audio/ogg,audio/mp4,audio/x-m4a,audio/mpeg,audio/wav' capture hidden><p id='speech-status-l2' class='hint'>The LLM selects one displayed HRRT route from this preference.</p>
<button id='score-button' {'disabled' if not routes else ''}>Select route, score, and plot</button><p id='pipeline-status' class='hint'>{'HRRT family ready; enter L2 for LLM route selection.' if routes else 'Waiting for an accepted goal and HRRT route family.'}</p><ol id='pipeline-log' class='pipeline-log'></ol></form>
<div class='card'><h2>Result</h2><div id='score-results'>{score_block}</div>{message_block}<pre id='result-json'>{html.escape(result_json)}</pre></div>
</section><section class='card'><h2>Trajectory and safety-tube view</h2><div class='plot-key'><span class='key-item'><i class='key-line' style='border-color:#16a34a'></i>HRRT expert</span><span class='key-item'><i class='key-line key-dashed' style='border-color:#f97316'></i>Raw LLM</span><span class='key-item'><i class='key-line' style='border-color:#2563eb'></i>Refined LLM</span><span class='key-item'><i class='key-line' style='border-color:#b91c1c'></i>Ground truth obstacle</span><span class='key-item'><i class='key-line key-dashed' style='border-color:#d97706'></i>Perceived obstacle</span><span class='key-item'>Translucent green/blue = safety tubes</span></div><div id='plot-container'>{plot}</div><p id='plot-legend' class='legend'>{html.escape(legend)}</p></section></div>{SPEECH_SCRIPT}</main></body></html>"""


def scene_preview(state, args):
    if all(key in state for key in ("ground_truth_obstacles", "perceived_obstacles", "start")):
        return {
            "ground_truth_obstacles": state["ground_truth_obstacles"],
            "perceived_obstacles": state["perceived_obstacles"],
            "start": state["start"],
            "goal": state.get("goal"),
            "routes": state.get("routes", []),
        }
    rows = state.get("rows")
    if rows is None:
        _, rows = choose_capture(state["captures"], state["row_number"])
    gt, perceived, start, workspace = scene_setup(rows, args)
    return {
        "ground_truth_obstacles": state.get("ground_truth_obstacles", gt),
        "perceived_obstacles": state.get("perceived_obstacles", perceived),
        "start": state.get("start", start),
        "goal": state.get("goal"),
        "routes": state.get("routes", []),
        "workspace": workspace,
    }


def grid_axes(xy, min_x, max_x, min_y, max_y, width, height):
    vertical = "".join(
        f"<line x1='{xy((x, min_y))[0]:.1f}' y1='{xy((x, min_y))[1]:.1f}' x2='{xy((x, max_y))[0]:.1f}' y2='{xy((x, max_y))[1]:.1f}' stroke='#94a3b833'/>"
        f"<text x='{xy((x, min_y))[0]+2:.1f}' y='{height-45:.1f}' fill='#475569' font-size='12'>{x}</text>"
        for x in range(math.ceil(min_x), math.floor(max_x) + 1)
    )
    horizontal = "".join(
        f"<line x1='{xy((min_x, y))[0]:.1f}' y1='{xy((min_x, y))[1]:.1f}' x2='{xy((max_x, y))[0]:.1f}' y2='{xy((max_x, y))[1]:.1f}' stroke='#94a3b833'/>"
        f"<text x='25' y='{xy((min_x, y))[1]+4:.1f}' fill='#475569' font-size='12'>{y}</text>"
        for y in range(math.ceil(min_y), math.floor(max_y) + 1)
    )
    axes = ""
    if min_x <= 0 <= max_x:
        axes += f"<line x1='{xy((0, min_y))[0]:.1f}' y1='{xy((0, min_y))[1]:.1f}' x2='{xy((0, max_y))[0]:.1f}' y2='{xy((0, max_y))[1]:.1f}' stroke='#334155' stroke-width='1.5'/>"
    if min_y <= 0 <= max_y:
        axes += f"<line x1='{xy((min_x, 0))[0]:.1f}' y1='{xy((min_x, 0))[1]:.1f}' x2='{xy((max_x, 0))[0]:.1f}' y2='{xy((max_x, 0))[1]:.1f}' stroke='#334155' stroke-width='1.5'/>"
    sign = f"<g transform='translate({width-145},18)' fill='#0f172a' stroke='#0f172a' font-size='11'><rect x='0' y='0' width='132' height='54' rx='6' fill='#ffffffcc' stroke='#64748b'/><line x1='18' y1='35' x2='52' y2='35' stroke-width='2'/><polygon points='52,35 45,31 45,39' stroke='none'/><line x1='18' y1='35' x2='18' y2='12' stroke-width='2'/><polygon points='18,12 14,19 22,19' stroke='none'/><text x='56' y='39' stroke='none'>+x</text><text x='23' y='13' stroke='none'>+y</text><text x='76' y='20' stroke='none'>+z down</text><text x='76' y='36' stroke='none'>NED top view</text></g>"
    return vertical + horizontal + axes + sign


def plot_scene(scene):
    obstacles = scene["ground_truth_obstacles"] + scene["perceived_obstacles"]
    points = [(float(scene["start"]["x"]), float(scene["start"]["y"]))]
    if scene.get("goal"):
        points.append((float(scene["goal"]["x"]), float(scene["goal"]["y"])))
    for route in scene.get("routes", []):
        points.extend((float(p["x"]), float(p["y"])) for p in route["waypoints"])
    for key in ("selected_waypoints", "predicted_waypoints", "raw_predicted_waypoints"):
        points.extend((float(p["x"]), float(p["y"])) for p in scene.get(key, []))
    for obstacle in obstacles:
        points.extend(((float(obstacle["min_corner"][0]), float(obstacle["min_corner"][1])), (float(obstacle["max_corner"][0]), float(obstacle["max_corner"][1]))))
    if not points:
        return "<p class='hint'>No geometry is available for this capture.</p>"
    limits = scene.get("workspace", {"x": [-3.0, 3.0], "y": [-3.0, 3.0]})
    min_x, max_x = float(limits["x"][0]), float(limits["x"][1])
    min_y, max_y = float(limits["y"][0]), float(limits["y"][1])
    width, height, pad = 900, 650, 62
    def xy(point):
        return (pad + (point[0] - min_x) / max(max_x - min_x, 1e-6) * (width - 2 * pad), height - pad - (point[1] - min_y) / max(max_y - min_y, 1e-6) * (height - 2 * pad))
    def polyline(waypoints, color, width_px=3, dash=""):
        coords = " ".join(f"{xy((float(p['x']), float(p['y'])))[0]:.1f},{xy((float(p['x']), float(p['y'])))[1]:.1f}" for p in waypoints)
        dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
        return f"<polyline points='{coords}' fill='none' stroke='{color}' stroke-width='{width_px}' stroke-opacity='.75'{dash_attr}/>"
    shapes = []
    for obstacle in scene["ground_truth_obstacles"]:
        x0, y0 = xy((float(obstacle["min_corner"][0]), float(obstacle["max_corner"][1])))
        x1, y1 = xy((float(obstacle["max_corner"][0]), float(obstacle["min_corner"][1])))
        shapes.append(f"<rect x='{x0:.1f}' y='{y0:.1f}' width='{x1-x0:.1f}' height='{y1-y0:.1f}' fill='#ef444455' stroke='#b91c1c'/>")
    for obstacle in scene["perceived_obstacles"]:
        x0, y0 = xy((float(obstacle["min_corner"][0]), float(obstacle["max_corner"][1])))
        x1, y1 = xy((float(obstacle["max_corner"][0]), float(obstacle["min_corner"][1])))
        shapes.append(f"<rect x='{x0:.1f}' y='{y0:.1f}' width='{x1-x0:.1f}' height='{y1-y0:.1f}' fill='#f59e0b22' stroke='#d97706' stroke-dasharray='6 4'/>")
    grid = grid_axes(xy, min_x, max_x, min_y, max_y, width, height)
    route_colors = ("#64748b", "#8b5cf6", "#0891b2", "#db2777", "#65a30d", "#ea580c", "#0f766e", "#7c3aed")
    route_lines = "".join(polyline(route["waypoints"], route_colors[i % len(route_colors)]) for i, route in enumerate(scene.get("routes", [])))
    if scene.get("selected_waypoints"):
        route_lines += polyline(scene["selected_waypoints"], "#16a34a", 6)
    if scene.get("raw_predicted_waypoints"):
        route_lines += polyline(scene["raw_predicted_waypoints"], "#f97316", 5, "8 5")
    if scene.get("predicted_waypoints"):
        route_lines += polyline(scene["predicted_waypoints"], "#2563eb", 6)
    start_x, start_y = xy((float(scene["start"]["x"]), float(scene["start"]["y"])))
    markers = f"<circle cx='{start_x:.1f}' cy='{start_y:.1f}' r='9' fill='#2563eb' stroke='#fff' stroke-width='2'/><text x='{start_x+12:.1f}' y='{start_y:.1f}' fill='#111827'>drone</text>"
    if scene.get("goal"):
        goal_x, goal_y = xy((float(scene["goal"]["x"]), float(scene["goal"]["y"])))
        markers += f"<circle cx='{goal_x:.1f}' cy='{goal_y:.1f}' r='9' fill='#eab308' stroke='#111827' stroke-width='2'/><text x='{goal_x+12:.1f}' y='{goal_y:.1f}' fill='#111827'>goal</text>"
    return f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Dataset obstacles, drone pose, proposed goal, and HRRT route candidates'>{grid}{''.join(shapes)}{route_lines}{markers}</svg>"


def plot_svg(result):
    points = []
    for key in ("expert_waypoints", "predicted_waypoints", "raw_predicted_waypoints"):
        points += [(float(p["x"]), float(p["y"])) for p in result.get(key, [])]
    for key in ("start", "goal"):
        if result.get(key):
            points.append((float(result[key]["x"]), float(result[key]["y"])))
    for obstacle in result["ground_truth_obstacles"] + result.get("perceived_obstacles", result["planning_obstacles"]):
        points += [(float(obstacle["min_corner"][0]), float(obstacle["min_corner"][1])), (float(obstacle["max_corner"][0]), float(obstacle["max_corner"][1]))]
    if not points:
        return ""
    limits = result.get("workspace", {"x": [-3.0, 3.0], "y": [-3.0, 3.0]})
    min_x, max_x = float(limits["x"][0]), float(limits["x"][1])
    min_y, max_y = float(limits["y"][0]), float(limits["y"][1])
    width, height, pad = 900, 650, 62
    def xy(point):
        return (pad + (point[0] - min_x) / max(max_x - min_x, 1e-6) * (width - 2 * pad), height - pad - (point[1] - min_y) / max(max_y - min_y, 1e-6) * (height - 2 * pad))
    def polyline(waypoints, color, width_px=4, dash=""):
        coords = " ".join(f"{xy((float(p['x']), float(p['y'])))[0]:.1f},{xy((float(p['x']), float(p['y'])))[1]:.1f}" for p in waypoints)
        dash_attr = f" stroke-dasharray='{dash}'" if dash else ""
        return f"<polyline points='{coords}' fill='none' stroke='{color}' stroke-width='{width_px}'{dash_attr}/>"
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
    for obstacle in result.get("perceived_obstacles", result["planning_obstacles"]):
        x0, y0 = xy((float(obstacle["min_corner"][0]), float(obstacle["max_corner"][1])))
        x1, y1 = xy((float(obstacle["max_corner"][0]), float(obstacle["min_corner"][1])))
        shapes.append(f"<rect x='{x0:.1f}' y='{y0:.1f}' width='{x1-x0:.1f}' height='{y1-y0:.1f}' fill='#f59e0b22' stroke='#d97706' stroke-dasharray='6 4'/>")
    grid = grid_axes(xy, min_x, max_x, min_y, max_y, width, height)
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='LLM and HRRT trajectories'>"
        f"{grid}{''.join(shapes)}{tube(result['expert_samples'], '#16a34a')}"
        f"{tube(result['predicted_samples'], '#2563eb')}"
        f"{polyline(result['expert_waypoints'], '#16a34a')}"
        f"{polyline(result.get('raw_predicted_waypoints', []), '#f97316', 4, '8 5')}"
        f"{polyline(result['predicted_waypoints'], '#2563eb')}"
        f"<circle cx='{xy((float(result['start']['x']), float(result['start']['y'])))[0]:.1f}' cy='{xy((float(result['start']['x']), float(result['start']['y'])))[1]:.1f}' r='9' fill='#2563eb' stroke='#fff' stroke-width='2'/><circle cx='{xy((float(result['goal']['x']), float(result['goal']['y'])))[0]:.1f}' cy='{xy((float(result['goal']['x']), float(result['goal']['y'])))[1]:.1f}' r='9' fill='#eab308' stroke='#111827' stroke-width='2'/>"
        f"<rect x='12' y='10' width='220' height='34' rx='6' fill='#ffffffee' stroke='#334155'/><text x='23' y='32' fill='#111827' font-size='16' font-weight='700'>score = q = {result['tube_quantile']:.5f} m</text></svg>"
    )


def set_progress(state, stage, message, plot="", legend="", done=False, error=False, scores=None, result=None, selected_mode=""):
    payload = {"stage": stage, "message": message, "plot": plot, "legend": legend, "done": done, "error": error}
    if scores is not None:
        payload.update({"scores": scores, "result": result, "selected_mode": selected_mode})
    with state["lock"]:
        history = state.setdefault("progress_history", [])
        history.append({"stage": stage, "message": message})
        del history[:-30]
        payload["history"] = list(history)
        state["progress"] = payload


def run_scoring_job(state, args, form):
    last_plot = ""
    try:
        row_number = int(form.get("row", state["row_number"]))
        capture_id, rows = choose_capture(state["captures"], row_number)
        gt, perceived, start, workspace = scene_setup(rows, args)
        mode = form.get("mode", "env_uncertainty")
        l1 = form.get("l1", "").strip()
        l2 = form.get("l2", "").strip()
        routes = state.get("route_data", [])
        if not routes or row_number != state.get("row_number"):
            raise ValueError("generate the HRRT route family for this row first")
        if not l2:
            raise ValueError("enter an L2 route preference")
        goal = state["goal"]
        preview = scene_preview(state, args)
        last_plot = plot_scene(preview)
        set_progress(state, "LLM route selection", f"Matching L2 against {len(routes)} displayed HRRT routes.", last_plot, "All HRRT candidates remain visible while the expert selector decides.")
        cards = mission_route_cards(routes, args.route_nominal_speed_mps)
        selection = state["route_selector"].select(l2, cards, [])
        if selection.status != "SELECTED":
            raise ValueError(selection.clarifying_question or selection.reason)
        expert = next((route for route in routes if str(route["route_id"]) == str(selection.selected_route_id)), None)
        if expert is None:
            raise ValueError("the LLM selector returned a route outside the displayed HRRT family")
        expert_waypoints = refine_waypoints(expert["waypoints"], workspace, gt, args.clearance_m, args.fixed_z)
        expert_ok, expert_reason, expert_metrics = verify_waypoints(
            expert_waypoints, start, goal, gt, workspace, args.clearance_m
        )
        if not expert_ok:
            set_progress(state, "HRRT verification", f"Selected route failed sanity check: {expert_reason}; retrying once.", last_plot, "HRRT verification failed; regenerating one expert route.")
            retry_args = SimpleNamespace(**vars(args))
            retry_args.seed = args.seed + 1
            retry_family = plan_family(start, goal, gt, workspace, retry_args)
            retry_expert = next(
                (route for route in retry_family["routes"] if verify_waypoints(refine_waypoints(route["waypoints"], workspace, gt, args.clearance_m, args.fixed_z), start, goal, gt, workspace, args.clearance_m)[0]),
                None,
            )
            if retry_expert is None:
                raise ValueError(f"HRRT route failed refinement verification: {expert_reason}")
            expert = retry_expert
            expert_waypoints = refine_waypoints(expert["waypoints"], workspace, gt, args.clearance_m, args.fixed_z)
            expert_ok, expert_reason, expert_metrics = verify_waypoints(expert_waypoints, start, goal, gt, workspace, args.clearance_m)
            expert_reason = "passed after one HRRT refinement retry"
        else:
            expert_reason = "passed"
        selected_scene = dict(preview, routes=[], selected_waypoints=expert_waypoints)
        last_plot = plot_scene(selected_scene)
        set_progress(state, "HRRT verification", f"Selected {expert['route_id']} passed sanity verification ({expert_reason}).", last_plot, "Green: verified LLM-selected HRRT expert route.")
        results = {}
        for mode_name, mode_obstacles in (("env_uncertainty", perceived), ("planner_only", gt)):
            set_progress(state, "Trial 1/2 · LLM generation", f"Generating the first {mode_name} LLM prediction.", last_plot, "Green: selected HRRT expert route. Trial 1 prediction will appear in orange.")
            base_prompt = planner_prompt(start, goal, mode_obstacles, workspace, args.clearance_m, l1=l1)
            raw_predicted = planner_waypoints(start, goal, mode_obstacles, workspace, l1, args, prompt_text=base_prompt)
            live_scene = dict(selected_scene, raw_predicted_waypoints=raw_predicted)
            last_plot = plot_scene(live_scene)
            set_progress(state, "Trial 1/2 · refinement", f"First {mode_name} LLM prediction received; running PathRefinement.", last_plot, "Orange dashed: Trial 1 raw LLM prediction · green: HRRT expert.")
            refined_candidate = refine_waypoints(raw_predicted, workspace, mode_obstacles, args.clearance_m, args.fixed_z)
            live_scene["predicted_waypoints"] = refined_candidate
            last_plot = plot_scene(live_scene)
            raw_ok, raw_reason, raw_metrics = verify_waypoints(refined_candidate, start, goal, mode_obstacles, workspace, args.clearance_m)
            trial_records = [{"trial": 1, "passed": raw_ok, "reason": raw_reason, "metrics": raw_metrics}]
            predicted = refined_candidate
            refinement = "trial 1 passed"
            if not raw_ok:
                set_progress(state, "Trial 1/2 · verification failed", f"First {mode_name} prediction failed: {raw_reason}. Generating Trial 2/2 with verifier feedback.", last_plot, "Orange dashed: Trial 1 raw prediction · blue: Trial 1 refined prediction (failed verification).")
                feedback = (
                    "Previous plan failed verification. Regenerate a sparse waypoint plan that fixes the failed "
                    "metrics while keeping the same start, goal, workspace, and obstacle snapshot.\n"
                    f"Failed constraints: {', '.join(raw_metrics['failed_constraints']) or 'unknown'}\n"
                    f"{raw_metrics['feedback_table']}\n"
                    "Prefer a route with larger obstacle clearance, monotonic goal progress, and smoother segment changes."
                )
                raw_predicted = planner_waypoints(start, goal, mode_obstacles, workspace, l1, args, feedback=feedback, prompt_text=planner_prompt(start, goal, mode_obstacles, workspace, args.clearance_m, l1=l1, feedback=feedback))
                retry_scene = dict(selected_scene, raw_predicted_waypoints=raw_predicted)
                last_plot = plot_scene(retry_scene)
                set_progress(state, "Trial 2/2 · LLM generation", f"Second {mode_name} LLM prediction received from the feedback prompt.", last_plot, "Orange dashed now shows Trial 2 raw prediction; the failed Trial 1 display has been replaced.")
                predicted = refine_waypoints(raw_predicted, workspace, mode_obstacles, args.clearance_m, args.fixed_z)
                refined_scene = dict(retry_scene, predicted_waypoints=predicted)
                last_plot = plot_scene(refined_scene)
                set_progress(state, "Trial 2/2 · refinement", f"Trial 2 {mode_name} prediction refined; running the final PathVerifier check.", last_plot, "Orange dashed: Trial 2 raw prediction · blue: Trial 2 refined prediction.")
                refined_ok, refined_reason, refined_metrics = verify_waypoints(predicted, start, goal, mode_obstacles, workspace, args.clearance_m)
                trial_records.append({"trial": 2, "passed": refined_ok, "reason": refined_reason, "metrics": refined_metrics})
                if not refined_ok:
                    set_progress(state, "Trial 2/2 · verification failed", f"Second {mode_name} prediction also failed: {refined_reason}. No QP trajectory or score will be produced.", last_plot, "Final failed view: orange dashed is Trial 2 raw; blue is Trial 2 refined.")
                    raise ValueError(f"No safe LLM plan after 2/2 trials in {mode_name}; Trial 2 failed: {refined_reason}")
                set_progress(state, "Trial 2/2 · verification passed", f"Second {mode_name} prediction passed after one feedback retry.", last_plot, "Orange dashed: Trial 2 raw · blue: verified Trial 2 refined · green: verified HRRT.")
                refinement = f"trial 2 passed after trial 1 failed: {raw_reason}"
            else:
                set_progress(state, "Trial 1/2 · verification passed", f"First {mode_name} prediction passed; Trial 2 is not needed.", last_plot, "Orange dashed: Trial 1 raw · blue: verified Trial 1 refined · green: verified HRRT.")
            set_progress(state, "QP optimization", f"Solving paired minimum-control trajectories for {mode_name}.", last_plot, "Verified HRRT and LLM waypoints are being optimized.")
            scored = score_pair(expert_waypoints, predicted, args)
            result_expert = dict(expert, waypoints=expert_waypoints)
            item = public_result(capture_id, rows, mode_name, l1, l2, result_expert, raw_predicted, predicted, scored, (gt, mode_obstacles), start, goal, perceived, workspace, {"hrrt": expert_reason, "hrrt_metrics": expert_metrics, "llm_trials": trial_records, "llm_refinement": refinement})
            item.update({"expert_selector": "openai" if args.intent_provider == "openai" else "mock", "expert_model": args.expert_model if args.intent_provider == "openai" else "", "selection_reason": selection.reason})
            results[mode_name] = item
        with state["lock"]:
            state.update({"result": results[mode], "results": results, "mode": mode, "l1": l1, "l2": l2, "selected_route_id": expert["route_id"], "selection_reason": selection.reason})
        final_plot = plot_svg(results[mode])
        score_summary = {
            name: {
                "score": item["score"],
                "tube_quantile": item["tube_quantile"],
                "max_time_s": item["score_max_time_s"],
            }
            for name, item in results.items()
        }
        set_progress(
            state,
            "Complete",
            f"Selected {expert['route_id']} and scored both uncertainty modes. Final view is locked until the row changes.",
            final_plot,
            "Green: HRRT expert · orange dashed: raw LLM · blue: refined LLM · translucent green/blue: score tubes.",
            done=True,
            scores=score_summary,
            result=results[mode],
            selected_mode=mode,
        )
    except Exception as error:
        set_progress(state, "Error", str(error), last_plot, "The last completed stage remains visible.", done=True, error=True)
    finally:
        with state["lock"]:
            state["job_running"] = False


class GatewayHandler(BaseHTTPRequestHandler):
    state = None
    args = None
    def do_GET(self):
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/api/progress":
            with self.state["lock"]:
                body = compact(self.state.get("progress", {"stage": "Idle", "message": "", "plot": "", "legend": "", "done": True, "error": False})).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8"); self.end_headers(); self.wfile.write(body)
            return
        if parsed.path == "/" and query.get("row"):
            try:
                row_number = int(query["row"][-1])
                capture_id, rows = choose_capture(self.state["captures"], row_number)
                self.state.update({"row_number": row_number, "capture_id": capture_id, "rows": rows, "routes": [], "route_data": [], "result": None, "results": {}, "progress_history": []})
                for key in ("goal", "start", "ground_truth_obstacles", "perceived_obstacles", "intent", "goal_relations", "goal_relation_results"):
                    self.state.pop(key, None)
            except (TypeError, ValueError) as error:
                self.state["row_error"] = str(error)
        body = html_page(self.state, self.args)
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body.encode())
    def do_POST(self):
        if self.path == "/api/score":
            length = int(self.headers.get("Content-Length", "0"))
            form = {key: values[-1] for key, values in parse_qs(self.rfile.read(length).decode()).items()}
            with self.state["lock"]:
                if self.state.get("job_running"):
                    body = compact({"ok": False, "error": "a scoring job is already running"}).encode()
                    status = 409
                else:
                    self.state["job_running"] = True
                    self.state["progress_history"] = [{"stage": "Starting", "message": "Preparing LLM route selection."}]
                    self.state["progress"] = {"stage": "Starting", "message": "Preparing LLM route selection.", "plot": "", "legend": "", "done": False, "error": False, "history": list(self.state["progress_history"])}
                    body = compact({"ok": True}).encode()
                    status = 202
            if status == 202:
                threading.Thread(target=run_scoring_job, args=(self.state, self.args, form), daemon=True).start()
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.end_headers(); self.wfile.write(body)
            return
        if self.path == "/api/transcribe":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("audio recording is empty")
                if length > MAX_AUDIO_BYTES:
                    raise ValueError("audio recording is too large")
                mime_type, extension = audio_file_metadata(self.headers.get("Content-Type", ""))
                if not os.getenv("OPENAI_API_KEY"):
                    raise RuntimeError("OPENAI_API_KEY is not set in the verifier process")
                from openai import OpenAI
                response = OpenAI().audio.transcriptions.create(
                    file=(f"calibration-prompt.{extension}", self.rfile.read(length), mime_type),
                    model=self.args.transcription_model,
                    response_format="json",
                )
                text = str(response.text).strip()
                if not text:
                    raise RuntimeError("no speech was detected")
                body = compact({"ok": True, "text": text}).encode()
                self.send_response(200)
            except ValueError as error:
                body = compact({"ok": False, "error": str(error)}).encode()
                self.send_response(400)
            except Exception as error:
                body = compact({"ok": False, "error": str(error)}).encode()
                self.send_response(503)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers(); self.wfile.write(body)
            return
        length = int(self.headers.get("Content-Length", "0"))
        form = {key: values[-1] for key, values in parse_qs(self.rfile.read(length).decode()).items()}
        if self.path == "/reject":
            self.state.update({"routes": [], "route_data": [], "result": None, "results": {}})
            for key in ("goal", "goal_accepted", "goal_relations", "goal_relation_results", "intent"):
                self.state.pop(key, None)
            body = html_page(self.state, self.args, "Proposed goal rejected. Enter a new L1 instruction to propose another goal.")
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body.encode())
            return
        if self.path == "/score":
            with self.state["lock"]:
                if self.state.get("job_running"):
                    message = "Error: a scoring job is already running"
                else:
                    self.state["job_running"] = True
                    self.state["progress_history"] = []
                    message = ""
            if not message:
                run_scoring_job(self.state, self.args, form)
                with self.state["lock"]:
                    message = self.state["progress"]["message"]
                    if self.state["progress"].get("error"):
                        message = f"Error: {message}"
            body = html_page(self.state, self.args, message)
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body.encode())
            return
        try:
            row_number = int(form.get("row", self.state["row_number"]))
            capture_id, rows = choose_capture(self.state["captures"], row_number)
            gt, perceived, start, workspace = scene_setup(rows, self.args)
            mode = form.get("mode", "env_uncertainty")
            planning = perceived if mode == "env_uncertainty" else gt
            if self.path in ("/propose", "/prepare"):
                intent, relations, relation_results, goal = derive_goal(
                    form.get("l1", ""), perceived, gt, start, workspace,
                    self.state["intent_parser"], self.args,
                )
                self.state.update({"row_number": row_number, "capture_id": capture_id, "rows": rows, "mode": mode, "l1": form.get("l1", ""), "intent": intent.model_dump(), "goal_relations": relations, "goal_relation_results": relation_results, "goal": goal, "goal_accepted": False, "start": start, "ground_truth_obstacles": gt, "perceived_obstacles": perceived, "routes": [], "route_data": [], "result": None, "results": {}})
                message = f"Goal proposed at ({goal['x']:.2f}, {goal['y']:.2f}, {goal['z']:.2f}). Accept it or enter a manual goal before generating HRRT routes."
            elif self.path == "/generate":
                if not self.state.get("goal"):
                    raise ValueError("propose a goal first")
                goal = dict(self.state["goal"])
                if form.get("decision") == "manual":
                    goal = {"x": float(form["goal_x"]), "y": float(form["goal_y"]), "z": float(form.get("goal_z", self.args.fixed_z))}
                if not (workspace["x"][0] <= goal["x"] <= workspace["x"][1] and workspace["y"][0] <= goal["y"] <= workspace["y"][1]):
                    raise ValueError("manual goal is outside the planning workspace")
                family = plan_family(start, goal, gt, workspace, self.args)
                self.state.update({"row_number": row_number, "capture_id": capture_id, "rows": rows, "mode": mode, "goal": goal, "goal_accepted": True, "start": start, "ground_truth_obstacles": gt, "perceived_obstacles": perceived, "routes": route_cards(family["routes"]), "route_data": family["routes"], "result": None, "results": {}})
                message = f"Goal accepted at ({goal['x']:.2f}, {goal['y']:.2f}, {goal['z']:.2f}); generated {len(family['routes'])} HRRT routes. Enter L2 for LLM route selection."
            else:
                raise ValueError("unknown action")
        except Exception as exc:
            message = f"Error: {exc}"
            self.state.update({"routes": [], "route_data": [], "result": None, "results": {}})
            if "row_number" in locals():
                self.state["row_number"] = row_number
            if "capture_id" in locals():
                self.state["capture_id"] = capture_id
            if "rows" in locals():
                self.state["rows"] = rows
            for key in ("goal", "start", "ground_truth_obstacles", "perceived_obstacles", "intent", "goal_relations", "goal_relation_results"):
                self.state.pop(key, None)
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
    parser.add_argument("--intent-provider", choices=("openai", "mock"), default="openai")
    parser.add_argument("--intent-model", default="gpt-5.4-nano")
    parser.add_argument("--expert-model", default="gpt-6.1-sol")
    parser.add_argument("--route-nominal-speed-mps", type=float, default=0.5)
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    parser.add_argument("--llama-model", default="hrrt_planner")
    parser.add_argument("--transcription-model", default="gpt-4o-mini-transcribe")
    parser.add_argument("--start-x", type=float, default=-3.5); parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--fixed-z", type=float, default=-0.5)
    parser.add_argument("--workspace-x-min", type=float, default=-3.0); parser.add_argument("--workspace-x-max", type=float, default=3.0)
    parser.add_argument("--workspace-y-min", type=float, default=-3.0); parser.add_argument("--workspace-y-max", type=float, default=3.0)
    parser.add_argument("--clearance-m", type=float, default=0.40); parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--default-standoff-m", type=float, default=0.60)
    parser.add_argument("--goal-sample-resolution-m", type=float, default=0.10)
    parser.add_argument("--default-goal-range-half-width-m", type=float, default=0.15)
    parser.add_argument("--exact-goal-distance-tolerance-m", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=17); parser.add_argument("--hrrt-iterations", type=int, default=300)
    parser.add_argument("--max-candidates", type=int, default=8); parser.add_argument("--max-waypoints", type=int, default=8)
    return parser


def main():
    args = build_parser().parse_args()
    captures = load_captures(args.dataset)
    choose_capture(captures, args.row)
    intent_parser = MockIntentParser() if args.intent_provider == "mock" else OpenAIIntentParser(args.intent_model)
    route_selector = MockRouteSelector() if args.intent_provider == "mock" else OpenAIRouteSelector(args.expert_model)
    state = {"captures": captures, "row_number": args.row, "routes": [], "route_data": [], "result": None, "results": {}, "intent_parser": intent_parser, "route_selector": route_selector, "lock": threading.Lock(), "job_running": False, "progress": {"stage": "Idle", "message": "Generate an HRRT route family first.", "plot": "", "legend": "", "done": True, "error": False}}
    GatewayHandler.state = state; GatewayHandler.args = args
    server = ThreadingHTTPServer((args.host, args.port), GatewayHandler)
    print(f"Calibration verifier listening at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
