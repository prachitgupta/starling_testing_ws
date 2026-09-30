#!/usr/bin/env python3
"""Serve a local browser UI for human review of HRRT teacher selections."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Dict, List, Mapping
from urllib.parse import urlparse


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>HRRT label review</title>
<style>
body{margin:0;background:#07111d;color:#dbeafe;font:15px system-ui}main{display:grid;grid-template-columns:minmax(330px,430px) 1fr;gap:18px;padding:18px;min-height:calc(100vh - 36px)}
.panel{background:#0b1b2b;border:1px solid #1d4965;border-radius:14px;padding:16px}.muted{color:#8fb3c9}button,select,textarea{font:inherit}button{background:#0ea5e9;color:#03131e;border:0;border-radius:8px;padding:10px 14px;font-weight:700;margin:4px}button.secondary{background:#334155;color:white}button.warn{background:#f59e0b;color:#1f1300}button.danger{background:#ef4444;color:white}select,textarea{width:100%;box-sizing:border-box;background:#06121e;color:#dbeafe;border:1px solid #31546c;border-radius:8px;padding:9px}textarea{height:70px}svg{width:100%;height:600px;background:#f8fafc;border-radius:10px}table{width:100%;border-collapse:collapse;font-size:13px}td,th{border-bottom:1px solid #244156;padding:6px;text-align:left}.teacher{outline:4px solid #facc15}
</style></head><body><main><section class="panel"><h2>Human label audit</h2><div id="progress" class="muted"></div><h3 id="pref"></h3><p id="teacher"></p><label>Human-selected route</label><select id="route"></select><label>Notes</label><textarea id="notes"></textarea><div><button onclick="submitReview('accepted')">Accept teacher</button><button class="secondary" onclick="submitReview('corrected')">Use selected route</button><button class="warn" onclick="submitReview('ambiguous')">Mark ambiguous</button></div><h3>Route cards</h3><div id="cards"></div></section><section class="panel"><svg id="plot" viewBox="0 0 900 600"></svg></section></main>
<script>
let current=null;
const ns='http://www.w3.org/2000/svg';
function el(name,attrs){const n=document.createElementNS(ns,name);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);return n}
function renderPlot(row){const svg=document.getElementById('plot');svg.innerHTML='';const w=row.environment.workspace;const sx=x=>50+(x-w.x[0])/(w.x[1]-w.x[0])*800;const sy=y=>550-(y-w.y[0])/(w.y[1]-w.y[0])*500;
 for(const o of row.environment.obstacles){const r=el('rect',{x:sx(o.min_corner[0]),y:sy(o.max_corner[1]),width:sx(o.max_corner[0])-sx(o.min_corner[0]),height:sy(o.min_corner[1])-sy(o.max_corner[1]),fill:'#64748b',stroke:'#0f172a','stroke-width':2});svg.appendChild(r);const t=el('text',{x:(sx(o.min_corner[0])+sx(o.max_corner[0]))/2,y:(sy(o.min_corner[1])+sy(o.max_corner[1]))/2,fill:'white','text-anchor':'middle'});t.textContent=o.label;svg.appendChild(t)}
 for(const route of row.routes){const pts=route.waypoints.map(p=>`${sx(p.x)},${sy(p.y)}`).join(' ');svg.appendChild(el('polyline',{points:pts,fill:'none',stroke:route.color,'stroke-width':route.route_id===row.teacher_route_id?7:3,opacity:route.route_id===row.teacher_route_id?1:.72}));const p=route.waypoints[Math.floor(route.waypoints.length/2)];const t=el('text',{x:sx(p.x),y:sy(p.y)-7,fill:route.color,'font-weight':'bold'});t.textContent=route.route_id;svg.appendChild(t)}
 const s=row.environment.start,g=row.environment.goal;svg.appendChild(el('circle',{cx:sx(s.x),cy:sy(s.y),r:7,fill:'#111827'}));svg.appendChild(el('circle',{cx:sx(g.x),cy:sy(g.y),r:8,fill:'#dc2626'}));}
async function loadNext(){const r=await fetch('/api/next');const data=await r.json();if(data.done){document.getElementById('pref').textContent='Review complete';document.getElementById('progress').textContent=`${data.reviewed}/${data.total}`;current=null;return}current=data.row;document.getElementById('progress').textContent=`${data.reviewed}/${data.total} reviewed · ${data.assignment.review_split}`;document.getElementById('pref').textContent=current.preference.text;document.getElementById('teacher').textContent=`Teacher selected ${current.teacher_route_id}: ${current.teacher_reason}`;const select=document.getElementById('route');select.innerHTML=current.routes.map(r=>`<option value="${r.route_id}" ${r.route_id===current.teacher_route_id?'selected':''}>${r.route_id}</option>`).join('');document.getElementById('notes').value='';document.getElementById('cards').innerHTML='<table><tr><th>ID</th><th>Length</th><th>Duration</th><th>Min clearance</th></tr>'+current.route_cards.map(c=>`<tr><td>${c.route_id}</td><td>${c.path_length_m.toFixed(2)} m</td><td>${c.estimated_duration_s.toFixed(2)} s</td><td>${c.overall_minimum_clearance_m.toFixed(2)} m</td></tr>`).join('')+'</table>';renderPlot(current)}
async function submitReview(decision){if(!current)return;let route=document.getElementById('route').value;if(decision==='accepted')route=current.teacher_route_id;if(decision==='ambiguous')route='';await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sample_id:current.sample_id,decision,human_route_id:route,notes:document.getElementById('notes').value})});await loadNext()}
loadNext();
</script></body></html>"""


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_reviews(path: Path, reviews: Mapping[str, Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for sample_id in sorted(reviews):
            stream.write(json.dumps(reviews[sample_id], separators=(",", ":"), sort_keys=True) + "\n")
    temporary.replace(path)


class ReviewState:
    def __init__(self, raw_path: Path, manifest_path: Path, reviews_path: Path):
        self.rows = {str(row["sample_id"]): row for row in read_jsonl(raw_path)}
        self.assignments = read_jsonl(manifest_path)
        missing = [item["sample_id"] for item in self.assignments if str(item["sample_id"]) not in self.rows]
        if missing:
            raise ValueError(f"Audit manifest references missing samples: {missing[:5]}")
        self.reviews_path = reviews_path
        self.reviews = {str(row["sample_id"]): row for row in read_jsonl(reviews_path)}

    def next_payload(self) -> Dict[str, object]:
        for assignment in self.assignments:
            sample_id = str(assignment["sample_id"])
            if sample_id not in self.reviews:
                return {
                    "done": False,
                    "reviewed": len(self.reviews),
                    "total": len(self.assignments),
                    "assignment": assignment,
                    "row": self.rows[sample_id],
                }
        return {"done": True, "reviewed": len(self.reviews), "total": len(self.assignments)}

    def submit(self, payload: Mapping[str, object]) -> None:
        sample_id = str(payload.get("sample_id", ""))
        if sample_id not in self.rows:
            raise ValueError("Unknown sample_id")
        decision = str(payload.get("decision", ""))
        if decision not in ("accepted", "corrected", "ambiguous"):
            raise ValueError("decision must be accepted, corrected, or ambiguous")
        human_route_id = str(payload.get("human_route_id", ""))
        valid_ids = {str(route["route_id"]) for route in self.rows[sample_id]["routes"]}
        if decision != "ambiguous" and human_route_id not in valid_ids:
            raise ValueError("human_route_id is not one of the displayed candidates")
        assignment = next(item for item in self.assignments if str(item["sample_id"]) == sample_id)
        self.reviews[sample_id] = {
            "schema_version": "hrrt_human_review_v1",
            "sample_id": sample_id,
            "scene_id": self.rows[sample_id]["scene_id"],
            "review_split": assignment["review_split"],
            "decision": decision,
            "human_route_id": human_route_id or None,
            "notes": str(payload.get("notes", "")),
        }
        write_reviews(self.reviews_path, self.reviews)


def handler_factory(state: ReviewState):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, payload: Mapping[str, object], status: int = 200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if urlparse(self.path).path == "/api/next":
                self.send_json(state.next_payload())
                return
            body = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if urlparse(self.path).path != "/api/review":
                self.send_json({"error": "not found"}, 404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(size))
                state.submit(payload)
                self.send_json({"ok": True})
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)

        def log_message(self, format, *args):
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("fine_tuning/datasets/hrrt_teacher_raw.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("fine_tuning/datasets/hrrt_human_audit.jsonl"))
    parser.add_argument("--reviews", type=Path, default=Path("fine_tuning/datasets/hrrt_human_reviews.jsonl"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    state = ReviewState(args.raw, args.audit, args.reviews)
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(state))
    print(f"Human review UI: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
