#!/usr/bin/env python3
"""Serve a local browser UI for human review of HRRT teacher selections."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from typing import Dict, List, Mapping
from urllib.parse import parse_qs, urlparse


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>HRRT route audit</title>
<style>
:root{color-scheme:dark;--ink:#eef6ff;--muted:#91a8bd;--panel:#0d1b2a;--panel2:#112438;--line:#29455d;--accent:#48cae4;--teacher:#ffd166;--safe:#42d392;--warn:#ffb703;--danger:#ff5d73;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 70% -10%,#173b54 0,transparent 35%),#07111d;color:var(--ink);min-height:100vh}button,select,textarea,input{font:inherit}button{cursor:pointer}.shell{max-width:1500px;margin:auto;padding:18px}.topbar{display:flex;align-items:center;gap:14px;margin:0 0 14px}.brand{display:flex;align-items:center;gap:11px}.brand-mark{display:grid;place-items:center;width:40px;height:40px;border-radius:13px;background:linear-gradient(135deg,#48cae4,#4361ee);font-size:21px;box-shadow:0 8px 24px #0096c755}.brand h1{font-size:18px;margin:0}.brand p{font-size:12px;color:var(--muted);margin:2px 0 0}.progress-wrap{margin-left:auto;min-width:220px}.progress-copy{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-bottom:5px}.progress-track{height:7px;background:#152b3e;border-radius:99px;overflow:hidden}.progress-bar{height:100%;width:0;background:linear-gradient(90deg,#48cae4,#42d392);border-radius:inherit;transition:width .3s ease}.layout{display:grid;grid-template-columns:minmax(350px,440px) minmax(0,1fr);gap:16px;align-items:start}.panel{background:linear-gradient(180deg,#0e2031,#0a1826);border:1px solid var(--line);border-radius:18px;box-shadow:0 12px 42px #02091455}.review{padding:18px;max-height:calc(100vh - 100px);overflow:auto}.map-panel{padding:13px;position:sticky;top:14px}.eyebrow{display:flex;gap:8px;align-items:center;color:var(--accent);font-size:11px;font-weight:800;letter-spacing:.09em;text-transform:uppercase}.badge{display:inline-flex;align-items:center;border:1px solid #315773;border-radius:99px;padding:4px 8px;background:#0b1825;color:#b7cede;font-size:11px;font-weight:700}.preference{font-size:22px;line-height:1.28;margin:10px 0 16px}.reason-card{border:1px solid var(--line);background:#091725;border-radius:13px;padding:13px;margin:10px 0}.reason-card.teacher{border-color:#8c7628;background:linear-gradient(135deg,#211d10,#101b25)}.reason-title{display:flex;align-items:center;gap:7px;font-weight:800;font-size:13px;margin-bottom:7px}.reason-text{font-size:13px;line-height:1.5;color:#c7d8e7;margin:0}.teacher-dot{width:9px;height:9px;border-radius:50%;background:var(--teacher);box-shadow:0 0 0 4px #ffd16622}.section-label{display:block;font-size:12px;font-weight:800;color:#cbe3f5;margin:16px 0 7px}.route-select{width:100%;border:1px solid #3c617d;border-radius:11px;background:#071522;color:var(--ink);padding:11px 12px;font-weight:700}.route-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:9px}.route-card{position:relative;text-align:left;background:#0a1724;border:1px solid var(--line);color:var(--ink);border-radius:12px;padding:10px;margin:0;min-height:84px;transition:.15s ease}.route-card:hover{transform:translateY(-1px);border-color:#4e7c9e}.route-card.selected{border-color:var(--route-color);box-shadow:inset 4px 0 0 var(--route-color),0 0 0 1px var(--route-color)}.route-card.teacher-route:after{content:"Teacher";position:absolute;top:8px;right:8px;color:#231c00;background:var(--teacher);border-radius:99px;padding:2px 6px;font-size:9px;font-weight:900;text-transform:uppercase}.route-id{display:flex;align-items:center;gap:7px;font-weight:900}.swatch{width:10px;height:10px;border-radius:50%;background:var(--route-color)}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:3px;margin-top:9px;color:var(--muted);font-size:10px}.metrics strong{display:block;color:#dfeffc;font-size:12px}.notes{width:100%;min-height:104px;resize:vertical;background:#071522;color:var(--ink);border:1px solid #365a74;border-radius:11px;padding:11px;line-height:1.45}.hint{color:var(--muted);font-size:11px;line-height:1.4;margin:6px 0 0}.actions{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:13px}.action{border:0;border-radius:11px;min-height:46px;padding:10px 12px;font-weight:850;margin:0}.accept{grid-column:1/-1;background:linear-gradient(135deg,#49d49d,#20b878);color:#032417}.correct{background:#2c76db;color:white}.ambiguous{background:#4a3521;color:#ffdba0;border:1px solid #855a29}.action:disabled{opacity:.55;cursor:wait}.map-head{display:flex;align-items:center;gap:10px;padding:3px 3px 11px}.map-head h2{font-size:15px;margin:0}.map-head .badge{margin-left:auto}.map-tools{display:flex;gap:6px}.tool{border:1px solid #31536c;border-radius:8px;background:#0a1825;color:#cde2f2;padding:6px 9px;font-size:11px;font-weight:750}.tool.active{background:#153b54;border-color:#48cae4;color:white}.canvas{position:relative;border-radius:13px;overflow:hidden;background:#eef6fb;box-shadow:inset 0 0 0 1px #ffffff}svg{display:block;width:100%;height:auto;max-height:calc(100vh - 190px);min-height:480px;touch-action:manipulation}.map-legend{display:flex;flex-wrap:wrap;gap:12px;padding:10px 3px 1px;color:#a9bfd0;font-size:11px}.legend-item{display:flex;align-items:center;gap:5px}.legend-symbol{width:12px;height:12px;border-radius:3px;background:#334155}.legend-symbol.clearance{background:#ff5d7322;border:1px dashed #e25568}.legend-symbol.route{height:4px;border-radius:99px;background:#48cae4}.legend-symbol.teacher{border-radius:50%;background:var(--teacher)}.selected-readout{position:absolute;left:13px;top:13px;background:#061521e8;border:1px solid #345c78;border-radius:10px;padding:8px 10px;color:white;font-size:11px;backdrop-filter:blur(8px);pointer-events:none}.selected-readout strong{display:block;font-size:13px;margin-bottom:2px}.toast{position:fixed;left:50%;bottom:22px;transform:translate(-50%,120px);background:#e8fff6;color:#053b29;border-radius:10px;padding:11px 16px;font-weight:800;box-shadow:0 10px 35px #0008;transition:.25s ease;z-index:20}.toast.show{transform:translate(-50%,0)}.empty{display:none;text-align:center;padding:70px 20px}.empty-icon{font-size:44px}.loading{opacity:.55;pointer-events:none}
@media(max-width:900px){.shell{padding:10px}.topbar{padding:4px}.progress-wrap{min-width:140px}.layout{grid-template-columns:1fr}.map-panel{position:static;order:-1}.review{max-height:none}.map-panel{padding:9px}svg{min-height:0;max-height:none;aspect-ratio:1.35}.map-head{padding:5px 5px 10px}.preference{font-size:19px}.review{padding:14px}}
@media(max-width:520px){.brand p{display:none}.brand-mark{width:35px;height:35px}.progress-wrap{min-width:110px}.progress-copy span:first-child{display:none}.shell{padding:7px}.layout{gap:9px}.route-grid{grid-template-columns:1fr}.map-head{flex-wrap:wrap}.map-tools{order:3;width:100%}.map-tools .tool{flex:1;min-height:38px}.map-legend{gap:8px}.review{border-radius:14px}.actions{position:sticky;bottom:0;background:linear-gradient(transparent,#0a1826 18%);padding-top:18px}.selected-readout{font-size:10px}}
</style></head><body><div class="shell">
<header class="topbar"><div class="brand"><div class="brand-mark">⌁</div><div><h1>HRRT route audit</h1><p>Human verification of teacher trajectory choices</p></div></div><div class="progress-wrap"><div class="progress-copy"><span>Audit progress</span><span id="progress">Loading…</span></div><div class="progress-track"><div id="progressBar" class="progress-bar"></div></div></div></header>
<main id="workspace" class="layout">
<section class="panel review"><div class="eyebrow"><span id="split" class="badge">—</span><span id="sampleId">Loading sample</span></div><h2 id="pref" class="preference"></h2>
<div class="reason-card teacher"><div class="reason-title"><span class="teacher-dot"></span><span id="teacherTitle">Teacher selection</span></div><p id="teacherReason" class="reason-text"></p></div>
<div class="reason-card"><div class="reason-title">Llama-facing explanation</div><p id="studentReason" class="reason-text"></p></div>
<label class="section-label" for="route">Your route choice</label><select id="route" class="route-select" aria-label="Human-selected route"></select><div id="cards" class="route-grid"></div>
<label class="section-label" for="notes">Human reasoning / notes</label><textarea id="notes" class="notes" placeholder="Preference: …  Geometry: …  Decision: …  Safety: …"></textarea><p class="hint">Required when changing the teacher route. Mention the preference, route geometry, decision, and safety evidence.</p>
<div class="actions"><button class="action accept" data-decision="accepted">✓ Accept teacher selection</button><button class="action correct" data-decision="corrected">Use highlighted route</button><button class="action ambiguous" data-decision="ambiguous">Flag ambiguous</button></div></section>
<section class="panel map-panel"><div class="map-head"><h2>Route workspace</h2><div class="map-tools"><button id="focusToggle" class="tool active">View all routes</button><button id="clearanceToggle" class="tool active">Safety margins</button></div><span class="badge">Direction shown →</span></div><div class="canvas"><svg id="plot" viewBox="0 0 900 640" role="img" aria-label="Candidate route map"></svg><div id="selectedReadout" class="selected-readout"></div></div><div class="map-legend"><span class="legend-item"><i class="legend-symbol teacher"></i>teacher choice</span><span class="legend-item"><i class="legend-symbol route"></i>selected route</span><span class="legend-item"><i class="legend-symbol clearance"></i>0.40 m safety margin</span><span class="legend-item"><i class="legend-symbol"></i>obstacle</span></div></section></main>
<section id="empty" class="panel empty"><div class="empty-icon">✓</div><h2>Review complete</h2><p id="emptyCopy" class="muted"></p></section></div><div id="toast" class="toast" role="status"></div>
<script>
let current=null,showAll=false,showClearance=true,busy=false;
const reviewerId=localStorage.getItem('hrrt-reviewer-id')||(crypto.randomUUID?.()||`${Date.now()}-${Math.random()}`);localStorage.setItem('hrrt-reviewer-id',reviewerId);
const ns='http://www.w3.org/2000/svg',clearanceM=.40;
const $=id=>document.getElementById(id);
function el(name,attrs={},text=''){const n=document.createElementNS(ns,name);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text)n.textContent=text;return n}
function escapeHtml(value){return String(value??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
function metric(value,digits=2){return Number.isFinite(Number(value))?Number(value).toFixed(digits):'—'}
function routeCard(row,id){return (row.route_cards||[]).find(card=>card.route_id===id)||{}}
function showToast(message){$('toast').textContent=message;$('toast').classList.add('show');setTimeout(()=>$('toast').classList.remove('show'),1600)}
function selectedRouteId(){return $('route').value||(current?.teacher_route_id||'')}
function svgLabel(svg,x,y,text,fill='#0f172a',anchor='middle',size=12,weight=700){svg.appendChild(el('text',{x,y,fill,'text-anchor':anchor,'font-size':size,'font-weight':weight,'paint-order':'stroke','stroke':'#eef6fb','stroke-width':4,'stroke-linejoin':'round'},text))}
function renderPlot(row){
 const svg=$('plot');svg.innerHTML='';const w=row.environment.workspace,xmin=w.x[0],xmax=w.x[1],ymin=w.y[0],ymax=w.y[1],pad=64,W=900,H=640;
 const sx=x=>pad+(x-xmin)/(xmax-xmin)*(W-2*pad),sy=y=>H-pad-(y-ymin)/(ymax-ymin)*(H-2*pad),scaleX=(W-2*pad)/(xmax-xmin),scaleY=(H-2*pad)/(ymax-ymin),selected=selectedRouteId();
 const defs=el('defs');const grid=el('pattern',{id:'grid',width:Math.max(38,(W-2*pad)/8),height:Math.max(38,(H-2*pad)/6),patternUnits:'userSpaceOnUse'});grid.appendChild(el('path',{d:'M 100 0 L 0 0 0 100',fill:'none',stroke:'#b9cfdb','stroke-width':1,opacity:.5}));defs.appendChild(grid);
 for(const route of row.routes){const marker=el('marker',{id:`arrow-${route.route_id}`,viewBox:'0 0 10 10',refX:8,refY:5,markerWidth:7,markerHeight:7,orient:'auto-start-reverse'});marker.appendChild(el('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:route.color}));defs.appendChild(marker)}svg.appendChild(defs);
 svg.appendChild(el('rect',{x:pad,y:pad,width:W-2*pad,height:H-2*pad,rx:6,fill:'#f1f7fa'}));svg.appendChild(el('rect',{x:pad,y:pad,width:W-2*pad,height:H-2*pad,fill:'url(#grid)'}));
 for(let i=0;i<=4;i++){const x=xmin+(xmax-xmin)*i/4,y=ymin+(ymax-ymin)*i/4;svgLabel(svg,sx(x),H-28,metric(x,1),'#597283','middle',11,600);svgLabel(svg,38,sy(y)+4,metric(y,1),'#597283','middle',11,600)}
 for(const o of row.environment.obstacles){const x1=sx(o.min_corner[0]),x2=sx(o.max_corner[0]),y1=sy(o.max_corner[1]),y2=sy(o.min_corner[1]);if(showClearance)svg.appendChild(el('rect',{x:x1-clearanceM*scaleX,y:y1-clearanceM*scaleY,width:x2-x1+2*clearanceM*scaleX,height:y2-y1+2*clearanceM*scaleY,rx:9,fill:'#ff5d7314',stroke:'#dc5668','stroke-width':2,'stroke-dasharray':'8 7'}));svg.appendChild(el('rect',{x:x1,y:y1,width:x2-x1,height:y2-y1,rx:5,fill:'#344b5e',stroke:'#182b39','stroke-width':2}));svgLabel(svg,(x1+x2)/2,(y1+y2)/2+4,o.label.replaceAll('_',' '),'#fff','middle',12,800)}
 const ordered=[...row.routes].sort((a,b)=>(a.route_id===selected)-(b.route_id===selected));
 for(const route of ordered){const isSelected=route.route_id===selected,isTeacher=route.route_id===row.teacher_route_id;if(!showAll&&!isSelected)continue;const pts=route.waypoints.map(p=>`${sx(p.x)},${sy(p.y)}`).join(' ');if(isSelected)svg.appendChild(el('polyline',{points:pts,fill:'none',stroke:'#fff','stroke-width':14,'stroke-linejoin':'round','stroke-linecap':'round',opacity:.9}));svg.appendChild(el('polyline',{points:pts,fill:'none',stroke:route.color,'stroke-width':isSelected?8:3,'stroke-linejoin':'round','stroke-linecap':'round',opacity:isSelected?1:.38,'marker-mid':isSelected?`url(#arrow-${route.route_id})`:'','marker-end':isSelected?`url(#arrow-${route.route_id})`:''}));if(isSelected){route.waypoints.forEach((p,index)=>{if(index===0||index===route.waypoints.length-1)return;svg.appendChild(el('circle',{cx:sx(p.x),cy:sy(p.y),r:11,fill:route.color,stroke:'#fff','stroke-width':3}));svgLabel(svg,sx(p.x),sy(p.y)+4,String(index),'#fff','middle',11,900)});const p=route.waypoints[Math.floor(route.waypoints.length/2)];svgLabel(svg,sx(p.x),sy(p.y)-22,`${route.route_id}${isTeacher?' · teacher':''}`,route.color,'middle',15,900)}}
 const s=row.environment.start,g=row.environment.goal;svg.appendChild(el('circle',{cx:sx(s.x),cy:sy(s.y),r:13,fill:'#152536',stroke:'#fff','stroke-width':4}));svg.appendChild(el('circle',{cx:sx(g.x),cy:sy(g.y),r:14,fill:'#ef476f',stroke:'#fff','stroke-width':4}));svgLabel(svg,sx(s.x),sy(s.y)-22,'START','#14283a','middle',12,900);svgLabel(svg,sx(g.x),sy(g.y)-23,'GOAL','#c7254e','middle',12,900);
 const card=routeCard(row,selected);$('selectedReadout').innerHTML=`<strong>${escapeHtml(selected)}${selected===row.teacher_route_id?' · teacher':''}</strong>${metric(card.estimated_duration_s)} s · ${metric(card.path_length_m)} m · ${metric(card.overall_minimum_clearance_m)} m clearance`;
}
function renderCards(row){const selected=selectedRouteId();$('cards').innerHTML=row.routes.map(route=>{const c=routeCard(row,route.route_id);return `<button class="route-card ${route.route_id===selected?'selected':''} ${route.route_id===row.teacher_route_id?'teacher-route':''}" style="--route-color:${escapeHtml(route.color)}" data-route="${escapeHtml(route.route_id)}"><span class="route-id"><i class="swatch"></i>${escapeHtml(route.route_id)}</span><span class="metrics"><span><strong>${metric(c.estimated_duration_s)}</strong>sec</span><span><strong>${metric(c.path_length_m)}</strong>meters</span><span><strong>${metric(c.overall_minimum_clearance_m)}</strong>clearance</span></span></button>`}).join('');document.querySelectorAll('[data-route]').forEach(card=>card.onclick=()=>selectRoute(card.dataset.route))}
function selectRoute(id){$('route').value=id;renderCards(current);renderPlot(current)}
function setBusy(value){busy=value;$('workspace').classList.toggle('loading',value);document.querySelectorAll('.action').forEach(button=>button.disabled=value)}
async function loadNext(){setBusy(true);try{const response=await fetch(`/api/next?reviewer=${encodeURIComponent(reviewerId)}`,{cache:'no-store'});if(!response.ok)throw new Error('Could not load the next audit sample.');const data=await response.json();$('progress').textContent=`${data.reviewed}/${data.total}`;$('progressBar').style.width=`${data.total?100*data.reviewed/data.total:100}%`;if(data.done){$('workspace').style.display='none';$('empty').style.display='block';$('emptyCopy').textContent=`All ${data.total} assigned samples have a saved decision.`;current=null;return}if(data.waiting){throw new Error('All remaining samples are currently open with other reviewers. Try again shortly.')}current=data.row;$('split').textContent=data.assignment.review_split;$('sampleId').textContent=current.sample_id;$('pref').textContent=current.preference.text;$('teacherTitle').textContent=`Teacher selected ${current.teacher_route_id}`;$('teacherReason').textContent=current.teacher_selection_reason||'No audit reason was recorded.';$('studentReason').textContent=current.teacher_trajectory_reasoning||current.teacher_reason||'No Llama-facing explanation was recorded.';const select=$('route');select.innerHTML=current.routes.map(route=>`<option value="${escapeHtml(route.route_id)}" ${route.route_id===current.teacher_route_id?'selected':''}>${escapeHtml(route.route_id)}${route.route_id===current.teacher_route_id?' — teacher choice':''}</option>`).join('');$('notes').value='';renderCards(current);renderPlot(current)}catch(error){showToast(error.message)}finally{setBusy(false)}}
async function submitReview(decision){if(!current||busy)return;let route=selectedRouteId(),reasoning=$('notes').value.trim();if(decision==='corrected'&&route===current.teacher_route_id){showToast('Choose a different route before correcting.');return}if(decision==='corrected'&&!reasoning){$('notes').focus();showToast('Add your reasoning before submitting a correction.');return}if(decision==='accepted')route=current.teacher_route_id;if(decision==='ambiguous')route='';setBusy(true);try{const response=await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sample_id:current.sample_id,reviewer_id:reviewerId,decision,human_route_id:route,human_reason:reasoning,notes:reasoning})});const result=await response.json();if(!response.ok)throw new Error(result.error||'Review could not be saved.');showToast('Review saved');await loadNext()}catch(error){showToast(error.message);setBusy(false)}}
$('route').addEventListener('change',event=>selectRoute(event.target.value));$('focusToggle').onclick=()=>{showAll=!showAll;$('focusToggle').classList.toggle('active',showAll);$('focusToggle').textContent=showAll?'Focus selected route':'View all routes';renderPlot(current)};$('clearanceToggle').onclick=()=>{showClearance=!showClearance;$('clearanceToggle').classList.toggle('active',showClearance);renderPlot(current)};document.querySelectorAll('[data-decision]').forEach(button=>button.onclick=()=>submitReview(button.dataset.decision));document.addEventListener('keydown',event=>{if(event.target.matches('textarea,input,select')||!current)return;if(event.key==='a')submitReview('accepted');if(event.key==='c')submitReview('corrected')});loadNext();
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
    CLAIM_TTL_SECONDS = 30 * 60

    def __init__(self, raw_path: Path, manifest_path: Path, reviews_path: Path):
        self.rows = {str(row["sample_id"]): row for row in read_jsonl(raw_path)}
        self.assignments = read_jsonl(manifest_path)
        missing = [item["sample_id"] for item in self.assignments if str(item["sample_id"]) not in self.rows]
        if missing:
            raise ValueError(f"Audit manifest references missing samples: {missing[:5]}")
        self.reviews_path = reviews_path
        self.reviews = {str(row["sample_id"]): row for row in read_jsonl(reviews_path)}
        self.claims: Dict[str, Dict[str, object]] = {}
        self.lock = threading.Lock()

    def next_payload(self, reviewer_id: str = "") -> Dict[str, object]:
        reviewer_id = reviewer_id.strip() or "single-reviewer"
        with self.lock:
            now = time.time()
            self.claims = {
                sample_id: claim
                for sample_id, claim in self.claims.items()
                if now - float(claim["claimed_at"]) < self.CLAIM_TTL_SECONDS
                and sample_id not in self.reviews
            }
            for assignment in self.assignments:
                sample_id = str(assignment["sample_id"])
                if self.claims.get(sample_id, {}).get("reviewer_id") == reviewer_id:
                    return self._payload(assignment)
            for assignment in self.assignments:
                sample_id = str(assignment["sample_id"])
                if sample_id not in self.reviews and sample_id not in self.claims:
                    self.claims[sample_id] = {"reviewer_id": reviewer_id, "claimed_at": now}
                    return self._payload(assignment)
            if len(self.reviews) < len(self.assignments):
                return {
                    "done": False,
                    "waiting": True,
                    "reviewed": len(self.reviews),
                    "total": len(self.assignments),
                }
            return {"done": True, "reviewed": len(self.reviews), "total": len(self.assignments)}

    def _payload(self, assignment: Mapping[str, object]) -> Dict[str, object]:
        sample_id = str(assignment["sample_id"])
        return {
            "done": False,
            "reviewed": len(self.reviews),
            "total": len(self.assignments),
            "assignment": assignment,
            "row": self.rows[sample_id],
        }

    def submit(self, payload: Mapping[str, object]) -> None:
        sample_id = str(payload.get("sample_id", ""))
        if sample_id not in self.rows:
            raise ValueError("Unknown sample_id")
        decision = str(payload.get("decision", ""))
        if decision not in ("accepted", "corrected", "ambiguous"):
            raise ValueError("decision must be accepted, corrected, or ambiguous")
        human_route_id = str(payload.get("human_route_id", ""))
        human_reason = str(payload.get("human_reason") or payload.get("notes") or "").strip()
        valid_ids = {str(route["route_id"]) for route in self.rows[sample_id]["routes"]}
        if decision != "ambiguous" and human_route_id not in valid_ids:
            raise ValueError("human_route_id is not one of the displayed candidates")
        if decision == "corrected" and not human_reason:
            raise ValueError("human reasoning is required when correcting the route")
        assignment = next(item for item in self.assignments if str(item["sample_id"]) == sample_id)
        reviewer_id = str(payload.get("reviewer_id", "")).strip() or "single-reviewer"
        with self.lock:
            claim = self.claims.get(sample_id)
            if claim and claim["reviewer_id"] != reviewer_id:
                raise ValueError("This sample is currently assigned to another reviewer")
            self.reviews[sample_id] = {
                "schema_version": "hrrt_human_review_v1",
                "sample_id": sample_id,
                "scene_id": self.rows[sample_id]["scene_id"],
                "review_split": assignment["review_split"],
                "decision": decision,
                "human_route_id": human_route_id or None,
                "human_reason": human_reason,
                "notes": human_reason,
            }
            self.claims.pop(sample_id, None)
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
            parsed = urlparse(self.path)
            if parsed.path == "/api/next":
                reviewer_id = parse_qs(parsed.query).get("reviewer", [""])[0]
                self.send_json(state.next_payload(reviewer_id))
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
