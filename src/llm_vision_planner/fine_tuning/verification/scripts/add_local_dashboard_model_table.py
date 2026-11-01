#!/usr/bin/env python3
"""Add a local-only model/loss reference table to an ablation dashboard."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


SECTION_ID = "local-model-glossary-section"
PATCH_VERSION = "2"
SECTION = f"""
<section id="{SECTION_ID}" data-version="{PATCH_VERSION}"><h2>Model and loss definitions</h2>
<p class="muted">Display-only reference generated on this laptop. It does not change training, checkpoints, study state, or NCSA files.</p>
<div id="localModelGlossary"></div>
<h3>Selected Model A versus Model B</h3>
<p>The table below follows the two model selectors in “Same scene, different preferences.”</p>
<div id="localModelComparison"></div></section>
"""

SCRIPT = r"""
<script id="local-model-glossary-script">
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('study-data').textContent);
  const byId = id => document.getElementById(id);
  const definitions = {
    BASE: 'L_BASE = (1/8) Σⱼ₌₁⁸ ‖ŵⱼ − wⱼ‖₂ (mean Euclidean distance over all eight predicted waypoints).',
    E0: 'L_E0 = (1/6) Σⱼ₌₁⁶ ‖ŵⱼ − wⱼ‖₂ (mean Euclidean distance over the six predicted interior waypoints).',
    E1: 'E0 + 0.25 × mean squared interior waypoint error / 1 m.',
    E2: 'E0 + 1 × corridor-side hinge loss.',
    E3: 'E0 + 5 × corridor-side hinge loss.',
    E4: 'E0 + 0.25 × squared-error term + the selected corridor term (1× or 5×).'
  };
  const make = (tag, text, parent) => {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (parent) parent.append(node);
    return node;
  };
  const makeTable = (container, headers, rows) => {
    container.replaceChildren();
    const table = make('table', undefined, container);
    const head = make('tr', undefined, make('thead', undefined, table));
    headers.forEach(header => make('th', header, head));
    const body = make('tbody', undefined, table);
    rows.forEach(row => {
      const tr = make('tr', undefined, body);
      row.forEach(value => make('td', String(value), tr));
    });
  };
  const millions = value => Number.isFinite(value) ? `${(value / 1e6).toFixed(3)} M` : '—';
  const lossVariant = run => {
    if (run === 'baseline') return 'BASE';
    const match = run.match(/_(E[0-4])(?:_|$)/);
    return match ? match[1] : '—';
  };
  const architecture = run => run === 'baseline' ? 'Transformer 1 historical checkpoint' :
    run.startsWith('v1_') ? 'Transformer 1, fixed endpoints' :
    run.startsWith('v2_') ? 'Transformer 2.0, fixed endpoints' : 'Final selected model';
  const endpointTreatment = run => run === 'baseline' ?
    'Model predicts all 8 waypoints' : 'Start + goal fixed; model predicts 6 interior waypoints';

  makeTable(byId('localModelGlossary'), ['Code', 'Training loss definition'],
    Object.entries(definitions).map(([code, definition]) => [code, definition]));

  const renderComparison = () => {
    const rows = [];
    [byId('runA').value, byId('runB').value].forEach((run, index) => {
      const report = data.reports[run] || {};
      const total = Number(report.parameters);
      const planner = Number(report.planner_parameters);
      const text = Number.isFinite(total) && Number.isFinite(planner) ? total - planner : NaN;
      const variant = lossVariant(run);
      rows.push([
        index === 0 ? 'Model A' : 'Model B', run, architecture(run), variant,
        definitions[variant] || 'See the frozen run configuration.', millions(total),
        millions(text), millions(planner), endpointTreatment(run)
      ]);
    });
    makeTable(byId('localModelComparison'),
      ['Selector', 'Run', 'Architecture', 'Loss', 'Loss definition', 'Total parameters',
       'MiniLM encoder', 'Planner parameters', 'Waypoints'], rows);
  };
  byId('runA').addEventListener('change', renderComparison);
  byId('runB').addEventListener('change', renderComparison);
  renderComparison();
})();
</script>
"""


def patch_dashboard(path: Path) -> bool:
    html = path.read_text(encoding="utf-8")
    if f'id="{SECTION_ID}" data-version="{PATCH_VERSION}"' in html:
        return False
    required = ('<script id="study-data"', 'id="runA"', 'id="runB"', '</body>')
    missing = [marker for marker in required if marker not in html]
    if missing:
        raise ValueError(f"{path} is not a compatible ablation dashboard; missing {missing}")

    section_start = f'<section id="{SECTION_ID}"'
    script_start = '<script id="local-model-glossary-script">'
    if section_start in html:
        start = html.index(section_start)
        end = html.index('</section>', start) + len('</section>')
        html = html[:start] + SECTION.strip() + html[end:]
        start = html.index(script_start)
        end = html.index('</script>', start) + len('</script>')
        html = html[:start] + SCRIPT.strip() + html[end:]
    else:
        anchor = '<section><h2>Learning curves</h2>'
        if anchor not in html:
            anchor = '<section><h2>Components and provenance</h2>'
        if anchor not in html:
            raise ValueError(f"{path} has no supported insertion point")
        html = html.replace(anchor, SECTION + anchor, 1)
        html = html.replace('</body>', SCRIPT + '</body>', 1)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(html, encoding="utf-8")
    os.replace(temporary, path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add model/loss definitions to a downloaded ablation dashboard."
    )
    parser.add_argument("dashboard", type=Path, help="Path to dashboard/index.html")
    args = parser.parse_args()
    changed = patch_dashboard(args.dashboard.resolve())
    action = "Updated" if changed else "Already updated"
    print(f"{action}: {args.dashboard.resolve()}")


if __name__ == "__main__":
    main()
