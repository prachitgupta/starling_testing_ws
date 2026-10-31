"""Render a portable, offline interactive dashboard from real study reports."""
import json
import os
from pathlib import Path


def render_dashboard(root):
    root = Path(root)
    def read(path, default):
        return json.loads(path.read_text()) if path.exists() else default
    reports = {p.stem: read(p, {}) for p in sorted((root / 'reports').glob('*.json'))}
    if (root / 'final_test.json').exists():
        reports['final'] = read(root / 'final_test.json', {})
    histories = {p.parent.name: read(p, []) for p in sorted((root / 'runs').glob('*/history.json'))}
    data = {'manifest': read(root / 'manifest.json', {}), 'state': read(root / 'state.json', {}),
            'audit': read(root / 'audit.json', {}), 'selection': read(root / 'selection.json', {}),
            'preflight': {p.stem: read(p, {}) for p in root.glob('preflight_*.json')},
            'reports': reports, 'histories': histories}
    payload = json.dumps(data, allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    template = Path(__file__).with_suffix('.html').read_text(encoding='utf-8')
    output = root / 'dashboard'; output.mkdir(exist_ok=True)
    temporary = output / 'index.html.tmp'
    temporary.write_text(template.replace('__STUDY_DATA__', payload), encoding='utf-8')
    os.replace(temporary, output / 'index.html')
