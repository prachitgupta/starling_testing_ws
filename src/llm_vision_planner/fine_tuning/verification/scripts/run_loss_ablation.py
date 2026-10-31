#!/usr/bin/env python3
"""Resumable, validation-selected loss study on the production Transformers."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2]
VARIANTS = {
    'E0': {'corridor': 0.0, 'squared': 0.0},
    'E1': {'corridor': 0.0, 'squared': 0.25},
    'E2': {'corridor': 1.0, 'squared': 0.0},
    'E3': {'corridor': 5.0, 'squared': 0.0},
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dependency_versions():
    return {name: importlib.metadata.version(name)
            for name in ('torch', 'transformers', 'numpy', 'scipy', 'PyYAML')}


def score(result):
    return result['metrics']['mean_interior_error_m']


def run_id(architecture, variant, seed):
    return f'{architecture}_{variant}_s{seed}'


def plan(manifest, results):
    """Pure decision logic. Receives validation reports only; never reads test."""
    seed = manifest['seeds'][0]
    t2 = run_id('v2', 'E0', seed)
    if 'baseline' not in results:
        return {'id': 'baseline', 'kind': 'baseline'}
    if t2 not in results:
        return {'id': t2, 'kind': 'train', 'architecture': 'v2', 'variant': 'E0',
                'seed': seed, 'weights': VARIANTS['E0']}
    architecture = 'v2' if score(results[t2]) < score(results['baseline']) else 'v1'
    baseline_id = run_id(architecture, 'E0', seed)
    for variant in VARIANTS:
        key = run_id(architecture, variant, seed)
        if key not in results:
            return {'id': key, 'kind': 'train', 'architecture': architecture,
                    'variant': variant, 'seed': seed, 'weights': VARIANTS[variant]}
    variants = dict(VARIANTS)
    baseline_score = score(results[baseline_id])
    best_corridor = min(('E2', 'E3'), key=lambda v: score(results[run_id(architecture, v, seed)]))
    if (score(results[run_id(architecture, 'E1', seed)]) < baseline_score and
            score(results[run_id(architecture, best_corridor, seed)]) < baseline_score):
        variants['E4'] = {'corridor': VARIANTS[best_corridor]['corridor'], 'squared': 0.25}
        key = run_id(architecture, 'E4', seed)
        if key not in results:
            return {'id': key, 'kind': 'train', 'architecture': architecture,
                    'variant': 'E4', 'seed': seed, 'weights': variants['E4']}
    leader = min((v for v in variants if v != 'E0'),
                 key=lambda v: score(results[run_id(architecture, v, seed)]))
    for additional_seed in manifest['seeds'][1:]:
        for variant in ('E0', leader):
            key = run_id(architecture, variant, additional_seed)
            if key not in results:
                return {'id': key, 'kind': 'train', 'architecture': architecture,
                        'variant': variant, 'seed': additional_seed, 'weights': variants[variant]}
    improvements = [score(results[run_id(architecture, 'E0', s)]) -
                    score(results[run_id(architecture, leader, s)]) for s in manifest['seeds']]
    improvement = sum(improvements) / len(improvements)
    supported = (improvement >= manifest['minimum_improvement_m'] and
                 sum(x > 0 for x in improvements) > len(improvements) / 2)
    winner = leader if supported else 'E0'
    # Predeclared representative: median validation seed, rather than luckiest seed.
    ordered = sorted((run_id(architecture, winner, s) for s in manifest['seeds']),
                     key=lambda key: score(results[key]))
    return {'id': 'final', 'kind': 'final', 'architecture': architecture,
            'winner_variant': winner, 'selected_run': ordered[len(ordered) // 2],
            'candidate_variant': leader, 'paired_improvements_m': improvements,
            'mean_improvement_m': improvement,
            'conclusion': 'auxiliary loss supported' if supported else 'no conclusive auxiliary-loss gain; retain E0',
            'architecture_note': 'Single-seed screen against a historical checkpoint, not an isolated causal architecture study.'}


def sources():
    return [PACKAGE / 'fine_tuning/scripts/train_hrrt_waypoint_transformer.py',
            PACKAGE / 'fine_tuning/scripts/train_hrrt_waypoint_transformer_2_0.py',
            HERE / 'verify_transformer_components.py', HERE / 'ablation_support.py',
            HERE / 'run_loss_ablation.py', HERE / 'ablation_dashboard.py',
            HERE / 'ablation_dashboard.html']


def initialize(args):
    import ablation_support as support
    root = args.study_root.resolve()
    if (root / 'manifest.json').exists():
        raise ValueError('Study already initialized. Use advance/resume; do not overwrite its manifest.')
    paths = {s: (args.dataset_root / f'hrrt_sft_{s}.csv').resolve()
             for s in ('train', 'validation', 'test')}
    checkpoint = args.baseline_checkpoint.resolve()
    for path in (*paths.values(), checkpoint / 'model.pt', checkpoint / 'config.json'):
        if not path.is_file():
            raise FileNotFoundError(f'Required artifact missing: {path}')
    if not (checkpoint / 'tokenizer').is_dir() or not (checkpoint.parent / 'text_encoder').is_dir():
        raise ValueError('Copy the baseline tokenizer and sibling text_encoder directory as documented.')
    baseline_config = read(checkpoint / 'config.json')
    if baseline_config['model'].get('architecture_version') == 'transformer_2_0':
        raise ValueError('Baseline must be the Transformer 1 checkpoint from outputs/overfitted.')
    # Reject the earlier accidental dss/ versus dss_teacher_only_80_10_10 mixup.
    for split in ('train', 'validation'):
        historical = Path(baseline_config['data'][f'{split}_csv'])
        if historical.name != paths[split].name or historical.parent.name != paths[split].parent.name:
            raise ValueError(f'Baseline dataset path does not match study dataset: {historical}')
    config = support.v1.load_yaml(PACKAGE / 'fine_tuning/configs/hrrt_waypoint_transformer_2_0.yaml')
    config['data'].update(train_csv=str(paths['train']), validation_csv=str(paths['validation']), row_limit=None)
    config['training']['max_epochs'] = args.max_epochs
    config['validation']['max_safety_samples'] = None
    config['selection_metric'] = 'validation.mean_interior_error_m'
    config['study_endpoint_rule'] = 'fixed start + six predictions + fixed goal for every fresh run'
    # The trainer saved this base MiniLM before fine-tuning. Fine-tuned state lives in model.pt.
    config['model']['text_encoder_name'] = str(checkpoint.parent / 'text_encoder')
    audit = support.verify.dataset_audit(paths)
    if not audit['passed']:
        raise ValueError(f'Dataset audit failed: {audit}')
    if any(report['rows'] != report['waypoint_rows'] for report in audit['splits'].values()):
        raise ValueError('Every study row must have teacher waypoints; refusing silent row filtering.')
    # The legacy audit uses scene IDs. Also check actual geometry and duplicate sample IDs.
    import csv
    geometries = {}; samples = {}
    for split, path in paths.items():
        geometries[split] = set(); samples[split] = set()
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                sid = row['sample_id']
                if sid in samples[split]:
                    raise ValueError(f'Duplicate sample ID in {split}: {sid}')
                samples[split].add(sid)
                env = json.loads(row['environment'])
                geo = {k: env[k] for k in ('workspace', 'start', 'goal', 'obstacles')}
                geo['obstacles'] = sorted(geo['obstacles'], key=lambda x: json.dumps(x, sort_keys=True))
                geometries[split].add(json.dumps(geo, sort_keys=True))
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if geometries[a] & geometries[b] or samples[a] & samples[b]:
            raise ValueError(f'Geometry or sample overlap between {a} and {b}')
    fingerprints = {str(p): digest(p) for p in (*paths.values(), checkpoint / 'model.pt',
                                              checkpoint / 'config.json', *sources())}
    for p in (checkpoint / 'tokenizer').rglob('*'):
        if p.is_file(): fingerprints[str(p)] = digest(p)
    for p in (checkpoint.parent / 'text_encoder').rglob('*'):
        if p.is_file(): fingerprints[str(p)] = digest(p)
    # Check input width and strict checkpoint compatibility before queueing days of work.
    state = support.torch.load(checkpoint / 'model.pt', map_location='cpu', weights_only=True)
    if state['scene_projection.0.weight'].shape[1] != 10 or state['route_queries'].shape[0] != 8:
        raise ValueError('Expected the original 10-feature, 8-query Transformer 1 reference checkpoint.')
    if support.v2.SCENE_FEATURE_SIZE != 12:
        raise ValueError('This study requires the current 12-feature Transformer 2 implementation.')
    del state
    manifest = {'schema_version': 1, 'root': str(root), 'dataset': {k: str(v) for k, v in paths.items()},
                'baseline_checkpoint': str(checkpoint), 'baseline_config': baseline_config,
                'config': config, 'seeds': [42, 43, 44], 'minimum_improvement_m': args.minimum_improvement_m,
                'fingerprints': fingerprints, 'code_revision': subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=PACKAGE, text=True).strip(),
                'device': args.device, 'smoke': args.smoke, 'created_at': time.time(),
                'dependency_versions': dependency_versions(),
                'historical_provenance': 'Baseline records dataset paths, not original CSV hashes; exact historical contents cannot be certified.',
                'test_note': 'Existing test split has been inspected previously; it is not a pristine prospective holdout.'}
    if args.smoke:
        manifest['config']['training']['max_epochs'] = 1
        manifest['config']['training']['batch_size'] = 2
        manifest['config']['training']['gradient_accumulation_steps'] = 1
        manifest['config']['training']['validation_batch_size'] = 8
    support.save_json(root / 'manifest.json', manifest)
    support.save_json(root / 'audit.json', audit)
    support.save_json(root / 'state.json', {'status': 'ready', 'results': {}, 'task': plan(manifest, {})})
    render(root)
    print(f'Initialized {root}. No training jobs submitted yet.')


def render(root):
    from ablation_dashboard import render_dashboard
    render_dashboard(Path(root))


def effective_config(manifest, architecture, weights, seed):
    config = copy.deepcopy(manifest['config'])
    if architecture == 'v1':
        config['model'] = copy.deepcopy(manifest['baseline_config']['model'])
        config['model']['fixed_endpoints'] = True
        config['model']['text_encoder_name'] = manifest['config']['model']['text_encoder_name']
    config['training']['seed'] = seed
    config['study_loss_weights'] = weights
    config['loss'].update(corridor_weight=weights['corridor'], squared_tail_weight=weights['squared'],
                          clearance_bin_weight=0.0)
    return config


def advance(args):
    import ablation_support as support
    root = args.study_root.resolve()
    manifest = read(root / 'manifest.json')
    device = support.torch.device(manifest['device'])
    if device.type == 'cuda' and not support.torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU available. Run the study inside the documented GPU batch job.')
    state = read(root / 'state.json')
    results = state['results']
    if state['status'] == 'complete':
        print('Study already complete.'); return False
    if dependency_versions() != manifest['dependency_versions']:
        raise ValueError('Dependency versions changed since initialization. Restore the recorded environment.')
    for path, expected in manifest['fingerprints'].items():
        if digest(path) != expected:
            raise ValueError(f'Input/code changed since initialization: {path}. Restore it or initialize a new study.')
    task = plan(manifest, results)
    state.update(status='running', task=task, error=None)
    support.save_json(root / 'state.json', state); render(root)
    signal.signal(signal.SIGUSR1, support.request_stop)
    limit = 8 if manifest['smoke'] else None
    safety = not manifest['smoke']
    if task['kind'] == 'baseline':
        config = copy.deepcopy(manifest['config'])
        config['model'] = manifest['baseline_config']['model']
        report = support.checkpoint_report(manifest['baseline_checkpoint'], config,
                    manifest['dataset']['validation'], device, safety=safety, row_limit=limit)
        report.update(id='baseline', architecture='v1 historical 8-output', variant='historical Euclidean',
                      seed=manifest['baseline_config']['training']['seed'], split='validation')
    elif task['kind'] == 'train':
        config = effective_config(manifest, task['architecture'], task['weights'], task['seed'])
        run_dir = root / 'runs' / task['id']
        support.save_json(run_dir / 'config.json', config)
        preflight_path = root / f"preflight_{task['architecture']}.json"
        if not preflight_path.exists():
            support.verify.select_trainer(config['model'])
            support.verify.euclidean_imitation_loss = support.v2.euclidean_imitation_loss
            examples = support.examples_for(manifest['dataset']['train'], config)
            tok = support.AutoTokenizer.from_pretrained(config['model']['text_encoder_name'], use_fast=True)
            support.v1.set_seed(42)
            check = support.verify.tiny_overfit(config['model'], config['training'], examples, tok,
                        device, 8, 2 if manifest['smoke'] else 250, 5.0, 0.0 if manifest['smoke'] else .5)
            check['smoke_only'] = manifest['smoke']
            support.save_json(preflight_path, check)
        if not read(preflight_path)['passed'] and not manifest['smoke']:
            raise ValueError(f'Preflight failed. Inspect {preflight_path}; no full training started.')
        checkpoint = support.train_run(run_dir, config, task['weights'], task['seed'], device,
                                       epoch_callback=lambda history: render(root), row_limit=limit)
        if support.STOP_REQUESTED:
            raise support.ResumeLater()
        report = support.checkpoint_report(checkpoint, config, manifest['dataset']['validation'],
                                           device, safety=safety, row_limit=limit)
        report.update(task, split='validation', best_epoch=read(checkpoint / 'metrics.json')['epoch'],
                      training_elapsed_seconds=read(run_dir / 'history.json')[-1]['elapsed_seconds'])
    else:
        # Selection is persisted before any test inference, and cannot depend on it.
        support.save_json(root / 'selection.json', task)
        selected = results[task['selected_run']]
        config = read(Path(selected['checkpoint']) / 'config.json')
        report = support.checkpoint_report(selected['checkpoint'], config, manifest['dataset']['test'],
                                           device, safety=safety, row_limit=limit, include_diagnostics=False)
        report.update(id='final', split='test', selected_run=task['selected_run'])
        support.save_json(root / 'final_test.json', report)
        state.update(status='complete', task=task, completed_at=time.time())
        support.save_json(root / 'state.json', state); render(root)
        return False
    report['smoke_only'] = manifest['smoke']
    support.save_json(root / 'reports' / f"{task['id']}.json", report)
    results[task['id']] = {k: v for k, v in report.items() if k not in ('route_examples', 'diagnostics')}
    state.update(status='ready', results=results, task=plan(manifest, results))
    support.save_json(root / 'state.json', state); render(root)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'advance', 'local', 'dashboard'])
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=PACKAGE / 'fine_tuning/datasets/dss_teacher_only_80_10_10')
    parser.add_argument('--baseline-checkpoint', type=Path,
                        default=PACKAGE / 'fine_tuning/outputs/overfitted/hrrt_waypoint_transformer/checkpoint-epoch-040')
    parser.add_argument('--max-epochs', type=int, default=100)
    parser.add_argument('--minimum-improvement-m', type=float, default=.02)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true', help='Separate eight-row/one-epoch study; not scientific results')
    args = parser.parse_args()
    if args.max_epochs < 1 or args.minimum_improvement_m < 0:
        parser.error('Epoch count must be positive; improvement threshold must be nonnegative')
    root = args.study_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    # Cross-process lock prevents simultaneous manual submissions from corrupting state.
    with (root / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.action == 'init':
                initialize(args); return
            if args.action == 'dashboard':
                render(root); return
            while advance(args):
                if args.action == 'advance':
                    break
        except Exception as exc:
            import ablation_support as support
            if isinstance(exc, support.ResumeLater):
                state = read(root / 'state.json'); state['status'] = 'ready'
                support.save_json(root / 'state.json', state); render(root)
                print('Saved epoch boundary; requesting another allocation.', flush=True)
                sys.exit(75)
            if (root / 'state.json').exists():
                state = read(root / 'state.json'); state.update(status='failed', error=str(exc))
                support.save_json(root / 'state.json', state); render(root)
            raise


if __name__ == '__main__':
    main()
