#!/usr/bin/env python3
"""Eight-output V1 loss study on a standalone GPU server (no Slurm required)."""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import ablation_support as support
from run_loss_ablation import VARIANTS, dependency_versions, digest, read, run_id, score, sources

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2]
STUDY_KIND = 'baseline_eight_waypoint_losses_v1'
LOSS_CONFIG = dict(corridor_margin_m=.2, corridor_temperature_m=.35,
                   squared_tail_scale_m=1., minimum_clearance_m=.4,
                   clearance_samples_per_segment=8, clearance_bin_weight=0.)


def plan(manifest, results):
    """Choose losses on validation only; the historical reference is not a control."""
    seed = manifest['seeds'][0]
    if 'baseline' not in results:
        return dict(id='baseline', kind='baseline')
    variants = copy.deepcopy(VARIANTS)

    def task(variant, seed):
        return dict(id=run_id('v1', variant, seed), kind='train', architecture='v1',
                    variant=variant, seed=seed, weights=variants[variant])

    for variant in variants:
        if run_id('v1', variant, seed) not in results:
            return task(variant, seed)
    control = score(results[run_id('v1', 'E0', seed)])
    corridor = min(('E2', 'E3'), key=lambda v: score(results[run_id('v1', v, seed)]))
    if (score(results[run_id('v1', 'E1', seed)]) < control and
            score(results[run_id('v1', corridor, seed)]) < control):
        variants['E4'] = dict(squared=.25, corridor=variants[corridor]['corridor'])
        if run_id('v1', 'E4', seed) not in results:
            return task('E4', seed)
    leader = min((v for v in variants if v != 'E0'),
                 key=lambda v: score(results[run_id('v1', v, seed)]))
    for seed in manifest['seeds'][1:]:
        for variant in ('E0', leader):
            if run_id('v1', variant, seed) not in results:
                return task(variant, seed)
    gains = [score(results[run_id('v1', 'E0', s)]) - score(results[run_id('v1', leader, s)])
             for s in manifest['seeds']]
    mean = sum(gains) / len(gains)
    supported = mean >= manifest['minimum_improvement_m'] and sum(x > 0 for x in gains) > len(gains) / 2
    winner = leader if supported else 'E0'
    ordered = sorted((run_id('v1', winner, s) for s in manifest['seeds']), key=lambda k: score(results[k]))
    return dict(id='final', kind='final', architecture='v1', winner_variant=winner,
                candidate_variant=leader, selected_run=ordered[len(ordered) // 2],
                paired_improvements_m=gains, mean_improvement_m=mean,
                conclusion='auxiliary loss supported' if supported else 'no conclusive auxiliary-loss gain; retain E0')


def effective_config(manifest, weights, seed, output_dir):
    config = copy.deepcopy(manifest['config'])
    config['training'].update(seed=seed, output_dir=str(output_dir))
    config['study_loss_weights'] = copy.deepcopy(weights)
    config['loss'].update(corridor_weight=weights['corridor'], squared_tail_weight=weights['squared'])
    return config


def objective(predicted, batch, weights, loss_config):
    # Match the original V1 objective: every point, including endpoints, contributes.
    euclidean = support.v1.euclidean_imitation_loss(predicted, batch['target_xy'])
    terms = {'euclidean': euclidean}
    total = euclidean
    if weights['squared']:
        terms['squared'] = (predicted - batch['target_xy']).square().sum(-1).mean() / loss_config['squared_tail_scale_m']
        total = total + weights['squared'] * terms['squared']
    if weights['corridor']:
        terms['corridor'] = support.v2.corridor_side_loss(
            predicted, batch['target_xy'], batch['examples'],
            loss_config['corridor_margin_m'], loss_config['corridor_temperature_m'])
        total = total + weights['corridor'] * terms['corridor']
    return total, terms


def loader_for(examples, tokenizer, config, shuffle=False):
    collator = support.v1.BatchCollator(tokenizer, config['model']['max_text_length'], config['model']['max_scene_tokens'])
    training = config['training']
    return support.DataLoader(support.v1.WaypointDataset(examples),
        batch_size=training['batch_size' if shuffle else 'validation_batch_size'],
        shuffle=shuffle, num_workers=training['num_workers'],
        pin_memory=support.torch.cuda.is_available(), collate_fn=collator)


def train_run(run_dir, config, weights, seed, device, epoch_callback=None, row_limit=None):
    """Original V1 optimizer/schedule/accumulation with epoch-boundary resume."""
    torch = support.torch
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    support.v1.set_seed(seed)
    tokenizer = support.AutoTokenizer.from_pretrained(config['tokenizer_name'], use_fast=True)
    train = support.examples_for(config['data']['train_csv'], config, row_limit)
    val = support.examples_for(config['data']['validation_csv'], config, row_limit)
    train_loader = loader_for(train, tokenizer, config, True)
    val_loader = loader_for(val, tokenizer, config)
    model = support.v1.build_model(config['model']).to(device)
    training = config['training']
    optimizer = support.verify.make_optimizer(model, training)
    accumulation = training['gradient_accumulation_steps']
    total_updates = math.ceil(len(train_loader) / accumulation) * training['max_epochs']
    scheduler = support.get_cosine_schedule_with_warmup(optimizer,
        num_warmup_steps=max(1, int(total_updates * training['warmup_ratio'])), num_training_steps=total_updates)
    amp = device.type == 'cuda'
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp and dtype == torch.float16)
    history, best, best_state, start, elapsed = [], None, None, 1, 0.
    resume_path = run_dir / 'resume.pt'
    if resume_path.exists():
        resume = torch.load(resume_path, map_location='cpu', weights_only=False)
        if resume['config'] != config or resume['weights'] != weights or resume['seed'] != seed:
            raise ValueError('Resume configuration differs; initialize a separate study.')
        model.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler']); scaler.load_state_dict(resume['scaler'])
        history, best, best_state = resume['history'], resume['best'], resume['best_state']
        start, elapsed = resume['epoch'] + 1, resume['elapsed_seconds']
        support.restore_rng(resume['rng']); del resume
    if not (run_dir / 'text_encoder').exists():
        # Save the base encoder for offline rebuilds; model.pt contains fine-tuned weights.
        model.text_encoder.save_pretrained(run_dir / 'text_encoder')
    tokenizer.save_pretrained(run_dir / 'tokenizer')
    clock = time.monotonic()
    for epoch in range(start, training['max_epochs'] + 1):
        if support.STOP_REQUESTED:
            raise support.ResumeLater()
        model.train(); optimizer.zero_grad(set_to_none=True)
        totals, count, interior_sum, batch_losses = {}, 0, 0., []
        for step, raw in enumerate(train_loader, start=1):
            batch = support.v1.move_batch(raw, device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                predicted = support.forward(model, batch)
                loss, terms = objective(predicted, batch, weights, config['loss'])
                # Deliberately preserve the historical divisor even for a partial final group.
                scaled_loss = loss / accumulation
            if not torch.isfinite(loss):
                raise ValueError(f'Non-finite loss at epoch {epoch}, batch {step}')
            scaler.scale(scaled_loss).backward()
            n = len(batch['examples']); count += n
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.) + float(value.detach()) * n
            interior_sum += float(support.v2.euclidean_imitation_loss(predicted.detach(), batch['target_xy'])) * n
            batch_losses.append(float(loss.detach()))
            if step % accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training['max_gradient_norm'])
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        metrics, _ = support.evaluate(model, val_loader, device, config)
        record = dict(epoch=epoch, train_mean_interior_error_m=interior_sum / count,
            train_euclidean_all8_m=totals['euclidean'] / count,
            train_objective_batch_mean=sum(batch_losses) / len(batch_losses),
            train_terms={k: v / count for k, v in totals.items()}, validation=metrics,
            elapsed_seconds=elapsed + time.monotonic() - clock)
        history.append(record)
        if best is None or metrics['mean_interior_error_m'] < best['validation']['mean_interior_error_m']:
            best = copy.deepcopy(record)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        state = dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), rng=support.rng_state(),
            history=history, best=best, best_state=best_state, elapsed_seconds=record['elapsed_seconds'],
            config=config, weights=weights, seed=seed)
        torch.save(state, run_dir / 'resume.pt.tmp'); os.replace(run_dir / 'resume.pt.tmp', resume_path)
        support.save_json(run_dir / 'history.json', history)
        print(json.dumps({'run': run_dir.name, **record}), flush=True)
        if epoch_callback:
            epoch_callback(history)
    checkpoint = run_dir / 'best'; checkpoint.mkdir(exist_ok=True)
    torch.save(best_state, checkpoint / 'model.pt.tmp'); os.replace(checkpoint / 'model.pt.tmp', checkpoint / 'model.pt')
    support.save_json(checkpoint / 'config.json', config)
    support.save_json(checkpoint / 'metrics.json', best)
    tokenizer.save_pretrained(checkpoint / 'tokenizer')
    return checkpoint


def initialize(args):
    root = args.study_root.resolve()
    if (root / 'manifest.json').exists():
        raise ValueError('Study exists; use run to resume, or choose a new output directory.')
    if any(p.name != '.lock' for p in root.iterdir()):
        raise ValueError('Output directory is not empty; choose a new directory. Existing dashboards and study files will not be overwritten.')
    checkpoint = args.baseline_checkpoint.resolve()
    paths = {s: (args.dataset_root / f'hrrt_sft_{s}.csv').resolve() for s in ('train', 'validation', 'test')}
    for path in (*paths.values(), checkpoint / 'config.json', checkpoint / 'model.pt',
                 checkpoint / 'tokenizer/tokenizer_config.json', checkpoint.parent / 'text_encoder/config.json'):
        if not path.is_file():
            raise FileNotFoundError(f'Required real artifact missing: {path}')
    baseline = read(checkpoint / 'config.json')
    if (baseline['model'].get('architecture_version') == 'transformer_2_0' or
            baseline['model'].get('fixed_endpoints', False) or baseline['model']['waypoint_count'] != 8):
        raise ValueError('Expected the original eight-output Transformer 1 checkpoint.')
    for split in ('train', 'validation'):
        original = Path(baseline['data'][f'{split}_csv'])
        if original.name != paths[split].name or original.parent.name != paths[split].parent.name:
            raise ValueError(f'Dataset differs from baseline configuration: {original}')
    audit = support.verify.dataset_audit(paths)
    if not audit['passed'] or any(r['rows'] != r['waypoint_rows'] for r in audit['splits'].values()):
        raise ValueError(f'Dataset audit failed: {audit}')
    geometries, sample_ids = {}, {}
    for split, path in paths.items():
        geometries[split], sample_ids[split] = set(), set()
        with path.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                sid = row['sample_id']
                if sid in sample_ids[split]:
                    raise ValueError(f'Duplicate sample: {split}/{sid}')
                sample_ids[split].add(sid)
                env = json.loads(row['environment'])
                geo = {k: env[k] for k in ('workspace', 'start', 'goal', 'obstacles')}
                geo['obstacles'] = sorted(geo['obstacles'], key=lambda ob: json.dumps(ob, sort_keys=True))
                geometries[split].add(json.dumps(geo, sort_keys=True))
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if geometries[a] & geometries[b] or sample_ids[a] & sample_ids[b]:
            raise ValueError(f'Dataset overlap: {a}/{b}')
    weights = support.torch.load(checkpoint / 'model.pt', map_location='cpu', weights_only=True)
    if weights['route_queries'].shape[0] != 8 or weights['scene_projection.0.weight'].shape[1] != 10:
        raise ValueError('Expected 8 queries and 10 scene features.')
    del weights
    config = copy.deepcopy(baseline)
    config['data'].update(train_csv=str(paths['train']), validation_csv=str(paths['validation']))
    config['model']['text_encoder_name'] = str(checkpoint.parent / 'text_encoder')
    config['tokenizer_name'] = str(checkpoint / 'tokenizer')
    config['loss'] = copy.deepcopy(LOSS_CONFIG)
    config['training']['output_dir'] = str(root / 'runs')
    config['selection_metric'] = 'validation.mean_interior_error_m'
    if config['data'].get('row_limit') is not None:
        raise ValueError('Historical config has a row limit; expected full teacher-only data.')
    if args.smoke:
        config['training'].update(max_epochs=1, batch_size=2, validation_batch_size=8,
                                  gradient_accumulation_steps=1, num_workers=0)
    artifacts = [*paths.values(), checkpoint / 'config.json', checkpoint / 'model.pt', *sources(), Path(__file__).resolve()]
    for folder in (checkpoint / 'tokenizer', checkpoint.parent / 'text_encoder'):
        artifacts.extend(p for p in folder.rglob('*') if p.is_file())
    manifest = dict(study_kind=STUDY_KIND, root=str(root), config=config, baseline_config=baseline,
        baseline_checkpoint=str(checkpoint), dataset={k: str(v) for k, v in paths.items()},
        seeds=[baseline['training']['seed'], baseline['training']['seed'] + 1, baseline['training']['seed'] + 2],
        minimum_improvement_m=args.minimum_improvement_m, device=args.device, smoke=args.smoke,
        created_at=time.time(), dependency_versions=dependency_versions(),
        fingerprints={str(p): digest(p) for p in artifacts},
        code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PACKAGE, text=True).strip(),
        provenance='Original config and current production V1 code; historical CSV hashes/software versions were not recorded. Fresh planners start from base MiniLM, not epoch-40 weights.',
        schedule_note='Historical optimizer, learning rates, batch sizes, workers, cosine schedule and accumulation. Fixed epoch budget; validation selects best interior error. Full pipeline evaluated per best run, not each epoch.')
    support.save_json(root / 'manifest.json', manifest)
    support.save_json(root / 'audit.json', audit)
    support.save_json(root / 'state.json', dict(status='ready', results={}, task=plan(manifest, {})))
    render(root)
    print(f'Initialized {root}; training settings copied from {checkpoint / "config.json"}', flush=True)


def render(root):
    """Reuse the existing offline dashboard, adapting labels to eight-output training."""
    def optional(path, default):
        return read(path) if path.exists() else default
    reports = {p.stem: read(p) for p in sorted((root / 'reports').glob('*.json'))}
    if (root / 'final_test.json').exists():
        reports['final'] = read(root / 'final_test.json')
    data = dict(manifest=read(root / 'manifest.json'), state=read(root / 'state.json'),
        audit=optional(root / 'audit.json', {}), selection=optional(root / 'selection.json', {}),
        reports=reports, histories={p.parent.name: read(p) for p in sorted((root / 'runs').glob('*/history.json'))},
        preflight={p.stem: read(p) for p in root.glob('preflight_*.json')})
    template = (HERE / 'ablation_dashboard.html').read_text()
    template = template.replace('Transformer loss ablation study', 'Baseline eight-waypoint loss study')
    template = template.replace('The architecture screen compares complete configurations against the historical 8-output checkpoint. All fresh runs fix endpoints and learn six points. Selection is provisional until repeated-seed confirmation.',
        'All fresh runs use the original eight-output V1 configuration. Loss selection compares against fresh E0 across paired seeds; the saved baseline is a reference.')
    template = template.replace('x.train_mean_interior_error_m', 'x.train_euclidean_all8_m')
    template = template.replace('x.validation.mean_interior_error_m', 'x.validation.full_route_mean_error_m')
    template = template.replace('Training Euclidean', 'Training all-8 Euclidean').replace('Validation Euclidean', 'Validation all-8 Euclidean')
    template = template.replace('Validation episode P80', 'Validation interior episode P80')
    glossary = '''<section><h2>Eight-waypoint loss definitions</h2>
<p>All runs predict all 8 points, including start and goal. d_j = ‖ŵ_j − w_j‖₂; C is the existing teacher-side corridor hinge, margin 0.20 m, soft-selection temperature 0.35 m. S = (1/8) Σ d_j² / (1 m).</p>
<table><tr><th>Run</th><th>Training objective</th></tr>
<tr><td>baseline / E0</td><td>L₀ = (1/8) Σⱼ₌₁⁸ d_j</td></tr>
<tr><td>E1</td><td>L₀ + 0.25 S</td></tr><tr><td>E2</td><td>L₀ + C</td></tr>
<tr><td>E3</td><td>L₀ + 5 C</td></tr><tr><td>E4 (conditional)</td><td>L₀ + 0.25 S + selected corridor weight × C</td></tr></table>
<p>v1_E1_s42 means V1 architecture, E1 loss, random seed 42. The historical checkpoint is a reference; every fresh run starts from the same base MiniLM and a fresh planner for its seed. Total and planner parameter counts are in the leaderboard; text encoder count = total − planner.</p></section>'''
    template = template.replace('<section><h2>Learning curves</h2>', glossary + '<section><h2>Learning curves</h2>')
    payload = json.dumps(data, allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    output = root / 'dashboard'; output.mkdir(exist_ok=True)
    temporary = output / 'index.html.tmp'
    temporary.write_text(template.replace('__STUDY_DATA__', payload), encoding='utf-8')
    os.replace(temporary, output / 'index.html')


def advance(args):
    root = args.study_root.resolve()
    manifest, state = read(root / 'manifest.json'), read(root / 'state.json')
    if manifest.get('study_kind') != STUDY_KIND:
        raise ValueError('Wrong study type; use a separate baseline-eight output directory.')
    if state['status'] == 'complete':
        return False
    if dependency_versions() != manifest['dependency_versions']:
        raise ValueError('Dependencies changed; restore the initialized environment.')
    for path, expected in manifest['fingerprints'].items():
        if digest(path) != expected:
            raise ValueError(f'Frozen artifact changed: {path}')
    device = support.torch.device(manifest['device'])
    if device.type == 'cuda' and not support.torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Check the GPU environment and CUDA_VISIBLE_DEVICES.')
    results = state['results']; task = plan(manifest, results)
    state.update(status='running', task=task, error=None)
    support.save_json(root / 'state.json', state); render(root)
    print(json.dumps(dict(event='task_started', task=task, timestamp=time.time())), flush=True)
    limit, safety = (8 if manifest['smoke'] else None), not manifest['smoke']
    if task['kind'] == 'baseline':
        report = support.checkpoint_report(manifest['baseline_checkpoint'], manifest['config'],
            manifest['dataset']['validation'], device, safety=safety, row_limit=limit)
        report.update(id='baseline', architecture='V1 eight outputs', variant='all-eight Euclidean', split='validation')
    elif task['kind'] == 'train':
        run_dir = root / 'runs' / task['id']
        config = effective_config(manifest, task['weights'], task['seed'], run_dir)
        support.save_json(run_dir / 'config.json', config)
        preflight = root / 'preflight_v1.json'
        if not preflight.exists():
            support.verify.select_trainer(config['model'])
            examples = support.examples_for(manifest['dataset']['train'], config, limit=8)
            tok = support.AutoTokenizer.from_pretrained(config['tokenizer_name'], use_fast=True)
            support.v1.set_seed(manifest['seeds'][0])
            check = support.verify.tiny_overfit(config['model'], config['training'], examples, tok,
                device, 8, 2 if manifest['smoke'] else 250, 5., 0. if manifest['smoke'] else .5)
            check['smoke_only'] = manifest['smoke']; support.save_json(preflight, check)
        if not read(preflight)['passed'] and not manifest['smoke']:
            raise ValueError(f'Preflight failed: {preflight}')
        checkpoint = train_run(run_dir, config, task['weights'], task['seed'], device,
                               epoch_callback=lambda history: render(root), row_limit=limit)
        if support.STOP_REQUESTED:
            raise support.ResumeLater()
        report = support.checkpoint_report(checkpoint, config, manifest['dataset']['validation'],
            device, safety=safety, row_limit=limit)
        report.update(task, split='validation', best_epoch=read(checkpoint / 'metrics.json')['epoch'],
                      training_elapsed_seconds=read(run_dir / 'history.json')[-1]['elapsed_seconds'])
    else:
        support.save_json(root / 'selection.json', task)
        checkpoint = Path(results[task['selected_run']]['checkpoint'])
        report = support.checkpoint_report(checkpoint, read(checkpoint / 'config.json'),
            manifest['dataset']['test'], device, safety=safety, row_limit=limit, include_diagnostics=False)
        report.update(id='final', split='test', selected_run=task['selected_run'], smoke_only=manifest['smoke'])
        support.save_json(root / 'final_test.json', report)
        state.update(status='complete', task=task, completed_at=time.time())
        support.save_json(root / 'state.json', state); render(root)
        print(json.dumps(dict(event='study_complete', selected_run=task['selected_run'])), flush=True)
        return False
    report['smoke_only'] = manifest['smoke']
    support.save_json(root / 'reports' / f'{task["id"]}.json', report)
    results[task['id']] = {k: v for k, v in report.items() if k not in ('route_examples', 'diagnostics')}
    state.update(status='ready', results=results, task=plan(manifest, results))
    support.save_json(root / 'state.json', state); render(root)
    print(json.dumps(dict(event='task_finished', run=task['id'], next_task=state['task']['id'])), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'run', 'advance', 'dashboard'])
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, default=PACKAGE / 'fine_tuning/datasets/dss_teacher_only_80_10_10')
    parser.add_argument('--baseline-checkpoint', type=Path,
        default=PACKAGE / 'fine_tuning/outputs/overfitted/hrrt_waypoint_transformer/checkpoint-epoch-040')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--minimum-improvement-m', type=float, default=.02)
    parser.add_argument('--smoke', action='store_true', help='8 rows / 1 epoch; not scientific results')
    args = parser.parse_args()
    if not math.isfinite(args.minimum_improvement_m) or args.minimum_improvement_m < 0:
        parser.error('Minimum improvement must be finite and nonnegative')
    root = args.study_root.resolve(); root.mkdir(parents=True, exist_ok=True)
    with (root / '.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(2, 'A worker already holds this study lock.\n')
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            signal.signal(signum, support.request_stop)
        try:
            if args.action == 'init':
                initialize(args); return
            if read(root / 'manifest.json').get('study_kind') != STUDY_KIND:
                raise ValueError('Expected a baseline-eight study; do not reuse the NCSA study directory.')
            if args.action == 'dashboard':
                render(root); return
            while True:
                if support.STOP_REQUESTED:
                    raise support.ResumeLater()
                if not advance(args) or args.action == 'advance':
                    break
        except support.ResumeLater:
            state = read(root / 'state.json'); state.update(status='paused')
            support.save_json(root / 'state.json', state); render(root)
            print('Paused at a saved epoch/task boundary. Use run to resume.', flush=True)
        except Exception as exc:
            if args.action != 'init' and (root / 'state.json').exists() and (root / 'manifest.json').exists() and read(root / 'manifest.json').get('study_kind') == STUDY_KIND:
                state = read(root / 'state.json'); state.update(status='failed', error=str(exc))
                support.save_json(root / 'state.json', state); render(root)
            raise


if __name__ == '__main__':
    main()
