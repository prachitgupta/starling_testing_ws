"""Shared training/evaluation for loss studies; production models and losses only."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

import verify_transformer_components as verify

v1, v2 = verify.trainer_v1, verify.trainer_v2
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


class ResumeLater(Exception):
    """Stop at a saved epoch boundary and schedule another allocation."""


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean_json(value), indent=2), encoding='utf-8')
    os.replace(temporary, path)


def trainer_for(config):
    return v2 if config['model'].get('architecture_version') == 'transformer_2_0' else v1


def examples_for(path, config, limit=None):
    # Always retain the teacher key used by the shared auxiliary losses.
    examples = v2.read_examples(Path(path), config['data']['distillation_mode'], 8, limit)
    if trainer_for(config) is v1:
        examples = [replace(e, scene_tokens=v1.encode_scene(e.environment)) for e in examples]
    if not examples:
        raise ValueError(f'No usable examples: {path}')
    for e in examples:
        expected = np.array([[e.environment[k]['x'], e.environment[k]['y']]
                             for k in ('start', 'goal')])
        if not np.allclose(e.target_xy[[0, -1]], expected, atol=1e-4):
            raise ValueError(f'Teacher endpoints disagree with scene: {e.sample_id}')
    return examples


def loader_for(examples, tokenizer, config, shuffle=False):
    trainer = trainer_for(config)
    collator = trainer.BatchCollator(tokenizer, config['model']['max_text_length'],
                                    config['model']['max_scene_tokens'])
    size = config['training']['batch_size' if shuffle else 'validation_batch_size']
    # A single-process loader makes epoch-boundary RNG restoration reproducible.
    return DataLoader(v1.WaypointDataset(examples), batch_size=size, shuffle=shuffle,
                      collate_fn=collator, num_workers=0), collator


def forward(model, batch):
    normalized = model(batch['input_ids'], batch['attention_mask'],
                       batch['scene_tokens'], batch['scene_padding_mask'])
    return v1.denormalize_predictions(normalized, batch['examples'])


def objective(predicted, batch, weights, loss_config):
    euclidean = v2.euclidean_imitation_loss(predicted, batch['target_xy'])
    total = euclidean
    terms = {'euclidean': euclidean}
    if weights['corridor']:
        terms['corridor'] = v2.corridor_side_loss(
            predicted, batch['target_xy'], batch['examples'],
            loss_config['corridor_margin_m'], loss_config['corridor_temperature_m'])
        total = total + weights['corridor'] * terms['corridor']
    if weights['squared']:
        terms['squared'] = v2.squared_tail_loss(
            predicted, batch['target_xy'], loss_config['squared_tail_scale_m'])
        total = total + weights['squared'] * terms['squared']
    return total, terms


def evaluate(model, loader, device, config, detailed=False, safety=False):
    model.eval()
    distances, predictions, examples = [], [], []
    side_penalties, signed, bin_penalties, bin_correct = [], [], [], []
    loss_cfg = config['loss']
    with torch.no_grad():
        for raw in loader:
            batch = v1.move_batch(raw, device)
            predicted = forward(model, batch)
            distances.append(torch.linalg.vector_norm(predicted - batch['target_xy'], dim=-1).cpu().numpy())
            predictions.extend(predicted.cpu().numpy())
            examples.extend(batch['examples'])
            if detailed:
                p, s = v2.corridor_side_terms(predicted, batch['target_xy'], batch['examples'],
                                            loss_cfg['corridor_margin_m'], loss_cfg['corridor_temperature_m'])
                b, c = v2.clearance_bin_terms(predicted, batch['target_xy'], batch['examples'],
                                            loss_cfg['minimum_clearance_m'], loss_cfg['clearance_samples_per_segment'])
                side_penalties.extend(p.cpu().tolist()); signed.extend(s.cpu().tolist())
                bin_penalties.extend(b.cpu().tolist()); bin_correct.extend(c.cpu().tolist())
    errors = np.concatenate(distances)
    if not np.isfinite(errors).all():
        raise ValueError('Evaluation produced non-finite waypoint errors')
    interior = errors[:, 1:-1]
    episode = interior.mean(axis=1)
    metrics = {
        'samples': len(errors), 'mean_interior_error_m': float(interior.mean()),
        'median_interior_waypoint_error_m': float(np.median(interior)),
        'p80_episode_error_m': float(np.quantile(episode, .80)),
        'p95_interior_waypoint_error_m': float(np.quantile(interior, .95)),
        'maximum_interior_waypoint_error_m': float(interior.max()),
        'full_route_mean_error_m': float(errors.mean()),
        'endpoint_mean_error_m': float(errors[:, [0, -1]].mean()),
        'squared_tail_loss_m': float(np.square(interior).mean() / loss_cfg['squared_tail_scale_m']),
    }
    if detailed:
        metrics.update(wrong_side_rate=float(np.mean(np.array(signed) <= 0)) if signed else None,
                       corridor_decisions=len(signed),
                       corridor_loss_m=float(np.mean(side_penalties)) if signed else None,
                       clearance_bin_accuracy=float(np.mean(bin_correct)) if bin_correct else None,
                       clearance_bin_decisions=len(bin_correct),
                       clearance_bin_loss_m=float(np.mean(bin_penalties)) if bin_correct else None)
    if safety:
        metrics.update(v1.safety_radius_for_predictions(predictions, examples, config['validation']))
        metrics['safety_radius_status'] = ('finite' if math.isfinite(metrics['safety_radius_m'])
                                           else 'unavailable: pipeline did not pass every evaluated episode')
    routes = []
    if detailed:
        for e, pred, err in zip(examples, predictions, errors):
            routes.append({'sample_id': e.sample_id, 'preference': e.preference,
                           'scene_key': json.dumps(e.environment, sort_keys=True),
                           'environment': e.environment, 'expert_xy': e.target_xy.tolist(),
                           'predicted_xy': pred.tolist(), 'interior_mean_error_m': float(err[1:-1].mean()),
                           'full_route_mean_error_m': float(err.mean())})
    return metrics, routes


def diagnostics(model, examples, collator, config, device):
    verify.select_trainer(config['model'])
    batch = v1.move_batch(collator(examples[:8]), device)
    model.eval(); model.zero_grad(set_to_none=True)
    v2.euclidean_imitation_loss(forward(model, batch), batch['target_xy']).backward()
    gradients = verify.grouped_gradient_norms(model)
    probes = []
    seen = set()
    for i, example in enumerate(examples):
        key = json.dumps(example.environment, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        probes.append(verify.input_sensitivity(model, collator, examples[i:] + examples[:i], device))
        if len(probes) == 12:
            break
    model.zero_grad(set_to_none=True)
    return {'gradient_norms': gradients, 'gradient_loss': 'interior Euclidean, first 8 validation rows',
            'finite_nonzero_groups': all(math.isfinite(v) and v > 0 for v in gradients.values()),
            'sensitivity_probes': probes,
            'sensitivity_note': 'Finite input swaps across up to 12 scenes; not a normalized importance ranking.'}


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def train_run(run_dir, config, weights, seed, device, epoch_callback=None, row_limit=None):
    """Common six-point objective; atomic full-state resume at epoch boundaries."""
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    v1.set_seed(seed)
    trainer = trainer_for(config)
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text_encoder_name'], use_fast=True)
    train = examples_for(config['data']['train_csv'], config, row_limit)
    val = examples_for(config['data']['validation_csv'], config, row_limit)
    train_loader, _ = loader_for(train, tokenizer, config, True)
    val_loader, _ = loader_for(val, tokenizer, config)
    model = trainer.build_model(config['model']).to(device)
    optimizer = verify.make_optimizer(model, config['training'])
    accumulation = config['training']['gradient_accumulation_steps']
    epochs = config['training']['max_epochs']
    total_updates = math.ceil(len(train_loader) / accumulation) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer,
        num_warmup_steps=max(1, int(total_updates * config['training']['warmup_ratio'])),
        num_training_steps=total_updates)
    amp = device.type == 'cuda'
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=amp and dtype == torch.float16)
    history, best, best_state, start = [], None, None, 1
    elapsed = 0.0
    resume_path = run_dir / 'resume.pt'
    if resume_path.exists():
        # Only load our own trusted, locally generated training state.
        resume = torch.load(resume_path, map_location='cpu', weights_only=False)
        if resume['config'] != config or resume['weights'] != weights or resume['seed'] != seed:
            raise ValueError('Resume configuration differs; start a new study instead')
        model.load_state_dict(resume['model']); optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler']); scaler.load_state_dict(resume['scaler'])
        history, best, best_state = resume['history'], resume['best'], resume['best_state']
        start = resume['epoch'] + 1; elapsed = resume['elapsed_seconds']
        restore_rng(resume['rng'])
        del resume
    if not (run_dir / 'text_encoder').exists():
        model.text_encoder.save_pretrained(run_dir / 'text_encoder')
    tokenizer.save_pretrained(run_dir / 'tokenizer')
    clock = time.monotonic()
    for epoch in range(start, epochs + 1):
        if STOP_REQUESTED:
            raise ResumeLater()
        model.train(); optimizer.zero_grad(set_to_none=True)
        total_error = total_objective = 0.0; count = 0
        totals = {}
        for step, raw in enumerate(train_loader):
            batch = v1.move_batch(raw, device)
            group_start = (step // accumulation) * accumulation
            group_size = min(accumulation, len(train_loader) - group_start)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                loss, terms = objective(forward(model, batch), batch, weights, config['loss'])
            if not torch.isfinite(loss):
                raise ValueError(f'Non-finite training objective at epoch {epoch}, batch {step}')
            scaler.scale(loss / group_size).backward()
            n = len(batch['examples']); count += n
            total_error += float(terms['euclidean'].detach()) * n
            total_objective += float(loss.detach()) * n
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * n
            if (step + 1) % accumulation == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_gradient_norm'])
                scaler.step(optimizer); scaler.update(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        metrics, _ = evaluate(model, val_loader, device, config)
        record = {'epoch': epoch, 'train_mean_interior_error_m': total_error / count,
                  'train_objective': total_objective / count,
                  'train_terms': {k: v / count for k, v in totals.items()},
                  'validation': metrics, 'elapsed_seconds': elapsed + time.monotonic() - clock}
        history.append(record)
        if best is None or metrics['mean_interior_error_m'] < best['validation']['mean_interior_error_m']:
            best = copy.deepcopy(record)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'rng': rng_state(),
                 'history': history, 'best': best, 'best_state': best_state,
                 'elapsed_seconds': record['elapsed_seconds'], 'config': config, 'weights': weights, 'seed': seed}
        torch.save(state, run_dir / 'resume.pt.tmp'); os.replace(run_dir / 'resume.pt.tmp', resume_path)
        save_json(run_dir / 'history.json', history)
        print(json.dumps({'run': run_dir.name, **record}), flush=True)
        if epoch_callback:
            epoch_callback(history)
    checkpoint = run_dir / 'best'; checkpoint.mkdir(exist_ok=True)
    torch.save(best_state, checkpoint / 'model.pt.tmp')
    os.replace(checkpoint / 'model.pt.tmp', checkpoint / 'model.pt')
    save_json(checkpoint / 'config.json', config)
    save_json(checkpoint / 'metrics.json', best)
    tokenizer.save_pretrained(checkpoint / 'tokenizer')
    return checkpoint


def checkpoint_report(checkpoint, config, data_path, device, safety=True, row_limit=None, include_diagnostics=True):
    checkpoint = Path(checkpoint)
    model_config = copy.deepcopy(config['model'])
    if (checkpoint.parent / 'text_encoder').is_dir():
        model_config['text_encoder_name'] = str(checkpoint.parent / 'text_encoder')
    trainer = trainer_for(config)
    model = trainer.build_model(model_config).to(device)
    model.load_state_dict(torch.load(checkpoint / 'model.pt', map_location=device, weights_only=True), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / 'tokenizer', use_fast=True)
    examples = examples_for(data_path, config, row_limit)
    loader, collator = loader_for(examples, tokenizer, config)
    metrics, routes = evaluate(model, loader, device, config, detailed=True, safety=safety)
    return {'checkpoint': str(checkpoint), 'metrics': metrics, 'route_examples': routes,
            'diagnostics': diagnostics(model, examples, collator, config, device) if include_diagnostics else
                           {'note': 'No test-set gradient probes; diagnostics were measured on validation.'},
            'parameters': sum(p.numel() for p in model.parameters()),
            'planner_parameters': sum(p.numel() for n, p in model.named_parameters()
                                      if not n.startswith('text_encoder.')),
            'fixed_endpoints': bool(config['model'].get('fixed_endpoints') or trainer is v2)}
