"""Bounded checks for selection, endpoints and exact epoch-boundary resume."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

import ablation_support as support
from run_loss_ablation import plan, run_id, VARIANTS


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=8)
        self.embed = torch.nn.Embedding(16, 8)
        self.pooler = torch.nn.Linear(8, 8)

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)
        return types.SimpleNamespace(last_hidden_state=x, pooler_output=self.pooler(x[:, 0]))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


class TinyTokenizer:
    def __call__(self, texts, **kwargs):
        ids = torch.tensor([[1, 2 + len(t) % 10, 3] for t in texts])
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)


def fixture_config(root):
    environment = {'scene_id': 0, 'workspace': {'x': [-4, 4], 'y': [-3, 3], 'z': -.5},
                   'start': {'x': -3, 'y': 0, 'z': -.5}, 'goal': {'x': 3, 'y': 0, 'z': -.5},
                   'obstacles': []}
    data = root / 'data.csv'
    with data.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['sample_id', 'preference_text', 'environment', 'completion', 'distillation_mode'])
        writer.writeheader()
        for i in range(5):
            writer.writerow({'sample_id': str(i), 'preference_text': 'go straight ' + 'now ' * i,
                             'environment': json.dumps(environment), 'distillation_mode': 'dss',
                             'completion': json.dumps({'waypoints': [environment['start'], environment['goal']]})})
    encoder = root / 'encoder'; encoder.mkdir()
    return {'data': {'train_csv': str(data), 'validation_csv': str(data), 'distillation_mode': 'dss'},
            'model': {'text_encoder_name': str(encoder), 'hidden_size': 8, 'attention_heads': 2,
                      'feedforward_size': 16, 'fusion_layers': 1, 'decoder_layers': 1, 'dropout': .1,
                      'max_scene_tokens': 32, 'max_text_length': 32, 'waypoint_count': 8,
                      'fixed_endpoints': True, 'scene_mlp_layers': 1, 'scene_encoder_layers': 1,
                      'scene_feedforward_size': 16, 'interior_waypoint_count': 6},
            'training': {'batch_size': 2, 'validation_batch_size': 3, 'max_epochs': 2,
                         'gradient_accumulation_steps': 2, 'warmup_ratio': .05, 'max_gradient_norm': 1.,
                         'text_encoder_learning_rate': .001, 'planner_learning_rate': .001, 'weight_decay': .01},
            'loss': {'squared_tail_scale_m': 1., 'corridor_margin_m': .2, 'corridor_temperature_m': .35,
                     'minimum_clearance_m': .4, 'clearance_samples_per_segment': 8}, 'validation': {}}


class SelectionTests(unittest.TestCase):
    def test_full_sequence_and_no_test_selection(self):
        m = {'seeds': [42, 43, 44], 'minimum_improvement_m': .02}
        results = {}
        seen = []
        for _ in range(20):
            task = plan(m, results); seen.append(task['id'])
            if task['kind'] == 'final': break
            # T2 wins; E1 and E2 both improve, so combined E4 must be attempted.
            value = .6 if task['id'] == 'baseline' else {'E0': .5, 'E1': .46, 'E2': .45, 'E3': .55, 'E4': .4}[task['variant']]
            results[task['id']] = {'metrics': {'mean_interior_error_m': value}}
        self.assertEqual(task['kind'], 'final')
        self.assertEqual(task['winner_variant'], 'E4')
        self.assertIn('v2_E4_s42', seen)
        self.assertEqual(len(task['paired_improvements_m']), 3)
        results['test'] = {'metrics': {'mean_interior_error_m': -1000}}
        self.assertEqual(task, plan(m, results))

    def test_v1_fallback_and_inconclusive(self):
        m = {'seeds': [42, 43, 44], 'minimum_improvement_m': .02}
        r = {'baseline': {'metrics': {'mean_interior_error_m': .4}},
             'v2_E0_s42': {'metrics': {'mean_interior_error_m': .5}}}
        self.assertEqual(plan(m, r)['id'], 'v1_E0_s42')
        for seed in m['seeds']:
            for variant in VARIANTS:
                r[run_id('v1', variant, seed)] = {'metrics': {'mean_interior_error_m': .4 if variant == 'E0' else .41}}
        final = plan(m, r)
        self.assertEqual(final['winner_variant'], 'E0')


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = fixture_config(self.root)
        self.encoder_patch = patch('transformers.AutoModel.from_pretrained', side_effect=lambda *a, **k: TinyEncoder())
        self.token_patch = patch('transformers.AutoTokenizer.from_pretrained', return_value=TinyTokenizer())
        self.encoder_patch.start(); self.token_patch.start()

    def tearDown(self):
        support.STOP_REQUESTED = False
        self.encoder_patch.stop(); self.token_patch.stop(); self.tmp.cleanup()

    def test_model_shapes_endpoints_and_backward_compatibility(self):
        for architecture in ('v1', 'v2'):
            c = copy.deepcopy(self.cfg)
            if architecture == 'v2': c['model']['architecture_version'] = 'transformer_2_0'
            trainer = support.trainer_for(c)
            examples = support.examples_for(c['data']['train_csv'], c)
            loader, _ = support.loader_for(examples, TinyTokenizer(), c)
            b = next(iter(loader)); model = trainer.build_model(c['model'])
            predicted = support.forward(model, b)
            self.assertEqual(tuple(predicted.shape), (3, 8, 2))
            torch.testing.assert_close(predicted[:, [0, -1]], b['target_xy'][:, [0, -1]])
            self.assertEqual(model.route_queries.shape[0], 6)
            total, _ = support.objective(predicted, b, VARIANTS['E0'], c['loss'])
            torch.testing.assert_close(total, support.v2.euclidean_imitation_loss(predicted, b['target_xy']))
        old = dict(self.cfg['model']); old.pop('fixed_endpoints')
        first = support.v1.build_model(old); second = support.v1.build_model(old)
        second.load_state_dict(first.state_dict(), strict=True)
        self.assertEqual(first.route_queries.shape[0], 8)

    def test_exact_cpu_resume_and_sample_weighted_metric(self):
        weights = VARIANTS['E0']; device = torch.device('cpu')
        full = support.train_run(self.root / 'full', self.cfg, weights, 42, device)
        def interrupt(history):
            if len(history) == 1: raise support.ResumeLater()
        with self.assertRaises(support.ResumeLater):
            support.train_run(self.root / 'resumed', self.cfg, weights, 42, device, interrupt)
        resumed = support.train_run(self.root / 'resumed', self.cfg, weights, 42, device)
        a = torch.load(full / 'model.pt', weights_only=True)
        b = torch.load(resumed / 'model.pt', weights_only=True)
        for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        examples = support.examples_for(self.cfg['data']['train_csv'], self.cfg)
        loader, _ = support.loader_for(examples, TinyTokenizer(), self.cfg)
        model = support.v1.build_model(self.cfg['model']); model.load_state_dict(a)
        metrics, routes = support.evaluate(model, loader, device, self.cfg, detailed=True)
        self.assertEqual(metrics['samples'], 5)
        expected = np.mean([x['interior_mean_error_m'] for x in routes])
        self.assertAlmostEqual(expected, metrics['mean_interior_error_m'], places=6)


if __name__ == '__main__':
    unittest.main()
