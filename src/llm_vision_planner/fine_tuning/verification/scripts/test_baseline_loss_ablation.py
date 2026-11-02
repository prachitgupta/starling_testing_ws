"""Checks for the independent eight-waypoint GPU-server study."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_baseline_loss_ablation as study
from test_loss_ablation import TinyEncoder, TinyTokenizer, fixture_config


class BaselineStudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = fixture_config(self.root)
        self.config['model'].pop('fixed_endpoints')
        self.config['training'].update(num_workers=0, seed=42)
        self.config['tokenizer_name'] = str(self.root / 'encoder')
        self.encoder = patch('transformers.AutoModel.from_pretrained', side_effect=lambda *a, **k: TinyEncoder())
        self.tokenizer = patch('transformers.AutoTokenizer.from_pretrained', return_value=TinyTokenizer())
        self.encoder.start(); self.tokenizer.start()

    def tearDown(self):
        study.support.STOP_REQUESTED = False
        self.encoder.stop(); self.tokenizer.stop(); self.tmp.cleanup()

    def test_all_eight_loss_and_endpoint_gradients(self):
        predicted = torch.zeros(1, 8, 2, requires_grad=True)
        target = torch.zeros_like(predicted)
        target[:, 0, 0] = 2.; target[:, -1, 0] = 4.
        batch = dict(target_xy=target)
        loss, terms = study.objective(predicted, batch, study.VARIANTS['E0'], study.LOSS_CONFIG)
        self.assertAlmostEqual(float(loss), .75)
        loss.backward()
        self.assertEqual(float(predicted.grad[0, 0, 0]), -.125)
        self.assertEqual(float(predicted.grad[0, -1, 0]), -.125)
        e1, terms = study.objective(predicted, batch, study.VARIANTS['E1'], study.LOSS_CONFIG)
        self.assertAlmostEqual(float(terms['squared']), 2.5)
        self.assertAlmostEqual(float(e1), .75 + .25 * 2.5)
        with patch.object(study.support.v2, 'corridor_side_loss', return_value=torch.tensor(.3)):
            for variant, multiplier in [('E2', 1), ('E3', 5)]:
                loss, _ = study.objective(predicted, dict(batch, examples=[]), study.VARIANTS[variant], study.LOSS_CONFIG)
                self.assertAlmostEqual(float(loss), .75 + multiplier * .3, places=6)

    def test_historical_settings_preserved(self):
        config = copy.deepcopy(self.config)
        config['training'].update(planner_learning_rate=.0001, text_encoder_learning_rate=.00002,
            batch_size=32, validation_batch_size=64, num_workers=4, max_epochs=100)
        manifest = dict(config=config)
        result = study.effective_config(manifest, study.VARIANTS['E3'], 43, self.root / 'run')
        for key, value in config['training'].items():
            if key not in ('seed', 'output_dir'):
                self.assertEqual(result['training'][key], value)
        self.assertFalse(result['model'].get('fixed_endpoints', False))
        self.assertEqual(result['model'], config['model'])
        self.assertEqual(result['loss']['corridor_weight'], 5.)
        self.assertEqual(config, manifest['config'])

    def test_plan_never_screens_v2_or_uses_test_or_reference_to_select_loss(self):
        manifest = dict(seeds=[42, 43, 44], minimum_improvement_m=.02)
        results, seen = {}, []
        for _ in range(16):
            task = study.plan(manifest, results); seen.append(task['id'])
            if task['kind'] == 'final':
                break
            value = .001 if task['kind'] == 'baseline' else dict(E0=.5, E1=.46, E2=.45, E3=.51, E4=.4)[task['variant']]
            results[task['id']] = dict(metrics=dict(mean_interior_error_m=value))
        self.assertEqual(task['winner_variant'], 'E4')
        self.assertIn('v1_E0_s42', seen)
        self.assertIn('v1_E4_s42', seen)
        self.assertFalse(any('v2' in key for key in seen))
        results['test'] = dict(metrics=dict(mean_interior_error_m=-1000))
        results['baseline']['metrics']['mean_interior_error_m'] = 1000
        self.assertEqual(task, study.plan(manifest, results))
        for key in results:
            if key.startswith('v1_'):
                results[key]['metrics']['mean_interior_error_m'] = .5
        for seed in manifest['seeds']:
            results[f'v1_E1_s{seed}'] = dict(metrics=dict(mean_interior_error_m=.5))
        self.assertEqual(study.plan(manifest, results)['winner_variant'], 'E0')

    def test_eight_queries_and_historical_training_update(self):
        """Compare one epoch to the original fixed-divisor accumulation algorithm."""
        cfg = copy.deepcopy(self.config)
        cfg['training']['max_epochs'] = 1
        s = study.support
        s.v1.set_seed(42)
        examples = s.examples_for(cfg['data']['train_csv'], cfg)
        loader = study.loader_for(examples, TinyTokenizer(), cfg, True)
        model = s.v1.build_model(cfg['model'])
        self.assertEqual(model.route_queries.shape[0], 8)
        optimizer = s.verify.make_optimizer(model, cfg['training'])
        scheduler = s.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=1, num_training_steps=2)
        model.train(); optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, 1):
            predicted = s.forward(model, batch)
            self.assertFalse(torch.allclose(predicted[:, [0, -1]], batch['target_xy'][:, [0, -1]]))
            loss = s.v1.euclidean_imitation_loss(predicted, batch['target_xy']) / 2
            loss.backward()
            if step % 2 == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_gradient_norm'])
                optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step()
        with contextlib.redirect_stdout(io.StringIO()):
            study.train_run(self.root / 'parity', cfg, study.VARIANTS['E0'], 42, torch.device('cpu'))
        saved = torch.load(self.root / 'parity/resume.pt', weights_only=False)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, saved['model'][key], rtol=0, atol=0)

    def test_epoch_resume_matches_uninterrupted_training(self):
        def interrupt(history):
            if len(history) == 1:
                raise study.support.ResumeLater()
        for workers in (0, 2):
            cfg = copy.deepcopy(self.config); cfg['training']['num_workers'] = workers
            for name in ('full', 'resumed'):
                path = self.root / f'{name}-{workers}'
                with contextlib.redirect_stdout(io.StringIO()):
                    if name == 'resumed':
                        with self.assertRaises(study.support.ResumeLater):
                            study.train_run(path, cfg, study.VARIANTS['E1'], 42, torch.device('cpu'), interrupt)
                    study.train_run(path, cfg, study.VARIANTS['E1'], 42, torch.device('cpu'))
            full = torch.load(self.root / f'full-{workers}/resume.pt', weights_only=False)
            resumed = torch.load(self.root / f'resumed-{workers}/resume.pt', weights_only=False)
            for key in full['model']:
                torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
            self.assertEqual(full['history'][-1]['validation'], resumed['history'][-1]['validation'])
            with self.assertRaisesRegex(ValueError, 'Resume configuration differs'):
                study.train_run(self.root / f'full-{workers}', cfg, study.VARIANTS['E0'], 42, torch.device('cpu'))

    def test_dashboard_is_explicit_about_eight_outputs(self):
        s = study.support
        s.save_json(self.root / 'manifest.json', dict(smoke=True, seeds=[42, 43, 44], minimum_improvement_m=.02))
        s.save_json(self.root / 'state.json', dict(status='ready', task=dict(id='baseline')))
        study.render(self.root)
        html = (self.root / 'dashboard/index.html').read_text()
        self.assertIn('All runs predict all 8 points', html)
        self.assertIn('Training all-8 Euclidean', html)
        self.assertIn('Validation interior episode P80', html)
        self.assertNotIn('All fresh runs fix endpoints', html)
        self.assertIn('Same scene, different preferences', html)

    def test_reinitializing_does_not_overwrite_existing_state(self):
        study.support.save_json(self.root / 'manifest.json', dict(study_kind=study.STUDY_KIND))
        original = dict(status='complete', results={'baseline': {}}, task={'id': 'final'})
        study.support.save_json(self.root / 'state.json', original)
        with patch('sys.argv', ['run_baseline_loss_ablation.py', 'init', '--study-root', str(self.root)]):
            # Avoid changing the test process's signal handling.
            with patch.object(study.signal, 'signal'), self.assertRaisesRegex(ValueError, 'Study exists'):
                study.main()
        self.assertEqual(json.loads((self.root / 'state.json').read_text()), original)

    def test_downloaded_ncsa_dashboard_is_protected(self):
        # The laptop copy may have state/dashboard but no manifest.
        output = self.root / 'ncsa-download'; output.mkdir()
        dashboard = output / 'dashboard'; dashboard.mkdir()
        (dashboard / 'index.html').write_text('existing NCSA results')
        (output / 'state.json').write_text('{"status":"running"}')
        with patch('sys.argv', ['run_baseline_loss_ablation.py', 'init', '--study-root', str(output)]):
            with patch.object(study.signal, 'signal'), self.assertRaisesRegex(ValueError, 'not empty'):
                study.main()
        self.assertEqual((dashboard / 'index.html').read_text(), 'existing NCSA results')
        self.assertEqual((output / 'state.json').read_text(), '{"status":"running"}')
        self.assertFalse((output / 'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
