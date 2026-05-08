"""Unit tests for VBenchEvalTask (no real vbench/torch inference)."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from mmengine.config import ConfigDict

from ais_bench.benchmark.tasks.vbench_eval import VBenchEvalTask


@contextmanager
def _fake_vbench_imports(mock_torch_device=None):
    """Inject mock vbench / vbench.distributed / torch into sys.modules for ``run()``."""
    mock_vmod = MagicMock()
    mock_instance = MagicMock()
    mock_instance.build_full_dimension_list.return_value = [
        'subject_consistency',
        'aesthetic_quality',
    ]
    mock_instance.evaluate.return_value = {'subject_consistency': {'score': 0.75}}
    mock_vmod.VBench.return_value = mock_instance
    mock_vmod.set_progress_callback = MagicMock()

    mock_dist = MagicMock()
    mock_dist.dist_init = MagicMock()
    mock_dist.get_rank = MagicMock(return_value=0)
    mock_dist.get_device = MagicMock(return_value='cpu')
    mock_dist.dist_destroy = MagicMock()

    mock_torch = MagicMock()
    dev = MagicMock()

    def _device(_x):
        return dev

    mock_torch.device = _device if mock_torch_device is None else mock_torch_device

    prev = {}
    for name in ('vbench', 'vbench.distributed', 'torch'):
        prev[name] = sys.modules.pop(name, None)
    sys.modules['vbench'] = mock_vmod
    sys.modules['vbench.distributed'] = mock_dist
    sys.modules['torch'] = mock_torch
    try:
        yield mock_vmod, mock_dist, mock_torch, mock_instance
    finally:
        for name in ('vbench', 'vbench.distributed', 'torch'):
            sys.modules.pop(name, None)
        for name, mod in prev.items():
            if mod is not None:
                sys.modules[name] = mod


class TestVBenchEvalTaskHelpers(unittest.TestCase):
    """Tests for helpers that need no mocked heavy imports."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = ConfigDict({
            'work_dir': self.tmp,
            'models': [{'abbr': 'm_ab', 'type': 'stub'}],
            'datasets': [[{
                'abbr': 'ds_ab',
                'path': '/tmp/videos_placeholder',
                'eval_cfg': {},
            }]],
            'cli_args': {},
        })

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_infer_mode_explicit(self, _log):
        task = VBenchEvalTask(self.cfg)
        ds = {'path': '/x'}
        ev = {'mode': 'vbench_standard'}
        self.assertEqual(task._infer_mode(ds, ev), 'vbench_standard')

    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_infer_mode_category(self, _log):
        task = VBenchEvalTask(self.cfg)
        ds = {'path': '/x'}
        ev = {'category': 'motion'}
        self.assertEqual(task._infer_mode(ds, ev), 'vbench_category')

    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_infer_mode_prompt_sources(self, _log):
        task = VBenchEvalTask(self.cfg)
        ds = {'path': '/x', 'prompt_list': [{'a': 1}]}
        self.assertEqual(task._infer_mode(ds, {}), 'custom_input')
        ds2 = {'path': '/x'}
        ev2 = {'prompt_list': [{}]}
        self.assertEqual(task._infer_mode(ds2, ev2), 'custom_input')
        ev3 = {'prompt_file': '/p.json'}
        self.assertEqual(task._infer_mode(ds2, ev3), 'custom_input')

    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_infer_mode_default_standard(self, _log):
        task = VBenchEvalTask(self.cfg)
        self.assertEqual(task._infer_mode({'path': '/x'}, {}), 'vbench_standard')

    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_wrap_results_dict_score_and_tuple(self, _log):
        task = VBenchEvalTask(self.cfg)
        raw = {
            'd1': {'score': 0.5},
            'd2': {'mean_score': 0.5},
            'd3': (0.4, [{'v': 1}]),
            'd4': 'no_score',
        }
        out = task._wrap_results(raw)
        self.assertAlmostEqual(out['accuracy'], 46.666666666666664, places=10)
        self.assertIn('d1', out['details'])
        self.assertEqual(out['details']['d3']['video_results'], [{'v': 1}])

    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_wrap_results_empty_scores_accuracy_zero(self, _log):
        task = VBenchEvalTask(self.cfg)
        out = task._wrap_results({'only': {'value': 'x'}})
        self.assertEqual(out['accuracy'], 0.0)

    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    @patch.dict(os.environ, {}, clear=False)
    def test_apply_cache_dir_priority_and_expand(self, _mock_log):
        task = VBenchEvalTask(self.cfg)
        with patch.dict(os.environ, {}, clear=True):
            task.cfg['VBENCH_CACHE_DIR'] = '/wins_when_both_set'
            task.cfg['vbench_cache_dir'] = '~/vb_cache'
            task._apply_vbench_cache_dir_from_cfg()
            self.assertEqual(os.environ['VBENCH_CACHE_DIR'], '/wins_when_both_set')
        with patch.dict(os.environ, {}, clear=True):
            task.cfg.pop('VBENCH_CACHE_DIR', None)
            task.cfg['vbench_cache_dir'] = '~/vb_cache'
            with patch.object(os.path, 'expanduser', side_effect=lambda p: p.replace('~/', '/home/me/')):
                with patch.object(os.path, 'expandvars', side_effect=lambda p: p):
                    task._apply_vbench_cache_dir_from_cfg()
            self.assertEqual(os.environ['VBENCH_CACHE_DIR'], '/home/me/vb_cache')

    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    @patch.dict(os.environ, {}, clear=False)
    def test_apply_cache_dir_skips_empty(self, _mock_log):
        task = VBenchEvalTask(self.cfg)
        with patch.dict(os.environ, {}, clear=True):
            task.cfg.pop('VBENCH_CACHE_DIR', None)
            task.cfg.pop('vbench_cache_dir', None)
            task._apply_vbench_cache_dir_from_cfg()
            self.assertNotIn('VBENCH_CACHE_DIR', os.environ)
        with patch.dict(os.environ, {}, clear=True):
            task.cfg['VBENCH_CACHE_DIR'] = '   '
            task._apply_vbench_cache_dir_from_cfg()
            self.assertNotIn('VBENCH_CACHE_DIR', os.environ)

    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    @patch('ais_bench.benchmark.tasks.vbench_eval.get_infer_output_path')
    def test_get_output_paths(self, mock_path, _log):
        mock_path.side_effect = lambda m, d, base, ext='json': f'{base}/{m["abbr"]}_{d["abbr"]}.{ext}'
        task = VBenchEvalTask(self.cfg)
        paths = task.get_output_paths()
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].endswith('m_ab_ds_ab.json'))


class TestVBenchEvalTaskRunMocked(unittest.TestCase):
    """``run()`` with sys.modules mocks."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.videos = os.path.join(self.tmp, 'vids')
        os.makedirs(self.videos, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _task_cfg(self, **dataset_overrides):
        ds = {
            'abbr': 'vb_ds',
            'path': self.videos,
            'eval_cfg': {},
        }
        ds.update(dataset_overrides)
        return ConfigDict({
            'work_dir': self.tmp,
            'models': [{'abbr': 'm1', 'type': 'stub'}],
            'datasets': [[ds]],
            'cli_args': {},
        })

    @patch('ais_bench.benchmark.tasks.vbench_eval.osp.isfile', return_value=True)
    @patch('ais_bench.benchmark.tasks.vbench_eval.osp.isdir', return_value=True)
    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_run_calls_build_full_dimension_list_when_missing(self, _log, _isdir, _isfile):
        cfg = self._task_cfg()
        task = VBenchEvalTask(cfg)
        with _fake_vbench_imports() as (mock_vmod, mock_dist, _t, mock_inst):
            task.run(None)
            mock_inst.build_full_dimension_list.assert_called_once()
            mock_inst.evaluate.assert_called_once()
            mock_dist.dist_destroy.assert_called()

        out_json = os.path.join(
            self.tmp, task.output_subdir, 'm1', 'vb_ds.json',
        )
        self.assertTrue(os.path.isfile(out_json))
        with open(out_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.assertIn('accuracy', data)
        self.assertIn('details', data)

    @patch('ais_bench.benchmark.tasks.vbench_eval.osp.isfile', return_value=True)
    @patch('ais_bench.benchmark.tasks.vbench_eval.osp.isdir', return_value=True)
    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_run_prompt_file_not_dict_raises_valueerror(self, _log, _isdir, _isfile):
        bad_json = os.path.join(self.tmp, 'prompts_bad.json')
        with open(bad_json, 'w', encoding='utf-8') as f:
            json.dump([], f)

        cfg = self._task_cfg(
            eval_cfg={'prompt_file': bad_json},
        )
        task = VBenchEvalTask(cfg)
        with _fake_vbench_imports():
            with self.assertRaises(ValueError) as ctx:
                task.run(None)
            self.assertIn('prompt_file', str(ctx.exception))

    @patch('ais_bench.benchmark.tasks.vbench_eval.osp.isfile', return_value=True)
    @patch('ais_bench.benchmark.tasks.vbench_eval.osp.isdir', return_value=True)
    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_run_registers_progress_callback_with_state_manager(self, _log, _isdir, _isfile):
        cfg = self._task_cfg()
        task = VBenchEvalTask(cfg)
        tsm = MagicMock()
        with _fake_vbench_imports() as (mock_vmod, *_):
            task.run(tsm)
            mock_vmod.set_progress_callback.assert_called_once()


if __name__ == '__main__':
    unittest.main()
