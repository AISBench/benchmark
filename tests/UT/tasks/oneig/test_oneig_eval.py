"""Unit tests for OneIGEvalTask (no real model/dataset inference)."""
import os.path as osp
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from mmengine.config import ConfigDict

from ais_bench.benchmark.tasks.oneig.oneig_eval import OneIGEvalTask


class TestOneIGEvalTask(unittest.TestCase):
    """Tests for OneIGEvalTask public API."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = ConfigDict({
            'work_dir': self.tmp,
            'models': [{'abbr': 'm_ab', 'type': 'stub'}],
            'datasets': [[
                {'abbr': 'ds1', 'type': 'stub', 'eval_cfg': {'num_gpus': 2}},
                {'abbr': 'ds2', 'type': 'stub', 'eval_cfg': {'num_gpus': 4}},
            ]],
            'cli_args': {},
        })
        self.out_path = osp.join(self.tmp, 'results', 'm1', 'ds1.json')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_oneig_eval_init_num_gpus(self, _log):
        """num_gpus should be the max across dataset_cfgs."""
        task = OneIGEvalTask(self.cfg)
        self.assertEqual(task.num_gpus, 4)

    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_oneig_eval_init_class_attributes(self, _log):
        """Class attributes should be set correctly."""
        self.assertEqual(OneIGEvalTask.name_prefix, 'OneIGEval')
        self.assertEqual(OneIGEvalTask.log_subdir, 'logs/eval')
        self.assertEqual(OneIGEvalTask.output_subdir, 'results')

    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_oneig_eval_get_command_format(self, _log):
        """get_command should return command with python, script, cfg_path."""
        cfg = ConfigDict({
            'work_dir': self.tmp,
            'models': [{'abbr': 'm_ab', 'type': 'stub'}],
            'datasets': [[{'abbr': 'ds1', 'type': 'stub', 'eval_cfg': {}}]],
            'cli_args': {},
        })
        task = OneIGEvalTask(cfg)
        cmd = task.get_command('/tmp/config.py', 'run {task_cmd}')
        self.assertIn(sys.executable, cmd)
        self.assertIn('/tmp/config.py', cmd)
        self.assertIn('oneig_eval.py', cmd)
        self.assertTrue(cmd.startswith('run '))

    def _setup_run_mocks(self, mock_build_ds, mock_icl, mock_out_path,
                         result=None):
        """Set up common mocks for run() tests."""
        mock_ds = MagicMock()
        mock_ds.test = [{'q': 'a'}]
        mock_build_ds.return_value = mock_ds

        mock_evaluator = MagicMock()
        mock_evaluator.score.return_value = result or {
            'accuracy': 0.9, 'details': []}
        mock_icl.build.return_value = mock_evaluator
        mock_out_path.return_value = self.out_path
        return mock_ds, mock_evaluator

    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.mkdir_or_exist')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.mmengine')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.task_abbr_from_cfg',
           return_value='m1/ds1')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.dataset_abbr_from_cfg',
           return_value='ds1')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.get_infer_output_path')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.ICL_EVALUATORS')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.build_dataset_from_cfg')
    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_oneig_eval_run_single_dataset(
            self, _log, mock_build_ds, mock_icl, mock_out_path,
            _mock_ds_abbr, _mock_task_abbr, mock_mmengine, _mock_mkdir):
        """run() should build dataset, run evaluator, and dump results."""
        cfg = ConfigDict({
            'work_dir': self.tmp,
            'models': [{'abbr': 'm1', 'type': 'stub'}],
            'datasets': [[
                {'abbr': 'ds1', 'type': 'stub', 'task_type': 'alignment',
                 'eval_cfg': {'evaluator': {'type': 'OneIGAlignmentEvaluator'}}},
            ]],
            'cli_args': {},
        })
        mock_ds, mock_evaluator = self._setup_run_mocks(
            mock_build_ds, mock_icl, mock_out_path)

        task = OneIGEvalTask(cfg)
        task.run(None)

        mock_build_ds.assert_called_once()
        mock_icl.build.assert_called_once_with(
            {'type': 'OneIGAlignmentEvaluator'})
        mock_evaluator.score.assert_called_once_with(
            predictions=[], references=[], test_set=mock_ds.test)
        mock_mmengine.dump.assert_called_once()

    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.mkdir_or_exist')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.mmengine')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.task_abbr_from_cfg',
           return_value='m1/ds1')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.dataset_abbr_from_cfg',
           return_value='ds1')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.get_infer_output_path')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.ICL_EVALUATORS')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.build_dataset_from_cfg')
    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_oneig_eval_run_multiple_datasets(
            self, _log, mock_build_ds, mock_icl, mock_out_path,
            _mock_ds_abbr, _mock_task_abbr, mock_mmengine, _mock_mkdir):
        """run() should handle multiple datasets."""
        cfg = ConfigDict({
            'work_dir': self.tmp,
            'models': [{'abbr': 'm1', 'type': 'stub'}],
            'datasets': [[
                {'abbr': 'ds1', 'type': 'stub', 'task_type': 'alignment',
                 'eval_cfg': {'evaluator': {'type': 'OneIGAlignmentEvaluator'}}},
                {'abbr': 'ds2', 'type': 'stub', 'task_type': 'diversity',
                 'eval_cfg': {'evaluator': {'type': 'OneIGDiversityEvaluator'}}},
            ]],
            'cli_args': {},
        })
        mock_ds = MagicMock()
        mock_ds.test = [{'q': 'a'}]
        mock_build_ds.return_value = mock_ds

        mock_evaluator = MagicMock()
        mock_evaluator.score.side_effect = [
            {'accuracy': 0.9, 'details': []},
            {'accuracy': 0.8, 'details': []},
        ]
        mock_icl.build.return_value = mock_evaluator

        out_path1 = osp.join(self.tmp, 'results', 'm1', 'ds1.json')
        out_path2 = osp.join(self.tmp, 'results', 'm1', 'ds2.json')
        mock_out_path.side_effect = [out_path1, out_path2]

        task = OneIGEvalTask(cfg)
        task.run(None)

        self.assertEqual(mock_build_ds.call_count, 2)
        self.assertEqual(mock_icl.build.call_count, 2)
        self.assertEqual(mock_evaluator.score.call_count, 2)
        self.assertEqual(mock_mmengine.dump.call_count, 2)

    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.mkdir_or_exist')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.mmengine')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.task_abbr_from_cfg',
           return_value='m1/ds1')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.dataset_abbr_from_cfg',
           return_value='ds1')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.get_infer_output_path')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.ICL_EVALUATORS')
    @patch('ais_bench.benchmark.tasks.oneig.oneig_eval.build_dataset_from_cfg')
    @patch('ais_bench.benchmark.tasks.base.AISLogger')
    def test_oneig_eval_run_saves_results(
            self, _log, mock_build_ds, mock_icl, mock_out_path,
            _mock_ds_abbr, _mock_task_abbr, mock_mmengine, _mock_mkdir):
        """run() should save results with correct format."""
        cfg = ConfigDict({
            'work_dir': self.tmp,
            'models': [{'abbr': 'm1', 'type': 'stub'}],
            'datasets': [[
                {'abbr': 'ds1', 'type': 'stub', 'task_type': 'alignment',
                 'eval_cfg': {'evaluator': {'type': 'OneIGAlignmentEvaluator'}}},
            ]],
            'cli_args': {},
        })
        result = {'accuracy': 0.9, 'details': [{'score': 1.0}]}
        self._setup_run_mocks(
            mock_build_ds, mock_icl, mock_out_path, result=result)

        task = OneIGEvalTask(cfg)
        task.run(None)

        mock_mmengine.dump.assert_called_once_with(
            result, self.out_path, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    unittest.main()
