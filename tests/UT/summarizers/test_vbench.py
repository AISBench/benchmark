"""Unit tests for VBenchSummarizer and helper functions."""
import unittest
from unittest.mock import patch

from mmengine.config import ConfigDict

import ais_bench.benchmark.summarizers.vbench as vbench_mod
from ais_bench.benchmark.summarizers.vbench import (
    VBenchSummarizer,
    _abbr_to_const_key,
    _get_normalized_score,
)


class TestVBenchHelpers(unittest.TestCase):

    def test_abbr_from_regex(self):
        self.assertEqual(
            _abbr_to_const_key('vbench_custom_subject_consistency'),
            'subject consistency',
        )

    def test_abbr_fallback_strip_prefix(self):
        self.assertEqual(
            _abbr_to_const_key('vbench_foo_bar'),
            'foo bar',
        )

    def test_abbr_no_prefix(self):
        self.assertEqual(_abbr_to_const_key('other_key'), 'other key')

    def test_get_normalized_score_divides_when_above_one(self):
        s = _get_normalized_score(100.0, 'subject consistency')
        self.assertGreater(s, 0)
        same_as_one = _get_normalized_score(1.0, 'subject consistency')
        self.assertAlmostEqual(s, same_as_one, places=8)

    def test_get_normalized_score_unknown_dim(self):
        self.assertEqual(_get_normalized_score(0.99, 'not a dimension'), 0.0)

    def test_get_normalized_score_zero_span_branch(self):
        fake_norm = dict(vbench_mod.NORMALIZE_DIC)
        fake_norm['zero_span_test'] = {'Min': 0.5, 'Max': 0.5}
        fake_w = dict(vbench_mod.DIM_WEIGHT)
        fake_w['zero_span_test'] = 2.0
        with patch.multiple(vbench_mod, NORMALIZE_DIC=fake_norm, DIM_WEIGHT=fake_w):
            high = _get_normalized_score(0.7, 'zero_span_test')
            low = _get_normalized_score(0.3, 'zero_span_test')
            self.assertEqual(high, 2.0)
            self.assertEqual(low, 0.0)


class TestVBenchSummarizer(unittest.TestCase):

    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_calculate_group_metrics_writes_aggregate_scores(self, _logger):
        cfg = ConfigDict({
            'models': [{'abbr': 'model_a'}],
            'datasets': [{'abbr': 'd1'}],
            'work_dir': '/tmp',
        })
        summ = VBenchSummarizer(cfg, dataset_abbrs=['d1'], summary_groups=[])

        raw_results = {'model_a': {}}
        parsed_results = {
            'model_a': {
                'vbench_run_subject_consistency': {'accuracy': 100.0},
                'vbench_run_object_class': {'accuracy': 100.0},
            },
        }
        dataset_metrics = {}
        dataset_eval_mode = {}

        raw_results, parsed_results, dm, dem = summ._calculate_group_metrics(
            raw_results, parsed_results, dataset_metrics, dataset_eval_mode,
        )

        self.assertIn('vbench_quality', parsed_results['model_a'])
        self.assertIn('vbench_semantic', parsed_results['model_a'])
        self.assertIn('vbench_total', parsed_results['model_a'])
        for name in ('vbench_quality', 'vbench_semantic', 'vbench_total'):
            self.assertAlmostEqual(
                raw_results['model_a'][name]['accuracy'],
                parsed_results['model_a'][name]['accuracy'],
            )
            self.assertIn(name, dm)
            self.assertEqual(dm[name], ['accuracy'])
            self.assertEqual(dem[name], 'gen')

    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    def test_calculate_group_metrics_skips_non_vbench_abbr(self, _logger):
        cfg = ConfigDict({
            'models': [{'abbr': 'm'}],
            'datasets': [{'abbr': 'd1'}],
            'work_dir': '/tmp',
        })
        summ = VBenchSummarizer(cfg)
        parsed_results = {'m': {'plain_ds': {'accuracy': 55.0}}}
        raw_results = {'m': {}}
        _, pr, _, __ = summ._calculate_group_metrics(
            raw_results, parsed_results.copy(), {}, {},
        )
        self.assertNotIn('vbench_total', pr['m'])


if __name__ == '__main__':
    unittest.main()
