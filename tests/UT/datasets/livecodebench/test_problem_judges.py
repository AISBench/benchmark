"""Regression cases for tasks that require more than reference-output equality."""

import importlib.util
import json
import unittest
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[4] / 'ais_bench' / 'benchmark'
          / 'datasets' / 'livecodebench' / 'problem_judges.py')
SPEC = importlib.util.spec_from_file_location('problem_judges', SOURCE)
judges = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(judges)


class TestProblemJudges(unittest.TestCase):

    def test_abc392_f_excludes_invalid_inputs_not_entire_problem(self):
        sample = json.dumps({
            'inputs': [
                '4\n1 1 2 1', '6\n3 3 2 5 4 6', '5\n5 4 3 2 1',
            ],
            'outputs': [
                '4 2 3 1', '0 3 0 5 2 1', '5 0 4 0 1',
            ],
            'fn_name': None,
        })
        prepared, excluded = judges.prepare_test_cases('abc392_f', sample)
        self.assertEqual(excluded, [1, 2])
        self.assertEqual(json.loads(prepared)['inputs'], ['4\n1 1 2 1'])
        self.assertEqual(json.loads(prepared)['outputs'], ['4 2 3 1'])

    def test_abc392_f_rejects_empty_or_unpaired_tests(self):
        with self.assertRaises(ValueError):
            judges.prepare_test_cases('abc392_f', json.dumps({
                'inputs': ['6\n3 3 2 5 4 6'], 'outputs': ['0 3 0 5 2 1'],
            }))
        with self.assertRaises(ValueError):
            judges.prepare_test_cases('abc392_f', json.dumps({
                'inputs': ['1\n1'], 'outputs': [],
            }))

    def test_abc397_d_positive_integer_solutions(self):
        self.assertTrue(judges.judge_stdio('abc397_d', '27', '-1'))
        self.assertFalse(judges.judge_stdio('abc397_d', '27', '3 0'))
        self.assertTrue(judges.judge_stdio('abc397_d', '397', '12 11'))
        self.assertFalse(judges.judge_stdio('abc397_d', '397', '-1'))

    def test_arc191_c_accepts_another_valid_construction(self):
        self.assertTrue(judges.judge_stdio('arc191_c', '1\n3', '2 7'))
        self.assertTrue(judges.judge_stdio('arc191_c', '1\n3', '4 9'))
        self.assertFalse(judges.judge_stdio('arc191_c', '1\n3', '2 1'))
        self.assertFalse(judges.judge_stdio('arc191_c', '1\n3', '4 9 99'))

    def test_arc195_c_validates_all_piece_moves(self):
        self.assertTrue(judges.judge_stdio(
            'arc195_c', '1\n2 0', 'Yes\nR 1 1\nR 1 2'))
        self.assertFalse(judges.judge_stdio(
            'arc195_c', '1\n2 0', 'Yes\nR 1 1\nR 2 2'))
        self.assertFalse(judges.judge_stdio('arc195_c', '1\n2 0', 'No'))
        self.assertTrue(judges.judge_stdio('arc195_c', '1\n1 1', 'No'))

    def test_3763_compares_with_true_area_bisector(self):
        inputs = [[0, 0, 2], [1, 1, 1]]
        self.assertTrue(judges.judge_call('3763', [inputs], 7 / 6))
        self.assertTrue(judges.judge_call('3763', [inputs], 1.16667))
        self.assertFalse(judges.judge_call('3763', [inputs], 1.167))
        self.assertTrue(judges.judge_call(
            '3763', [[[0, 0, 1], [2, 2, 1]]], 1))

    def test_other_tasks_keep_existing_comparison(self):
        self.assertIsNone(judges.judge_stdio('abc396_e', '1', '-1'))
        self.assertIsNone(judges.judge_call('abc396_e', [], 1))
        sample = '{"inputs": ["1"], "outputs": ["2"], "fn_name": null}'
        self.assertEqual(judges.prepare_test_cases('abc396_e', sample),
                         (sample, []))


if __name__ == '__main__':
    unittest.main()
