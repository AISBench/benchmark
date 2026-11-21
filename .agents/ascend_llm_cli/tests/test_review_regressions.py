import unittest
from unittest.mock import patch
from deploy_eval_v2.evaluate import matches_prefix_target
from deploy_eval_v2.release import release_known


class ReviewRegressionTests(unittest.TestCase):
    def test_measured_lengths_and_minimum_prefix(self):
        case = dict(input_tokens=100, formal_requests=2, shared_prefix_ratio=0.8)
        for exact, common, valid in [(False, 0, False), (False, 80, False),
                                     (True, 79, False), (True, 80, True)]:
            observation = dict(status='MEASURED', exact_input_lengths=exact, common_prefix_tokens=common)
            self.assertEqual(matches_prefix_target(observation, case), valid)

    def test_cleanup_continues_after_failure(self):
        with patch('deploy_eval_v2.release.release_container',
                   side_effect=[RuntimeError(), dict(container_id='b', status='RELEASED_VERIFIED')]):
            results = release_known(['a', 'b'], 'task', 'token', None, 100)
        self.assertEqual([(r['container_id'], r['status']) for r in results],
                         [('a', 'UNVERIFIED'), ('b', 'RELEASED_VERIFIED')])
