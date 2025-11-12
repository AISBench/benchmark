import unittest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.livecodebench.pass_k_utils import (
        estimate_pass_at_k,
        compute_metrics_from_results,
        extract_instance_results
    )
    PASS_K_UTILS_AVAILABLE = True
except ImportError:
    PASS_K_UTILS_AVAILABLE = False


class PassKUtilsTestBase(unittest.TestCase):
    """PassKUtils测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not PASS_K_UTILS_AVAILABLE:
            cls.skipTest(cls, "PassKUtils modules not available")


class TestEstimatePassAtK(PassKUtilsTestBase):
    """测试estimate_pass_at_k函数"""
    
    def test_estimate_with_int_samples(self):
        """测试使用整数样本数估计pass@k"""
        num_samples = 10
        num_correct = [5, 7, 3]
        k = 1
        
        result = estimate_pass_at_k(num_samples, num_correct, k)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(num_correct))
        self.assertTrue(all(r >= 0 for r in result))
        self.assertTrue(all(r <= 100 for r in result))
    
    def test_estimate_with_list_samples(self):
        """测试使用列表样本数估计pass@k"""
        num_samples = [10, 20, 15]
        num_correct = [5, 10, 8]
        k = 1
        
        result = estimate_pass_at_k(num_samples, num_correct, k)
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result), len(num_correct))
    
    def test_estimate_when_n_minus_c_less_than_k(self):
        """测试当n-c < k时返回100"""
        num_samples = 5
        num_correct = [3, 4]
        k = 5
        
        result = estimate_pass_at_k(num_samples, num_correct, k)
        
        # 当n-c < k时，应该返回100
        self.assertTrue(all(r == 100.0 for r in result))


class TestComputeMetricsFromResults(PassKUtilsTestBase):
    """测试compute_metrics_from_results函数"""
    
    def test_compute_metrics_single_task(self):
        """测试计算单个任务的指标"""
        results = {
            0: [[True, True], [False, True]]
        }
        k_list = [1, 2]
        
        metrics = compute_metrics_from_results(results, k_list=k_list)
        
        self.assertIn('pass@1', metrics)
        self.assertIn('detail', metrics)
        self.assertIn('pass@1', metrics['detail'])
    
    def test_compute_metrics_multiple_tasks(self):
        """测试计算多个任务的指标"""
        results = {
            0: [[True, True], [False, True]],
            1: [[True, False], [True, True], [False, False]]
        }
        k_list = [1, 2]
        
        metrics = compute_metrics_from_results(results, k_list=k_list)
        
        self.assertIn('pass@1', metrics)
        self.assertIn('detail', metrics)
        self.assertIn('pass@1', metrics['detail'])
        self.assertIn(0, metrics['detail']['pass@1'])
        self.assertIn(1, metrics['detail']['pass@1'])
    
    def test_compute_metrics_with_all_passed(self):
        """测试所有测试用例都通过的情况"""
        results = {
            0: [[True, True], [True, True]]
        }
        k_list = [1, 2]
        
        metrics = compute_metrics_from_results(results, k_list=k_list)
        
        self.assertIn('pass@1', metrics)
        self.assertGreaterEqual(metrics['pass@1'], 0)
    
    def test_compute_metrics_with_all_failed(self):
        """测试所有测试用例都失败的情况"""
        results = {
            0: [[False, False], [False, False]]
        }
        k_list = [1, 2]
        
        metrics = compute_metrics_from_results(results, k_list=k_list)
        
        self.assertIn('pass@1', metrics)
        self.assertLessEqual(metrics['pass@1'], 100)


class TestExtractInstanceResults(PassKUtilsTestBase):
    """测试extract_instance_results函数"""
    
    def test_extract_single_task(self):
        """测试提取单个任务的结果"""
        results = {
            0: [[True, True], [False, True]]
        }
        
        instance_grades = extract_instance_results(results)
        
        self.assertIsInstance(instance_grades, list)
        self.assertEqual(len(instance_grades), 1)
        self.assertEqual(len(instance_grades[0]), 2)
        self.assertTrue(instance_grades[0][0])  # 第一个生成全部通过（all([True, True]) = True）
        self.assertFalse(instance_grades[0][1])  # 第二个生成未全部通过（all([False, True]) = False）
    
    def test_extract_multiple_tasks(self):
        """测试提取多个任务的结果"""
        results = {
            0: [[True, True], [False, True]],
            1: [[True, False], [True, True]]
        }
        
        instance_grades = extract_instance_results(results)
        
        self.assertIsInstance(instance_grades, list)
        self.assertEqual(len(instance_grades), 2)
        self.assertEqual(len(instance_grades[0]), 2)
        self.assertEqual(len(instance_grades[1]), 2)
    
    def test_extract_with_all_passed(self):
        """测试提取全部通过的结果"""
        results = {
            0: [[True, True], [True, True]]
        }
        
        instance_grades = extract_instance_results(results)
        
        # instance_grades[0] 是布尔值列表，每个布尔值表示一个generation是否全部通过
        self.assertTrue(all(instance_grades[0]))  # 所有generation都通过


if __name__ == '__main__':
    unittest.main()

