import unittest
from unittest import mock

import numpy as np
from datasets import Dataset

from ais_bench.benchmark.openicl.icl_evaluator.icl_base_evaluator import (
    BaseEvaluator,
    compute_pass_at_k,
    compute_g_pass_at_k,
)
from ais_bench.benchmark.utils.logging.exceptions import PredictionInvalidException, AISBenchImplementationError


class DummyEvaluator(BaseEvaluator):
    def __init__(self, result_details=None, result_metrics=None):
        super().__init__()
        self._result_details = result_details or []
        self._result_metrics = result_metrics or {"pass@1": 0.5}

    def score(self, **kwargs):
        return {**self._result_metrics, "details": self._result_details}


class TestComputeFunctions(unittest.TestCase):
    def test_compute_pass_at_k_basic(self):
        """测试compute_pass_at_k函数的基本计算功能"""
        self.assertAlmostEqual(compute_pass_at_k(n=5, c=3, k=1), 1.0 - np.prod(1.0 - 1 / np.arange(3, 6)))

    def test_compute_pass_at_k_shortcut(self):
        """测试compute_pass_at_k函数在n-c<k时的快捷返回路径"""
        self.assertEqual(compute_pass_at_k(n=3, c=2, k=2), 1.0)

    def test_compute_g_pass_at_k_invalid(self):
        """测试compute_g_pass_at_k函数在无效边界条件下返回0.0"""
        self.assertEqual(compute_g_pass_at_k(n=0, c=0, k=1, t=0.5), 0.0)
        self.assertEqual(compute_g_pass_at_k(n=10, c=-1, k=1, t=0.5), 0.0)

    def test_compute_g_pass_at_k_valid(self):
        """测试compute_g_pass_at_k函数在有效参数下的计算结果在0-1范围内"""
        v = compute_g_pass_at_k(n=10, c=3, k=5, t=0.4)
        self.assertTrue(0.0 <= v <= 1.0)


class TestBaseEvaluatorGroupReduceEvaluate(unittest.TestCase):
    def test_group_success_and_mismatch(self):
        """测试BaseEvaluator的group方法成功分组和不匹配时抛出异常"""
        evaluator = BaseEvaluator()
        details = [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]
        test_set = Dataset.from_dict({
            "subdivision": ["cat", "cat", "dog", "dog"],
            "idx": [0, 0, 1, 1],
        })

        grouped = evaluator.group(n=2, details=details, test_set=test_set)
        self.assertIn("cat_0", grouped)
        self.assertIn("dog_1", grouped)
        self.assertEqual(len(grouped["cat_0"]), 2)

        with self.assertRaises(PredictionInvalidException):
            evaluator.group(n=3, details=details, test_set=test_set)

    def test_reduce_with_categories_and_extra_fields(self):
        """测试BaseEvaluator的reduce方法对分类指标和额外字段的聚合处理"""
        evaluator = BaseEvaluator()
        details = [
            {
                "example_abbr": "cat_0",
                "avg@2": 0.5,
                "pass@2": 0.6,
                "cons@2": 1.0,
                "custom_numeric": 0.4,
                "custom_list": [1, 2],
            },
            {
                "example_abbr": "dog_1",
                "avg@2": 1.0,
                "pass@2": 0.2,
                "cons@2": 0.0,
                "custom_numeric": 0.6,
                "custom_list": [3],
            },
        ]
        results = evaluator.reduce(details=details, k_list=[2], n_val=2)
        self.assertAlmostEqual(results["avg@2"], 100 * np.mean([0.5, 1.0]))
        self.assertAlmostEqual(results["pass@2"], 100 * np.mean([0.6, 0.2]))
        self.assertIn("cat/avg@2", results)
        self.assertIn("dog/cons@2", results)
        self.assertAlmostEqual(results["custom_numeric"], 100 * 0.5)
        self.assertIn("cat/custom_numeric", results)
        self.assertIsInstance(results["custom_list"], list)

    def test_reduce_without_example_abbr(self):
        """测试BaseEvaluator的reduce方法在details缺少example_abbr字段时的处理"""
        evaluator = BaseEvaluator()
        details = [
            {"avg@2": 0.5, "pass@2": 0.6, "cons@2": 1.0},
            {"example_abbr": "cat_0", "avg@2": 1.0, "pass@2": 0.2, "cons@2": 0.0},
        ]
        results = evaluator.reduce(details=details, k_list=[2], n_val=2)
        self.assertIn("avg@2", results)

    def test_reduce_typeerror_in_category_aggregation(self):
        """测试BaseEvaluator的reduce方法在分类聚合时遇到TypeError的处理"""
        evaluator = BaseEvaluator()
        details = [
            {"example_abbr": "cat_0", "avg@2": 0.5, "pass@2": 0.6, "cons@2": 1.0, "bad_field": "not_numeric"},
            {"example_abbr": "cat_0", "avg@2": 1.0, "pass@2": 0.2, "cons@2": 0.0, "bad_field": "also_not_numeric"},
        ]
        results = evaluator.reduce(details=details, k_list=[2], n_val=2)
        self.assertIn("cat/bad_field", results)

    def test_reduce_single_category(self):
        """测试BaseEvaluator的reduce方法在单一分类时的处理"""
        evaluator = BaseEvaluator()
        details = [
            {"example_abbr": "cat_0", "avg@2": 0.5, "pass@2": 0.6, "cons@2": 1.0},
            {"example_abbr": "cat_1", "avg@2": 1.0, "pass@2": 0.2, "cons@2": 0.0},
        ]
        results = evaluator.reduce(details=details, k_list=[2], n_val=2)
        self.assertIn("avg@2", results)

    def test_dataset_replica_idx_property(self):
        """测试BaseEvaluator的dataset_replica_idx属性"""
        evaluator = BaseEvaluator()
        self.assertEqual(evaluator.dataset_replica_idx, 0)
        evaluator._dataset_replica_idx = 2
        self.assertEqual(evaluator.dataset_replica_idx, 2)

    def test_pred_postprocess_noop_and_with_proc(self):
        """测试BaseEvaluator的pred_postprocess方法在无处理器和有处理器时的行为"""
        evaluator = BaseEvaluator()
        preds = ["a", "b"]
        self.assertEqual(evaluator.pred_postprocess(preds), preds)

        evaluator.pred_postprocessor = {"type": "dummy", "lower": True}
        with mock.patch(
            "ais_bench.benchmark.openicl.icl_evaluator.icl_base_evaluator.TEXT_POSTPROCESSORS.get",
            return_value=lambda s, lower=False: s.lower() if lower else s,
        ):
            out = evaluator.pred_postprocess(["AbC", "DEF"])
            self.assertEqual(out, ["abc", "def"])

    def test_evaluate_pipeline_with_details_and_aggregation(self):
        """测试BaseEvaluator的evaluate方法在有多副本和聚合场景下的完整流程"""
        from ais_bench.benchmark.openicl.icl_evaluator.icl_hf_evaluator import AccwithDetailsEvaluator

        preds = ["A", "B", "A", "B"]
        refs = ["A", "B", "X", "B"]
        origin_prompt = ["p0", "p1", "p0", "p1"]
        test_set = Dataset.from_dict({
            "subdivision": ["cat", "cat", "cat", "cat"],
            "idx": [0, 0, 1, 1],
        })

        evaluator = AccwithDetailsEvaluator()
        results = evaluator.evaluate(k=[2, 3], n=2, original_dataset=test_set,
                                     predictions=preds, references=refs, origin_prompt=origin_prompt)
        self.assertIn("accuracy (2 runs average)", results)
        self.assertIn("avg@2", results)
        self.assertIn("pass@2", results)
        self.assertIn("cons@2", results)
        self.assertIn("details", results)

    def test_evaluate_n_single_replica(self):
        """测试BaseEvaluator的evaluate方法在n=1单副本时不添加"runs average"后缀"""
        evaluator = DummyEvaluator(result_metrics={"pass@1": 0.8})
        test_set = Dataset.from_dict({"subdivision": ["cat"], "idx": [0]})
        results = evaluator.evaluate(k=1, n=1, original_dataset=test_set,
                                     predictions=["a"], references=["a"])
        self.assertIn("pass@1", results)
        self.assertNotIn("pass@1 (1 runs average)", results)

    def test_evaluate_no_details_path(self):
        """测试BaseEvaluator的evaluate方法在没有details时的处理路径"""
        evaluator = DummyEvaluator(result_metrics={"accuracy": 0.5}, result_details=None)
        test_set = Dataset.from_dict({"subdivision": ["cat"], "idx": [0]})
        results = evaluator.evaluate(k=1, n=1, original_dataset=test_set,
                                     predictions=["a"], references=["a"])
        self.assertIn("accuracy", results)

    def test_evaluate_details_as_dict(self):
        """测试BaseEvaluator的evaluate方法将Dict类型的details转换为list"""
        evaluator = DummyEvaluator(result_details={"0": {"correct": True}})
        test_set = Dataset.from_dict({"subdivision": ["cat"], "idx": [0]})
        results = evaluator.evaluate(k=[2], n=1, original_dataset=test_set,
                                     predictions=["a"], references=["a"])
        self.assertIn("details", results)

    def test_evaluate_select_fn_with_dataset(self):
        """测试BaseEvaluator的evaluate方法在select_fn使用Dataset时的处理"""
        evaluator = DummyEvaluator(result_metrics={"pass@1": 0.8})
        test_set = Dataset.from_dict({"subdivision": ["cat", "dog"], "idx": [0, 1]})
        results = evaluator.evaluate(k=1, n=2, original_dataset=test_set,
                                     predictions=["a", "b", "c", "d"], references=["a", "b", "c", "d"])
        self.assertIsNotNone(results)

    def test_evaluate_select_fn_with_non_iterable(self):
        """测试BaseEvaluator的evaluate方法在select_fn处理非可迭代对象时的处理"""
        evaluator = DummyEvaluator(result_metrics={"pass@1": 0.8})
        test_set = Dataset.from_dict({"subdivision": ["cat"], "idx": [0]})
        results = evaluator.evaluate(k=1, n=1, original_dataset=test_set,
                                     predictions=["a"], references=["a"], some_param=42)
        self.assertIsNotNone(results)

    def test_is_num_equal(self):
        """测试BaseEvaluator的is_num_equal静态方法对不同长度和相同长度列表的比较"""
        result = BaseEvaluator.is_num_equal([1, 2], [1])
        self.assertIn("error", result)
        result = BaseEvaluator.is_num_equal([1, 2], [1, 2])
        self.assertIsNone(result)

    def test_evaluate_pred_ref_len_mismatch_raises(self):
        """测试BaseEvaluator的evaluate方法在预测和参考长度不匹配时抛出异常"""
        evaluator = BaseEvaluator()
        test_set = Dataset.from_dict({"subdivision": [], "idx": []})
        with self.assertRaises(PredictionInvalidException):
            evaluator.evaluate(k=1, n=1, original_dataset=test_set,
                               predictions=[1, 2], references=[1])

    def test_score_not_implemented(self):
        """测试BaseEvaluator的score方法未实现时抛出异常"""
        evaluator = BaseEvaluator()
        with self.assertRaises(AISBenchImplementationError):
            evaluator.score()


if __name__ == '__main__':
    unittest.main()


