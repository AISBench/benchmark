import unittest
from unittest import mock

from datasets import Dataset

from ais_bench.benchmark.openicl.icl_evaluator.icl_hf_evaluator import (
    HuggingfaceEvaluator,
    AccEvaluator,
    AccContaminationEvaluator,
    RougeEvaluator,
    BleuEvaluator,
    BleuFloresEvaluator,
    MccEvaluator,
    SquadEvaluator,
    EDAccEvaluator,
)


class TestHuggingfaceEvaluator(unittest.TestCase):
    def test_len_mismatch_returns_error(self):
        """测试HuggingfaceEvaluator在预测和参考长度不匹配时返回错误"""
        ev = HuggingfaceEvaluator(metric="accuracy")
        out = ev.score(predictions=[1, 2], references=[1])
        self.assertIn("error", out)

    @mock.patch("evaluate.load")
    def test_load_metric_local_then_compute(self, m_load):
        """测试HuggingfaceEvaluator加载指标并计算分数"""
        class DummyMetric:
            def compute(self, **kwargs):
                return {"accuracy": 1.0}

        m_load.return_value = DummyMetric()
        ev = HuggingfaceEvaluator(metric="accuracy")
        out = ev.score(predictions=[1, 0], references=[1, 0])
        self.assertEqual(out, {"accuracy": 1.0})


class TestAccEvaluatorFamily(unittest.TestCase):
    @mock.patch("evaluate.load")
    def test_acc_preprocess_and_postprocess(self, m_load):
        """测试AccEvaluator的预处理和后处理功能"""
        class DummyMetric:
            def compute(self, predictions, references):
                correct = sum(int(p == r) for p, r in zip(predictions, references))
                return {"accuracy": correct / len(predictions)}

        m_load.return_value = DummyMetric()
        ev = AccEvaluator()
        out = ev.score(predictions=["yes", "no"], references=["yes", "no"]) 
        self.assertAlmostEqual(out["accuracy"], 100.0)

    @mock.patch("evaluate.load")
    def test_acc_contamination_evaluator(self, m_load):
        """测试AccContaminationEvaluator对不同污染类型的准确率计算"""
        class DummyMetric:
            def compute(self, predictions, references):
                correct = sum(int(p == r) for p, r in zip(predictions, references))
                return {"accuracy": correct / len(predictions)}

        m_load.return_value = DummyMetric()
        ev = AccContaminationEvaluator()
        ds = Dataset.from_dict({
            "is_clean": [
                "clean",
                "input contamination",
                "input-and-label contamination",
                "clean",
            ]
        })
        preds = ["a", "b", "c", "x"]
        refs = ["a", "b", "c", "y"]
        out = ev.score(preds, refs, ds)
        self.assertTrue(any(k.endswith("- clean") for k in out))
        self.assertTrue(any(k.endswith("- input contaminated") for k in out))
        self.assertTrue(any(k.endswith("- input-and-label contaminated") for k in out))

    @mock.patch("evaluate.load")
    def test_rouge_postprocess(self, m_load):
        """测试RougeEvaluator的后处理将分数乘以100"""
        class DummyMetric:
            def compute(self, predictions, references):
                return {"rouge1": 0.5, "rouge2": 0.25, "rougeL": 0.75}

        m_load.return_value = DummyMetric()
        ev = RougeEvaluator()
        out = ev.score(["a"], ["a"])
        self.assertEqual(out, {"rouge1": 50.0, "rouge2": 25.0, "rougeL": 75.0})

    @mock.patch("evaluate.load")
    def test_bleu_flores_preprocess(self, m_load):
        """测试BleuFloresEvaluator的预处理设置tokenize为flores200"""
        captured = {}

        class DummyMetric:
            def compute(self, predictions, references, tokenize=None):
                captured["tokenize"] = tokenize
                return {"score": 1.0}

        m_load.return_value = DummyMetric()
        ev = BleuFloresEvaluator()
        _ = ev.score(["a"], ["a"])
        self.assertEqual(captured.get("tokenize"), "flores200")

    @mock.patch("evaluate.load")
    def test_mcc_postprocess(self, m_load):
        """测试MccEvaluator的后处理将分数乘以100"""
        class DummyMetric:
            def compute(self, predictions, references):
                return {"matthews_correlation": 0.4}

        m_load.return_value = DummyMetric()
        ev = MccEvaluator()
        out = ev.score(["a"], ["a"])
        self.assertEqual(out["matthews_correlation"], 40.0)

    @mock.patch("evaluate.load")
    def test_squad_pre_and_post(self, m_load):
        """测试SquadEvaluator的预处理和后处理功能"""
        captured = {}

        class DummyMetric:
            def compute(self, predictions, references):
                captured["predictions"] = predictions
                captured["references"] = references
                return {"f1": {"f1": 77.0}}

        m_load.return_value = DummyMetric()
        ev = SquadEvaluator()
        out = ev.score(["ans\nnoise"], ["ans"])
        self.assertEqual(out, {"f1": 77.0})
        self.assertEqual(captured["predictions"][0]["id"], "0")
        self.assertIn("answers", captured["references"][0])

    def test_edacc_preprocess_with_dummy_dist(self):
        """测试EDAccEvaluator使用编辑距离进行预处理和准确率计算"""
        import sys
        import types

        rf_module = types.ModuleType("rapidfuzz")
        distance_module = types.ModuleType("rapidfuzz.distance")
        class DummyLev:
            @staticmethod
            def distance(a, b):
                return 0 if a == b else 1
        distance_module.Levenshtein = DummyLev
        rf_module.distance = distance_module
        sys.modules["rapidfuzz"] = rf_module
        sys.modules["rapidfuzz.distance"] = distance_module

        ev = EDAccEvaluator()
        refs = [
            {"candidates": ["x", ["y", "z"]], "label": 1},
            {"candidates": [["a", "b"], "c"], "label": 0},
        ]
        preds = ["z", "a"]
        with mock.patch("evaluate.load") as m_load:
            class DummyMetric:
                def compute(self, predictions, references):
                    correct = sum(int(p == r) for p, r in zip(predictions, references))
                    return {"accuracy": correct / len(predictions)}
            m_load.return_value = DummyMetric()
            out = ev.score(preds, refs)
            self.assertEqual(out["accuracy"], 100.0)


if __name__ == '__main__':
    unittest.main()


