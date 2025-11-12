import unittest
from unittest import mock


class TestAccuracyMetric(unittest.TestCase):
    def test__compute_uses_sklearn_accuracy(self):
        """测试Accuracy指标使用sklearn的accuracy_score计算准确率"""
        from ais_bench.benchmark.openicl.icl_evaluator.hf_metrics import accuracy as acc_mod

        with mock.patch.object(acc_mod, "accuracy_score", return_value=0.75) as m_acc:
            metric = acc_mod.Accuracy()
            out = metric._compute(predictions=[1, 0, 1, 1], references=[1, 0, 0, 1])
            self.assertEqual(out, {"accuracy": 0.75})
            m_acc.assert_called_once()

    def test__compute_with_normalize_false_and_weights(self):
        """测试Accuracy指标在normalize=False和sample_weight参数下的计算"""
        from ais_bench.benchmark.openicl.icl_evaluator.hf_metrics import accuracy as acc_mod

        def fake_acc(refs, preds, normalize=True, sample_weight=None):
            self.assertFalse(normalize)
            self.assertEqual(sample_weight, [0.5, 1.0])
            return 1.5

        metric = acc_mod.Accuracy()
        with mock.patch.object(acc_mod, "accuracy_score", side_effect=fake_acc):
            out = metric._compute(predictions=[1, 0], references=[1, 0], normalize=False, sample_weight=[0.5, 1.0])
            self.assertEqual(out, {"accuracy": 1.5})


if __name__ == '__main__':
    unittest.main()


