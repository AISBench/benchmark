import unittest
from unittest import mock

from ais_bench.benchmark.openicl.icl_evaluator.icl_jieba_rouge_evaluator import JiebaRougeEvaluator


class TestJiebaRougeEvaluator(unittest.TestCase):
    def test_len_mismatch(self):
        """测试JiebaRougeEvaluator在预测和参考长度不匹配时返回错误"""
        ev = JiebaRougeEvaluator()
        out = ev.score(["a", "b"], ["a"])
        self.assertIn("error", out)

    @mock.patch("ais_bench.benchmark.openicl.icl_evaluator.icl_jieba_rouge_evaluator.Rouge")
    @mock.patch("ais_bench.benchmark.openicl.icl_evaluator.icl_jieba_rouge_evaluator.jieba.cut")
    def test_tokenization_and_scores(self, m_cut, m_rouge):
        """测试JiebaRougeEvaluator使用jieba分词和Rouge计算分数，包括空字符串处理"""
        m_cut.side_effect = lambda s: s.split()

        class DummyRouge:
            def get_scores(self, predictions, references, avg=False):
                assert "__PREDPLACEHOLDER__" in predictions
                assert "__REFRPLACEHOLDER__" in references
                return {
                    "rouge-1": {"f": 0.3},
                    "rouge-2": {"f": 0.2},
                    "rouge-l": {"f": 0.4},
                }

        m_rouge.return_value = DummyRouge()
        ev = JiebaRougeEvaluator()
        out = ev.score(["hello world", ""], ["hello world", ""])
        self.assertEqual(out, {"rouge1": 30.0, "rouge2": 20.0, "rougeL": 40.0})


if __name__ == '__main__':
    unittest.main()


