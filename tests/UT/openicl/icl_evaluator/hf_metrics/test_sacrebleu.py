import sys
import types
import unittest


class TestSacrebleuMetric(unittest.TestCase):
    
    def test__compute_corpus_bleu(self):
        """测试Sacrebleu指标使用corpus_bleu计算语料库级别的BLEU分数"""
        if "sacrebleu" in sys.modules:
            del sys.modules["sacrebleu"]
        if "ais_bench.benchmark.openicl.icl_evaluator.hf_metrics.sacrebleu" in sys.modules:
            del sys.modules["ais_bench.benchmark.openicl.icl_evaluator.hf_metrics.sacrebleu"]
        sb = types.ModuleType("sacrebleu")
        sb.__version__ = "2.3.0"

        class DummyOutput:
            def __init__(self):
                self.score = 100.0
                self.counts = [1, 2, 3, 4]
                self.totals = [2, 3, 4, 5]
                self.precisions = [100.0, 50.0, 25.0, 12.5]
                self.bp = 1.0
                self.sys_len = 4
                self.ref_len = 4

        def corpus_bleu(preds, refs, **kwargs):
            assert isinstance(refs, list) and isinstance(refs[0], list)
            return DummyOutput()

        sb.corpus_bleu = corpus_bleu
        sys.modules["sacrebleu"] = sb

        from ais_bench.benchmark.openicl.icl_evaluator.hf_metrics import sacrebleu as sb_mod
        metric = sb_mod.Sacrebleu()
        result = metric._compute(
            predictions=["hello there"],
            references=[["hello there"]],
            tokenize="13a",
        )
        self.assertIn("score", result)
        self.assertEqual(result["score"], 100.0)


if __name__ == '__main__':
    unittest.main()


