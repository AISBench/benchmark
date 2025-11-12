import unittest
from unittest import mock


class TestRougeMetric(unittest.TestCase):
    def test__compute_with_aggregator_and_tokenizer(self):
        """测试Rouge指标使用聚合器和自定义分词器的计算"""
        from ais_bench.benchmark.openicl.icl_evaluator.hf_metrics import rouge as rouge_mod

        captured = {"tokenizer_used": False}

        class DummyMid:
            def __init__(self, f):
                self.fmeasure = f

        class DummyStat:
            def __init__(self, f):
                self.mid = DummyMid(f)

        class DummyAgg:
            def __init__(self):
                self.scores = []
            def add_scores(self, score):
                self.scores.append(score)
            def aggregate(self):
                keys = self.scores[0].keys()
                out = {}
                for k in keys:
                    avg = sum(s[k].fmeasure for s in self.scores) / len(self.scores)
                    out[k] = DummyStat(avg)
                return out

        class DummyScore:
            def __init__(self, f):
                self.fmeasure = f

        class DummyRougeScorer:
            def __init__(self, rouge_types, use_stemmer=False, tokenizer=None):
                captured["tokenizer_used"] = tokenizer is not None
                self.rouge_types = rouge_types
            def score(self, ref, pred):
                return {t: DummyScore(1.0 if ref == pred else 0.5) for t in self.rouge_types}

        with mock.patch.object(rouge_mod.rouge_scorer, "RougeScorer", DummyRougeScorer) \
             as m_rouge_scorer, \
             mock.patch.object(rouge_mod.scoring, "BootstrapAggregator", DummyAgg) as m_aggregator:
            metric = rouge_mod.Rouge()
            res = metric._compute(
                predictions=["hello there", "hi"],
                references=["hello there", "hello"],
                rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
                tokenizer=lambda s: s.split(),
                use_aggregator=True,
            )
            self.assertAlmostEqual(res["rouge1"], 0.75)
            self.assertTrue(captured["tokenizer_used"]) 
    
    def test__compute_without_aggregator_returns_per_sample(self):
        """测试Rouge指标不使用聚合器时返回每个样本的分数"""
        from ais_bench.benchmark.openicl.icl_evaluator.hf_metrics import rouge as rouge_mod

        class DummyScore:
            def __init__(self, f):
                self.fmeasure = f

        class DummyRougeScorer:
            def __init__(self, *args, **kwargs):
                pass
            def score(self, ref, pred):
                return {"rouge1": DummyScore(1.0 if ref == pred else 0.0)}

        with mock.patch.object(rouge_mod.rouge_scorer, "RougeScorer", DummyRougeScorer):
            metric = rouge_mod.Rouge()
            res = metric._compute(
                predictions=["a", "b"],
                references=["a", "x"],
                rouge_types=["rouge1"],
                use_aggregator=False,
            )
            self.assertEqual(res["rouge1"], [1.0, 0.0])


if __name__ == '__main__':
    unittest.main()


