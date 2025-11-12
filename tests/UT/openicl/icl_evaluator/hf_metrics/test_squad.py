import sys
import types
import unittest


class TestSquadMetric(unittest.TestCase):
    def test__compute_uses_compute_score_module(self):
        """测试Squad指标使用compute_score模块计算精确匹配和F1分数"""
        pkg_prefix = 'ais_bench.benchmark.openicl.icl_evaluator.hf_metrics'
        fake_mod = types.ModuleType(pkg_prefix + '.compute_score')

        def compute_score(dataset, predictions):
            qas = dataset[0]['paragraphs'][0]['qas']
            correct = 0
            for qa in qas:
                qid = qa['id']
                gold = qa['answers'][0]['text']
                pred = predictions.get(qid, '')
                correct += int(gold == pred)
            total = len(qas)
            return {'exact_match': 100.0 * correct / total, 'f1': 100.0 * correct / total}

        fake_mod.compute_score = compute_score
        sys.modules[pkg_prefix + '.compute_score'] = fake_mod

        from ais_bench.benchmark.openicl.icl_evaluator.hf_metrics import squad as squad_mod
        metric = squad_mod.Squad()
        preds = [{'prediction_text': 'ans', 'id': '0'}]
        refs = [{'id': '0', 'answers': {'text': ['ans'], 'answer_start': [0]}}]
        res = metric._compute(predictions=preds, references=refs)
        self.assertEqual(res, {'exact_match': 100.0, 'f1': 100.0})


if __name__ == '__main__':
    unittest.main()


