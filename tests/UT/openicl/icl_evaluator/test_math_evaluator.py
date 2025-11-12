import unittest
from unittest import mock

from ais_bench.benchmark.openicl.icl_evaluator.math_evaluator import MATHEvaluator
from ais_bench.benchmark.utils.logging.exceptions import AISBenchImportError


class TestMathEvaluator(unittest.TestCase):
    def test_import_error_branch(self):
        """测试MATHEvaluator在导入失败时抛出AISBenchImportError"""
        with mock.patch.dict('sys.modules', {"latex2sympy2_extended": None, "math_verify": None}):
            with self.assertRaises(AISBenchImportError):
                MATHEvaluator().score(["1"], ["1"])

    def test_success_path_with_fakes(self):
        """测试MATHEvaluator使用模拟模块成功计算数学答案的准确率"""
        import types, sys

        math_verify = types.ModuleType("math_verify")
        def fake_parse(s, extraction_mode=None, extraction_config=None):
            return [f"PARSED({s})"]
        def fake_verify(answer_parsed, gold_parsed):
            return float(str(answer_parsed[0]).replace("PARSED(", "").rstrip(")") ==
                         str(gold_parsed[0]).replace("PARSED(", "").rstrip(")"))
        class ExprExtractionConfig: pass
        class LatexExtractionConfig:
            def __init__(self, normalization_config=None, boxed_match_priority=None, try_extract_without_anchor=None):
                pass
        math_verify.parse = fake_parse
        math_verify.verify = fake_verify
        math_verify.ExprExtractionConfig = ExprExtractionConfig
        math_verify.LatexExtractionConfig = LatexExtractionConfig

        latex2sympy2_extended = types.ModuleType("latex2sympy2_extended")
        class NormalizationConfig:
            def __init__(self, **kwargs):
                pass
        latex2sympy2_extended.NormalizationConfig = NormalizationConfig

        with mock.patch.dict('sys.modules', {
            'math_verify': math_verify,
            'latex2sympy2_extended': latex2sympy2_extended,
        }):
            ev = MATHEvaluator()
            out = ev.score(["x"], ["x"])
            self.assertIn('accuracy', out)
            self.assertEqual(out['accuracy'], 100.0)
            self.assertTrue(isinstance(out.get('details'), list))


if __name__ == '__main__':
    unittest.main()


