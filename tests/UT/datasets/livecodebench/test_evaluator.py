import unittest
import json
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.livecodebench.evaluator import (
        LCBCodeGenerationEvaluator,
        LCBCodeExecutionEvaluator,
        LCBTestOutputEvaluator,
        codegen_check_correctness,
        evaluate_generations_by_problem,
        evaluate_generations,
        codegen_metrics,
        code_execution_metrics,
        parse_assert_statement,
        check_testcase_output,
    )
    from ais_bench.benchmark.datasets.livecodebench.evaluator import test_output_metrics as _test_output_metrics
    LCB_EVALUATOR_AVAILABLE = True
except ImportError:
    LCB_EVALUATOR_AVAILABLE = False


class LiveCodeBenchEvaluatorTestBase(unittest.TestCase):
    """LiveCodeBench评估器测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not LCB_EVALUATOR_AVAILABLE:
            cls.skipTest(cls, "LiveCodeBench evaluator modules not available")


class TestLCBCodeGenerationEvaluator(LiveCodeBenchEvaluatorTestBase):
    """测试LCBCodeGenerationEvaluator类"""

    def test_init(self):
        """测试评估器初始化"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import livecodebench
        from ais_bench.benchmark.datasets.livecodebench import evaluator as evaluator_module

        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = MagicMock(return_value={'question_id': 'q1', 'evaluation_sample': '{"test": "data"}'})
        mock_dataset.__len__ = MagicMock(return_value=1)

        # Patch LCBCodeGenerationDataset where it's used in evaluator module
        with patch.object(evaluator_module, 'LCBCodeGenerationDataset') as mock_dataset_class:
            mock_dataset_class.load.return_value = {'test': mock_dataset}

            evaluator = LCBCodeGenerationEvaluator(
                num_process_evaluate=4,
                timeout=6,
                release_version='release_v1',
                extractor_version='v1'
            )

            self.assertEqual(evaluator.num_process_evaluate, 4)
            self.assertEqual(evaluator.timeout, 6)
            self.assertEqual(evaluator.extractor_version, 'v1')

    def test_score(self):
        """测试score方法"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import evaluator as evaluator_module

        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = MagicMock(return_value={'question_id': 'q1', 'evaluation_sample': '{"input_output": "test"}'})
        mock_dataset.__len__ = MagicMock(return_value=1)

        with patch.object(evaluator_module, 'LCBCodeGenerationDataset') as mock_dataset_class, \
             patch.object(evaluator_module, 'extract_code_generation', return_value='extracted_code') as mock_extract, \
             patch.object(evaluator_module, 'codegen_metrics', return_value=[
                 {'pass@1': 100.0, 'detail': {'pass@1': {'0': 100.0}}},
                 {0: [[True]]},
                 [['metadata']]
             ]) as mock_codegen_metrics:
            mock_dataset_class.load.return_value = {'test': mock_dataset}

            evaluator_instance = LCBCodeGenerationEvaluator(num_process_evaluate=1, timeout=6)

            predictions = ['prediction1']
            references = ['q1']
            test_set = [{'question_id': 'q1', 'evaluation_sample': '{"input_output": "test"}'}]

            result = evaluator_instance.score(predictions, references, test_set)

            self.assertIn('pass@1', result)
            self.assertIn('details', result)

    def test_score_v2_extractor(self):
        """测试score方法使用v2提取器"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import evaluator as evaluator_module

        mock_dataset = MagicMock()
        mock_dataset.__getitem__ = MagicMock(return_value={'question_id': 'q1', 'evaluation_sample': '{"input_output": "test"}'})
        mock_dataset.__len__ = MagicMock(return_value=1)

        with patch.object(evaluator_module, 'LCBCodeGenerationDataset') as mock_dataset_class, \
             patch.object(evaluator_module, 'extract_code_generation_v2', return_value='extracted_code_v2') as mock_extract, \
             patch.object(evaluator_module, 'codegen_metrics', return_value=[
                 {'pass@1': 50.0, 'detail': {'pass@1': {'0': 50.0}}},
                 {0: [[False]]},
                 [['metadata']]
             ]) as mock_codegen_metrics:
            mock_dataset_class.load.return_value = {'test': mock_dataset}

            evaluator = LCBCodeGenerationEvaluator(
                num_process_evaluate=1,
                timeout=6,
                extractor_version='v2'
            )

            predictions = ['prediction1']
            references = ['q1']
            test_set = [{'question_id': 'q1', 'evaluation_sample': '{"input_output": "test"}'}]

            result = evaluator.score(predictions, references, test_set)

            self.assertIn('pass@1', result)
            self.assertIn('details', result)
        # 验证使用了v2提取器
        mock_extract.assert_called()


class TestLCBCodeExecutionEvaluator(LiveCodeBenchEvaluatorTestBase):
    """测试LCBCodeExecutionEvaluator类"""

    def test_init(self):
        """测试评估器初始化"""
        evaluator = LCBCodeExecutionEvaluator()
        self.assertIsNotNone(evaluator)

    def test_score(self):
        """测试score方法"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import evaluator as evaluator_module

        evaluator = LCBCodeExecutionEvaluator()

        with patch.object(evaluator_module, 'extract_code_execution', return_value='extracted_code') as mock_extract, \
             patch.object(evaluator_module, 'code_execution_metrics', return_value=[
                 {'pass@1': 100.0},
                 {0: [[True]]}
             ]) as mock_metrics:
            predictions = ['prediction1']
            references = ['{"code": "def f(): pass", "input": "test", "output": "result"}']

            result = evaluator.score(predictions, references)

            self.assertIn('pass@1', result)


class TestLCBTestOutputEvaluator(LiveCodeBenchEvaluatorTestBase):
    """测试LCBTestOutputEvaluator类"""

    def test_init(self):
        """测试评估器初始化"""
        evaluator = LCBTestOutputEvaluator()
        self.assertIsNotNone(evaluator)

    def test_score(self):
        """测试score方法"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import evaluator as evaluator_module

        evaluator = LCBTestOutputEvaluator()

        with patch.object(evaluator_module, 'extract_test_output_code', return_value='assert test() == "output"') as mock_extract, \
             patch.object(evaluator_module, 'test_output_metrics', return_value=[
                 {'pass@1': 100.0},
                 {0: [[True]]}
             ]) as mock_metrics:
            predictions = ['prediction1']
            references = ['{"input": "test", "output": "result"}']

            result = evaluator.score(predictions, references)

            self.assertIn('pass@1', result)


class TestParseAssertStatement(LiveCodeBenchEvaluatorTestBase):
    """测试parse_assert_statement函数"""

    def test_parse_valid_assert(self):
        """测试解析有效的assert语句"""
        statement = 'assert func(1) == 2'
        result = parse_assert_statement(statement)
        self.assertIn('2', result)

    def test_parse_invalid_syntax(self):
        """测试解析无效语法"""
        statement = 'assert func(1) =='
        result = parse_assert_statement(statement)
        self.assertIn('Invalid', result)

    def test_parse_not_assert(self):
        """测试解析非assert语句"""
        statement = 'x = 1'
        result = parse_assert_statement(statement)
        self.assertIn('Not an assert', result)


class TestCheckTestcaseOutput(LiveCodeBenchEvaluatorTestBase):
    """测试check_testcase_output函数"""

    def test_check_valid_output(self):
        """测试检查有效输出"""
        testcase_str = 'assert test() == "output"'
        expected_output = '"output"'
        result = check_testcase_output(testcase_str, expected_output)
        self.assertIsInstance(result, bool)

    def test_check_invalid_output(self):
        """测试检查无效输出"""
        testcase_str = 'assert test() == "wrong"'
        expected_output = '"output"'
        result = check_testcase_output(testcase_str, expected_output)
        self.assertFalse(result)


class TestCodeGenCheckCorrectness(LiveCodeBenchEvaluatorTestBase):
    """测试codegen_check_correctness函数"""

    def test_check_correctness_success(self):
        """测试检查正确性成功"""
        from unittest.mock import patch
        import subprocess

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = '{"res": [True], "meta": {}}'
        mock_proc.stderr = ''
        mock_subprocess = MagicMock(return_value=mock_proc)

        with patch.object(subprocess, 'run', mock_subprocess):
            sample = {'input_output': json.dumps({'inputs': ['test'], 'outputs': ['result']})}
            generation = 'def test(): return "result"'

            result, meta = codegen_check_correctness(sample, generation, timeout=6)

            self.assertIsInstance(result, list)
            self.assertIsInstance(meta, dict)

    def test_check_correctness_timeout(self):
        """测试检查正确性超时"""
        from unittest.mock import patch
        import subprocess

        mock_subprocess = MagicMock(side_effect=subprocess.TimeoutExpired('test', 6))

        with patch.object(subprocess, 'run', mock_subprocess):
            sample = {'input_output': json.dumps({'inputs': ['test'], 'outputs': ['result']})}
            generation = 'def test(): return "result"'

            result, meta = codegen_check_correctness(sample, generation, timeout=6)

            self.assertIsInstance(result, list)
            self.assertEqual(result[0], -1)

    def test_check_correctness_exception(self):
        """测试检查正确性异常"""
        from unittest.mock import patch
        import subprocess

        mock_subprocess = MagicMock(side_effect=Exception('subprocess error'))

        with patch.object(subprocess, 'run', mock_subprocess):
            sample = {'input_output': json.dumps({'inputs': ['test'], 'outputs': ['result']})}
            generation = 'def test(): return "result"'

            result, meta = codegen_check_correctness(sample, generation, timeout=6)

            self.assertIsInstance(result, list)
            self.assertEqual(result[0], -1)

    def test_check_correctness_nonzero_returncode(self):
        """测试检查正确性非零返回码"""
        from unittest.mock import patch
        import subprocess

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdout = '{"res": [True], "meta": {}, "error": "test error"}'
        mock_proc.stderr = 'error output'
        mock_subprocess = MagicMock(return_value=mock_proc)

        with patch.object(subprocess, 'run', mock_subprocess):
            sample = {'input_output': json.dumps({'inputs': ['test'], 'outputs': ['result']})}
            generation = 'def test(): return "result"'

            result, meta = codegen_check_correctness(sample, generation, timeout=6)

            self.assertIsInstance(result, list)
            self.assertIsInstance(meta, dict)

    def test_check_correctness_empty_stdout(self):
        """测试检查正确性空标准输出"""
        from unittest.mock import patch
        import subprocess

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = ''
        mock_proc.stderr = ''
        mock_subprocess = MagicMock(return_value=mock_proc)

        with patch.object(subprocess, 'run', mock_subprocess):
            sample = {'input_output': json.dumps({'inputs': ['test'], 'outputs': ['result']})}
            generation = 'def test(): return "result"'

            result, meta = codegen_check_correctness(sample, generation, timeout=6)

            self.assertIsInstance(result, list)
            self.assertEqual(result[0], -1)

    def test_check_correctness_invalid_json(self):
        """测试检查正确性无效JSON"""
        from unittest.mock import patch
        import subprocess

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = 'invalid json'
        mock_proc.stderr = ''
        mock_subprocess = MagicMock(return_value=mock_proc)

        with patch.object(subprocess, 'run', mock_subprocess):
            sample = {'input_output': json.dumps({'inputs': ['test'], 'outputs': ['result']})}
            generation = 'def test(): return "result"'

            result, meta = codegen_check_correctness(sample, generation, timeout=6)

            self.assertIsInstance(result, list)
            self.assertEqual(result[0], -1)


class TestEvaluateGenerationsByProblem(LiveCodeBenchEvaluatorTestBase):
    """测试evaluate_generations_by_problem函数"""

    def test_evaluate_by_problem(self):
        """测试按问题评估"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import evaluator as evaluator_module
        import numpy as np

        mock_check = MagicMock(return_value=([True], {}))

        with patch.object(evaluator_module, 'codegen_check_correctness', mock_check):
            problem_generations = ['code1', 'code2']
            sample = {'input_output': json.dumps({'inputs': ['test'], 'outputs': ['result']})}

            results, metadata = evaluate_generations_by_problem(
                problem_generations, sample, debug=False, timeout=6
            )

            self.assertEqual(len(results), 2)
            self.assertEqual(len(metadata), 2)

    def test_evaluate_by_problem_with_numpy_array(self):
        """测试按问题评估包含numpy数组的情况"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import evaluator as evaluator_module
        import numpy as np

        mock_check = MagicMock(return_value=([np.array([True]), np.bool_(True)], {}))

        with patch.object(evaluator_module, 'codegen_check_correctness', mock_check):
            problem_generations = ['code1']
            sample = {'input_output': json.dumps({'inputs': ['test'], 'outputs': ['result']})}

            results, metadata = evaluate_generations_by_problem(
                problem_generations, sample, debug=False, timeout=6
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(len(metadata), 1)

    def test_evaluate_by_problem_with_exception(self):
        """测试按问题评估异常情况"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import evaluator as evaluator_module

        mock_check = MagicMock(side_effect=Exception('test error'))

        with patch.object(evaluator_module, 'codegen_check_correctness', mock_check):
            problem_generations = ['code1']
            sample = {'input_output': json.dumps({'inputs': ['test'], 'outputs': ['result']})}

            results, metadata = evaluate_generations_by_problem(
                problem_generations, sample, debug=False, timeout=6
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0], [-2])  # 编译错误


class TestCodeExecutionMetrics(LiveCodeBenchEvaluatorTestBase):
    """测试code_execution_metrics函数"""

    def test_code_execution_metrics(self):
        """测试代码执行指标计算"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import evaluator as evaluator_module
        from concurrent.futures import ProcessPoolExecutor

        mock_evaluate = MagicMock(return_value=[True, False])

        # 模拟ProcessPoolExecutor
        mock_executor_instance = MagicMock()
        mock_executor_instance.map.return_value = [[True], [False]]
        mock_executor = MagicMock()
        mock_executor.return_value.__enter__ = MagicMock(return_value=mock_executor_instance)
        mock_executor.return_value.__exit__ = MagicMock(return_value=None)

        with patch.object(evaluator_module, 'evaluate_score', mock_evaluate), \
             patch.object(evaluator_module, 'ProcessPoolExecutor', mock_executor):
            samples = [
                {'code': 'def f(): return 1', 'input': 'test', 'output': '1'}
            ]
            generations = [['result1'], ['result2']]

            metrics, results = code_execution_metrics(samples, generations)

            self.assertIn('pass@1', metrics)
            self.assertIsInstance(results, dict)

    def test_code_execution_metrics_with_multiple_samples(self):
        """测试多个样本的代码执行指标计算"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import evaluator as evaluator_module

        def evaluate_score_impl(args):
            gs, (c, i, o) = args
            return [True if i in g else False for g in gs]

        mock_executor_instance = MagicMock()
        mock_executor_instance.map.side_effect = lambda func, args: [
            evaluate_score_impl(arg) for arg in args
        ]
        mock_executor = MagicMock()
        mock_executor.return_value.__enter__ = MagicMock(return_value=mock_executor_instance)
        mock_executor.return_value.__exit__ = MagicMock(return_value=None)

        with patch.object(evaluator_module, 'ProcessPoolExecutor', mock_executor):
            samples = [
                {'code': 'def f(x): return x', 'input': 'f(1)', 'output': '1'},
                {'code': 'def g(y): return y+1', 'input': 'g(2)', 'output': '3'}
            ]
            generations = [['1'], ['3']]

            metrics, results = code_execution_metrics(samples, generations)

            self.assertIn('pass@1', metrics)
            self.assertIsInstance(results, dict)
            self.assertEqual(len(results), 2)


class TestCodegenMetrics(LiveCodeBenchEvaluatorTestBase):
    """测试codegen_metrics函数"""

    def test_codegen_metrics(self):
        """测试代码生成指标计算"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import evaluator as evaluator_module

        mock_evaluate = MagicMock(return_value=(
            {0: [[True]], 1: [[False]]},
            {0: [{}], 1: [{}]}
        ))
        mock_compute_metrics = MagicMock(return_value={'pass@1': 50.0, 'detail': {'pass@1': {0: 100.0, 1: 0.0}}})

        with patch.object(evaluator_module, 'evaluate_generations', mock_evaluate), \
             patch.object(evaluator_module, 'compute_metrics_from_results', mock_compute_metrics):
            samples_list = [
                {'input_output': json.dumps({'inputs': ['test1'], 'outputs': ['result1']})},
                {'input_output': json.dumps({'inputs': ['test2'], 'outputs': ['result2']})}
            ]
            generations_list = [['code1'], ['code2']]

            result = codegen_metrics(samples_list, generations_list, k_list=[1], num_process_evaluate=1, timeout=6)

            self.assertEqual(len(result), 3)  # [metrics, results, metadata]
            self.assertIn('pass@1', result[0])


class TestEvaluateGenerations(LiveCodeBenchEvaluatorTestBase):
    """测试evaluate_generations函数"""

    def test_evaluate_generations(self):
        """测试评估多个生成"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import evaluator as evaluator_module
        from concurrent.futures import as_completed

        mock_evaluate_by_problem = MagicMock(return_value=([[True]], [{}]))

        mock_executor_instance = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = ([[True]], [{}])
        mock_executor_instance.submit.return_value = mock_future
        mock_executor = MagicMock()
        mock_executor.return_value.__enter__ = MagicMock(return_value=mock_executor_instance)
        mock_executor.return_value.__exit__ = MagicMock(return_value=None)

        mock_as_completed = MagicMock(return_value=[mock_future])

        with patch.object(evaluator_module, 'evaluate_generations_by_problem', mock_evaluate_by_problem), \
             patch.object(evaluator_module, 'ProcessPoolExecutor', mock_executor), \
             patch.object(evaluator_module, 'as_completed', mock_as_completed):
            samples_list = [
                {'input_output': json.dumps({'inputs': ['test'], 'outputs': ['result']})}
            ]
            generations_list = [['code1']]

            results, metadata = evaluate_generations(
                samples_list, generations_list, debug=False, num_process_evaluate=1, timeout=6
            )

            self.assertIsInstance(results, dict)
            self.assertIsInstance(metadata, dict)


class TestEvaluateScore(LiveCodeBenchEvaluatorTestBase):
    """测试evaluate_score函数"""

    def test_evaluate_score_with_matching_input(self):
        """测试evaluate_score函数当输入匹配时"""
        from ais_bench.benchmark.datasets.livecodebench.evaluator import evaluate_score

        generations = ['result1', 'result2']
        code = 'def f(x): return x'
        input_str = 'f(1)'
        output = '1'

        # 当generation包含input时，应该跳过执行
        result = evaluate_score((generations, (code, input_str, output)))
        # 由于input在generation中，应该返回False列表
        self.assertIsInstance(result, list)

    def test_evaluate_score_with_execution(self):
        """测试evaluate_score函数执行代码检查"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import evaluator as evaluator_module
        from ais_bench.benchmark.datasets.livecodebench.evaluator import evaluate_score

        mock_check = MagicMock(return_value=True)

        with patch.object(evaluator_module, 'codeexecute_check_correctness', mock_check):
            generations = ['result1']
            code = 'def f(x): return x'
            input_str = 'f(1)'
            output = '1'

            result = evaluate_score((generations, (code, input_str, output)))
            self.assertIsInstance(result, list)


class TestTestOutputMetrics(LiveCodeBenchEvaluatorTestBase):
    """测试test_output_metrics函数"""

    def test_test_output_metrics(self):
        """测试测试输出指标计算"""
        from unittest.mock import patch
        from ais_bench.benchmark.datasets.livecodebench import evaluator as evaluator_module

        mock_check = MagicMock(return_value=True)
        mock_compute_metrics = MagicMock(return_value={'pass@1': 100.0})

        with patch.object(evaluator_module, 'check_testcase_output', mock_check), \
             patch.object(evaluator_module, 'compute_metrics_from_results', mock_compute_metrics):
            samples = [
                {'input': 'test', 'output': '"result"'}
            ]
            generations = [['assert test() == "result"']]

            metrics, results = _test_output_metrics(samples, generations, k_list=[1])

            self.assertIn('pass@1', metrics)
            self.assertIsInstance(results, dict)


if __name__ == '__main__':
    unittest.main()

