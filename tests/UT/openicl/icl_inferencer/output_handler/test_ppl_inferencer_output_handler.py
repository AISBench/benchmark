import unittest
from unittest import mock
import sqlite3

from ais_bench.benchmark.openicl.icl_inferencer.output_handler.ppl_inferencer_output_handler import (
    PPLInferencerOutputHandler,
    PPLRequestOutput,
    PPLResponseOutput
)
from ais_bench.benchmark.models.output import Output
from ais_bench.benchmark.utils.logging.exceptions import AISBenchImplementationError
from ais_bench.benchmark.utils.logging.error_codes import CALC_CODES


class TestPPLRequestOutput(unittest.TestCase):
    def test_init(self):
        """测试PPLRequestOutput初始化"""
        output = PPLRequestOutput(perf_mode=False)
        self.assertFalse(output.perf_mode)
        self.assertEqual(output.ppl, 0)
        self.assertIsNone(output.origin_prompt_logprobs)

    def test_init_perf_mode(self):
        """测试PPLRequestOutput在perf模式下初始化"""
        output = PPLRequestOutput(perf_mode=True)
        self.assertTrue(output.perf_mode)
        self.assertEqual(output.ppl, 0)
        self.assertIsNone(output.origin_prompt_logprobs)


class TestPPLResponseOutput(unittest.TestCase):
    def test_init(self):
        """测试PPLResponseOutput初始化"""
        output = PPLResponseOutput(perf_mode=False)
        self.assertFalse(output.perf_mode)
        self.assertEqual(output.input, [])
        self.assertEqual(output.label_ppl_list, [])
        self.assertEqual(output.origin_prompt_logprobs, [])

    def test_init_perf_mode(self):
        """测试PPLResponseOutput在perf模式下初始化"""
        output = PPLResponseOutput(perf_mode=True)
        self.assertTrue(output.perf_mode)
        self.assertEqual(output.input, [])
        self.assertEqual(output.label_ppl_list, [])
        self.assertEqual(output.origin_prompt_logprobs, [])


class TestPPLInferencerOutputHandler(unittest.TestCase):
    def test_init_default_save_every(self):
        """测试PPLInferencerOutputHandler使用默认save_every"""
        handler = PPLInferencerOutputHandler(perf_mode=False)
        self.assertFalse(handler.perf_mode)
        self.assertEqual(handler.save_every, 100)  # 默认值是100
        self.assertTrue(handler.all_success)

    def test_init_perf_mode_default_save_every(self):
        """测试PPLInferencerOutputHandler在perf模式下使用默认save_every"""
        handler = PPLInferencerOutputHandler(perf_mode=True)
        self.assertTrue(handler.perf_mode)
        self.assertEqual(handler.save_every, 100)  # 默认值是100

    def test_get_prediction_result_success(self):
        """测试get_prediction_result成功情况"""
        handler = PPLInferencerOutputHandler(perf_mode=False)
        
        output = PPLResponseOutput(perf_mode=False)
        output.success = True
        output.uuid = "test_uuid"
        output.input = ["prompt1", "prompt2"]
        output.label_ppl_list = [
            {"label": "A", "ppl": 2.5},
            {"label": "B", "ppl": 1.5}
        ]
        output.origin_prompt_logprobs = [{"token1": {"logprob": -0.5}}]
        output.get_prediction = mock.Mock(return_value="B")
        
        result = handler.get_prediction_result(output, gold="A", input="input")
        
        self.assertTrue(result["success"])
        self.assertEqual(result["uuid"], "test_uuid")
        self.assertEqual(result["origin_prompt"], ["prompt1", "prompt2"])
        self.assertEqual(result["ppl_list"], [
            {"label": "A", "ppl": 2.5},
            {"label": "B", "ppl": 1.5}
        ])
        self.assertEqual(result["origin_prompt_logprobs"], [{"token1": {"logprob": -0.5}}])
        self.assertEqual(result["prediction"], "B")
        self.assertEqual(result["gold"], "A")

    def test_get_prediction_result_without_gold(self):
        """测试get_prediction_result不提供gold"""
        handler = PPLInferencerOutputHandler(perf_mode=False)
        
        output = PPLResponseOutput(perf_mode=False)
        output.success = True
        output.uuid = "test_uuid"
        output.input = ["prompt1"]
        output.label_ppl_list = [{"label": "A", "ppl": 2.5}]
        output.origin_prompt_logprobs = []
        output.get_prediction = mock.Mock(return_value="A")
        
        result = handler.get_prediction_result(output, input="input")
        
        self.assertTrue(result["success"])
        self.assertNotIn("gold", result)

    def test_get_prediction_result_failure(self):
        """测试get_prediction_result失败情况"""
        handler = PPLInferencerOutputHandler(perf_mode=False)
        
        output = PPLResponseOutput(perf_mode=False)
        output.success = False
        output.uuid = "test_uuid"
        output.input = []
        output.label_ppl_list = []
        output.origin_prompt_logprobs = []
        output.get_prediction = mock.Mock(return_value=None)
        
        result = handler.get_prediction_result(output, gold="A", input="input")
        
        self.assertFalse(result["success"])
        self.assertEqual(result["uuid"], "test_uuid")

    def test_get_prediction_result_list_input(self):
        """测试get_prediction_result使用列表输入"""
        handler = PPLInferencerOutputHandler(perf_mode=False)
        
        output = PPLResponseOutput(perf_mode=False)
        output.success = True
        output.uuid = "test_uuid"
        output.input = ["prompt1", "prompt2"]
        output.label_ppl_list = [{"label": "A", "ppl": 2.5}]
        output.origin_prompt_logprobs = []
        output.get_prediction = mock.Mock(return_value="A")
        
        result = handler.get_prediction_result(output, input=["input1", "input2"])
        
        self.assertTrue(result["success"])
        self.assertEqual(result["origin_prompt"], ["prompt1", "prompt2"])

    def test_get_prediction_result_empty_ppl_list(self):
        """测试get_prediction_result处理空的ppl_list"""
        handler = PPLInferencerOutputHandler(perf_mode=False)
        
        output = PPLResponseOutput(perf_mode=False)
        output.success = True
        output.uuid = "test_uuid"
        output.input = []
        output.label_ppl_list = []
        output.origin_prompt_logprobs = []
        output.get_prediction = mock.Mock(return_value=None)
        
        result = handler.get_prediction_result(output, input="input")
        
        self.assertTrue(result["success"])
        self.assertEqual(result["ppl_list"], [])
        self.assertEqual(result["origin_prompt_logprobs"], [])

    def test_get_result_inherited_behavior(self):
        """测试get_result继承自基类的行为"""
        handler = PPLInferencerOutputHandler(perf_mode=False)
        conn = sqlite3.connect(":memory:")
        
        output = PPLResponseOutput(perf_mode=False)
        output.success = True
        output.uuid = "test_uuid"
        output.input = ["prompt1"]
        output.label_ppl_list = [{"label": "A", "ppl": 2.5}]
        output.origin_prompt_logprobs = []
        output.get_prediction = mock.Mock(return_value="A")
        
        result = handler.get_result(conn, "data_abbr", "input", output, gold="A")
        
        self.assertTrue(result["success"])
        self.assertEqual(result["uuid"], "test_uuid")
        self.assertEqual(result["prediction"], "A")
        self.assertEqual(result["gold"], "A")
        self.assertIn("ppl_list", result)
        self.assertIn("origin_prompt_logprobs", result)
        
        conn.close()

    def test_get_result_with_failure(self):
        """测试get_result处理失败情况"""
        handler = PPLInferencerOutputHandler(perf_mode=False)
        conn = sqlite3.connect(":memory:")
        
        output = PPLResponseOutput(perf_mode=False)
        output.success = False
        output.uuid = "test_uuid"
        output.input = []
        output.label_ppl_list = []
        output.origin_prompt_logprobs = []
        output.error_info = "Test error"
        output.get_prediction = mock.Mock(return_value=None)
        
        result = handler.get_result(conn, "data_abbr", "input", output, gold="A")
        
        self.assertFalse(result["success"])
        self.assertIn("error_info", result)
        self.assertEqual(result["error_info"], "Test error")
        self.assertFalse(handler.all_success)
        
        conn.close()

    def test_get_result_perf_mode(self):
        """测试get_result在perf模式下的行为"""
        handler = PPLInferencerOutputHandler(perf_mode=True)
        conn = sqlite3.connect(":memory:")
        
        output = PPLResponseOutput(perf_mode=True)
        output.success = True
        output.uuid = "test_uuid"
        output.get_metrics = mock.Mock(return_value={"latency": 0.1, "throughput": 100})
        
        handler._extract_and_write_arrays = mock.Mock(return_value={"latency": 0.1, "throughput": 100})
        
        result = handler.get_result(conn, "data_abbr", "input", output, gold="A")
        
        self.assertIn("latency", result)
        self.assertIn("throughput", result)
        handler._extract_and_write_arrays.assert_called_once()
        
        conn.close()


if __name__ == '__main__':
    unittest.main()
