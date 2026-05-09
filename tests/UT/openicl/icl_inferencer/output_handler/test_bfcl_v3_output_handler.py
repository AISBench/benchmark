import unittest
from unittest import mock
import sqlite3

from ais_bench.benchmark.openicl.icl_inferencer.output_handler.bfcl_v3_output_handler import BFCLV3OutputHandler
from ais_bench.benchmark.models.output import FunctionCallOutput, Output
from ais_bench.benchmark.utils.logging.exceptions import AISBenchTypeError


class TestBFCLV3OutputHandler(unittest.TestCase):
    """Test BFCLV3OutputHandler class"""

    def test_init(self):
        """Test initialization"""
        handler = BFCLV3OutputHandler(perf_mode=False, save_every=10)
        self.assertFalse(handler.perf_mode)
        self.assertEqual(handler.save_every, 10)
        self.assertTrue(handler.all_success)

    def test_init_perf_mode(self):
        """Test initialization with perf_mode=True"""
        handler = BFCLV3OutputHandler(perf_mode=True, save_every=5)
        self.assertTrue(handler.perf_mode)
        self.assertEqual(handler.save_every, 5)

    def test_get_prediction_result_success(self):
        """Test get_prediction_result with successful FunctionCallOutput"""
        handler = BFCLV3OutputHandler(perf_mode=False)
        
        output = FunctionCallOutput()
        output.success = True
        output.uuid = "test_uuid_123"
        output.tool_calls = [
            {"function": "get_weather", "arguments": {"location": "Beijing"}},
            {"function": "get_time", "arguments": {}}
        ]
        output.inference_log = [
            {"step": 1, "action": "call_function", "function": "get_weather"},
            {"step": 2, "action": "call_function", "function": "get_time"}
        ]
        
        result = handler.get_prediction_result(output)
        
        self.assertEqual(result["success"], True)
        self.assertEqual(result["uuid"], "test_uuid_123")
        self.assertEqual(result["prediction"], output.tool_calls)
        self.assertEqual(result["inference_log"], output.inference_log)
        self.assertIn("success", result)
        self.assertIn("uuid", result)
        self.assertIn("prediction", result)
        self.assertIn("inference_log", result)

    def test_get_prediction_result_with_gold(self):
        """Test get_prediction_result with gold parameter"""
        handler = BFCLV3OutputHandler(perf_mode=False)
        
        output = FunctionCallOutput()
        output.success = True
        output.uuid = "test_uuid_456"
        output.tool_calls = [{"function": "test_func", "arguments": {}}]
        output.inference_log = []
        
        gold = "expected_result"
        result = handler.get_prediction_result(output, gold=gold)
        
        self.assertEqual(result["success"], True)
        self.assertEqual(result["uuid"], "test_uuid_456")
        self.assertEqual(result["prediction"], output.tool_calls)
        # Note: gold is not included in result_data, only in the returned dict if needed
        # The method doesn't add gold to result_data, so we just verify the basic structure

    def test_get_prediction_result_with_input(self):
        """Test get_prediction_result with input parameter"""
        handler = BFCLV3OutputHandler(perf_mode=False)
        
        output = FunctionCallOutput()
        output.success = True
        output.uuid = "test_uuid_789"
        output.tool_calls = []
        output.inference_log = []
        
        input_data = "test_input"
        result = handler.get_prediction_result(output, gold=None, input=input_data)
        
        self.assertEqual(result["success"], True)
        self.assertEqual(result["uuid"], "test_uuid_789")
        # Note: input is not used in the implementation, but parameter is accepted

    def test_get_prediction_result_failed_output(self):
        """Test get_prediction_result with failed FunctionCallOutput"""
        handler = BFCLV3OutputHandler(perf_mode=False)
        
        output = FunctionCallOutput()
        output.success = False
        output.uuid = "test_uuid_failed"
        output.tool_calls = []
        output.inference_log = []
        output.error_info = "Test error message"
        
        result = handler.get_prediction_result(output)
        
        self.assertEqual(result["success"], False)
        self.assertEqual(result["uuid"], "test_uuid_failed")
        self.assertEqual(result["prediction"], [])
        self.assertEqual(result["inference_log"], [])

    def test_get_prediction_result_empty_tool_calls(self):
        """Test get_prediction_result with empty tool_calls"""
        handler = BFCLV3OutputHandler(perf_mode=False)
        
        output = FunctionCallOutput()
        output.success = True
        output.uuid = "test_uuid_empty"
        output.tool_calls = []
        output.inference_log = []
        
        result = handler.get_prediction_result(output)
        
        self.assertEqual(result["success"], True)
        self.assertEqual(result["prediction"], [])
        self.assertEqual(result["inference_log"], [])

    def test_get_prediction_result_empty_inference_log(self):
        """Test get_prediction_result with empty inference_log"""
        handler = BFCLV3OutputHandler(perf_mode=False)
        
        output = FunctionCallOutput()
        output.success = True
        output.uuid = "test_uuid_no_log"
        output.tool_calls = [{"function": "test", "arguments": {}}]
        output.inference_log = []
        
        result = handler.get_prediction_result(output)
        
        self.assertEqual(result["success"], True)
        self.assertEqual(len(result["prediction"]), 1)
        self.assertEqual(result["inference_log"], [])

    def test_get_prediction_result_invalid_type(self):
        """Test get_prediction_result raises AISBenchTypeError for non-FunctionCallOutput"""
        handler = BFCLV3OutputHandler(perf_mode=False)
        
        # Test with regular Output
        output = Output()
        output.success = True
        output.uuid = "test_uuid"
        
        with self.assertRaises(AISBenchTypeError) as context:
            handler.get_prediction_result(output)
        
        self.assertIn("FunctionCallOutput", str(context.exception))
        self.assertIn("Expected FunctionCallOutput", str(context.exception))

    def test_get_prediction_result_string_output(self):
        """Test get_prediction_result raises AISBenchTypeError for string output"""
        handler = BFCLV3OutputHandler(perf_mode=False)
        
        with self.assertRaises(AISBenchTypeError) as context:
            handler.get_prediction_result("string_output")
        
        self.assertIn("FunctionCallOutput", str(context.exception))

    def test_get_prediction_result_none_output(self):
        """Test get_prediction_result raises AISBenchTypeError for None output"""
        handler = BFCLV3OutputHandler(perf_mode=False)
        
        with self.assertRaises(AISBenchTypeError) as context:
            handler.get_prediction_result(None)
        
        self.assertIn("FunctionCallOutput", str(context.exception))

    def test_get_prediction_result_complex_tool_calls(self):
        """Test get_prediction_result with complex tool_calls structure"""
        handler = BFCLV3OutputHandler(perf_mode=False)
        
        output = FunctionCallOutput()
        output.success = True
        output.uuid = "test_uuid_complex"
        output.tool_calls = [
            {
                "function": "complex_function",
                "arguments": {
                    "param1": "value1",
                    "param2": [1, 2, 3],
                    "param3": {"nested": "dict"}
                }
            }
        ]
        output.inference_log = [
            {"step": 1, "action": "parse", "content": "test"},
            {"step": 2, "action": "execute", "result": "success"}
        ]
        
        result = handler.get_prediction_result(output)
        
        self.assertEqual(result["success"], True)
        self.assertEqual(len(result["prediction"]), 1)
        self.assertEqual(result["prediction"][0]["function"], "complex_function")
        self.assertEqual(len(result["inference_log"]), 2)

    def test_get_result_with_function_call_output(self):
        """Test get_result method with FunctionCallOutput (integration test)"""
        handler = BFCLV3OutputHandler(perf_mode=False)
        conn = sqlite3.connect(":memory:")
        
        output = FunctionCallOutput()
        output.success = True
        output.uuid = "test_uuid_integration"
        output.tool_calls = [{"function": "test", "arguments": {}}]
        output.inference_log = []
        
        result = handler.get_result(conn, "data_abbr", "test_input", output, "test_gold")
        
        self.assertEqual(result["success"], True)
        self.assertEqual(result["uuid"], "test_uuid_integration")
        self.assertEqual(result["prediction"], output.tool_calls)
        self.assertEqual(result["inference_log"], [])
        
        conn.close()

    def test_get_result_with_failed_function_call_output(self):
        """Test get_result method with failed FunctionCallOutput"""
        handler = BFCLV3OutputHandler(perf_mode=False)
        conn = sqlite3.connect(":memory:")
        
        output = FunctionCallOutput()
        output.success = False
        output.uuid = "test_uuid_failed_integration"
        output.tool_calls = []
        output.inference_log = []
        output.error_info = "Integration test error"
        
        result = handler.get_result(conn, "data_abbr", "test_input", output, "test_gold")
        
        self.assertEqual(result["success"], False)
        self.assertIn("error_info", result)
        self.assertEqual(result["error_info"], "Integration test error")
        self.assertFalse(handler.all_success)
        
        conn.close()

    def test_get_result_perf_mode(self):
        """Test get_result in perf_mode with FunctionCallOutput"""
        handler = BFCLV3OutputHandler(perf_mode=True)
        conn = sqlite3.connect(":memory:")
        
        output = FunctionCallOutput(perf_mode=True)
        output.success = True
        output.uuid = "test_uuid_perf"
        output.tool_calls = [{"function": "test", "arguments": {}}]
        output.inference_log = []
        output.time_points = [1.0, 2.0, 3.0]
        
        # Mock get_metrics method
        output.get_metrics = mock.Mock(return_value={
            "latency": 0.1,
            "throughput": 100,
            "time_points": [1.0, 2.0, 3.0]
        })
        
        handler._extract_and_write_arrays = mock.Mock(return_value={
            "latency": 0.1,
            "throughput": 100
        })
        
        result = handler.get_result(conn, "data_abbr", "test_input", output, "test_gold")
        
        # In perf_mode, get_result should call get_metrics
        self.assertIn("latency", result)
        handler._extract_and_write_arrays.assert_called_once()
        
        conn.close()


if __name__ == '__main__':
    unittest.main()
