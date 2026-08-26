import unittest
from unittest import mock
import tempfile
import sqlite3

from ais_bench.benchmark.openicl.icl_inferencer.output_handler.gen_inferencer_output_handler import GenInferencerOutputHandler
from ais_bench.benchmark.models.output import Output


class TestGenInferencerOutputHandler(unittest.TestCase):
    def test_init(self):
        """Test initialization"""
        handler = GenInferencerOutputHandler(perf_mode=False, save_every=10)
        self.assertFalse(handler.perf_mode)
        self.assertEqual(handler.save_every, 10)
        self.assertTrue(handler.all_success)

    def test_init_perf_mode(self):
        """Test initialization with perf_mode=True"""
        handler = GenInferencerOutputHandler(perf_mode=True, save_every=5)
        self.assertTrue(handler.perf_mode)
        self.assertEqual(handler.save_every, 5)

    def test_get_result_perf_mode(self):
        """Test get_result with perf_mode=True (lines 57-61)"""
        handler = GenInferencerOutputHandler(perf_mode=True)
        conn = sqlite3.connect(":memory:")
        
        output = Output()
        output.success = True
        output.uuid = "test_uuid"
        output.get_metrics = mock.Mock(return_value={"latency": 0.1, "throughput": 100})
        
        handler._extract_and_write_arrays = mock.Mock(return_value={"latency": 0.1, "throughput": 100})
        
        result = handler.get_result(conn, "data_abbr", "input", output, "gold")
        self.assertIn("latency", result)
        handler._extract_and_write_arrays.assert_called_once()
        
        conn.close()

    def test_get_result_accuracy_mode_string_output(self):
        """Test get_result with accuracy mode and string output (lines 64-77)
        
        Note: The code now properly handles string output by generating a UUID
        when output is not an Output object (line 70).
        """
        handler = GenInferencerOutputHandler(perf_mode=False)
        conn = sqlite3.connect(":memory:")
        
        # Test with string output - should work now with UUID generation
        string_output = "predicted_text"
        
        result = handler.get_result(conn, "data_abbr", "input", string_output, "gold")
        self.assertEqual(result["success"], True)
        self.assertIn("uuid", result)
        # UUID should be generated (8 characters from uuid.uuid4().hex[:8])
        self.assertEqual(len(result["uuid"]), 8)
        self.assertEqual(result["prediction"], "predicted_text")
        self.assertEqual(result["origin_prompt"], "input")
        self.assertEqual(result["gold"], "gold")
        
        conn.close()

    def test_get_result_accuracy_mode_output_object(self):
        """Test get_result with accuracy mode and Output object (lines 64-76)"""
        handler = GenInferencerOutputHandler(perf_mode=False)
        conn = sqlite3.connect(":memory:")
        
        output = Output()
        output.success = True
        output.uuid = "test_uuid"
        output.get_prediction = mock.Mock(return_value="predicted_text")
        
        # Mock get_prediction_result to return expected result
        handler.get_prediction_result = mock.Mock(return_value={
            "success": True,
            "uuid": "test_uuid",
            "prediction": "predicted_text",
            "origin_prompt": "input",
            "gold": "gold"
        })
        
        result = handler.get_result(conn, "data_abbr", "input", output, "gold")
        self.assertEqual(result["success"], True)
        self.assertEqual(result["uuid"], "test_uuid")
        self.assertEqual(result["prediction"], "predicted_text")
        self.assertEqual(result["origin_prompt"], "input")
        self.assertEqual(result["gold"], "gold")
        
        conn.close()

    def test_get_result_with_failure(self):
        """Test get_result with failed output (lines 82-90)"""
        handler = GenInferencerOutputHandler(perf_mode=False)
        conn = sqlite3.connect(":memory:")
        
        output = Output()
        output.success = False
        output.uuid = "test_uuid"
        output.error_info = "Test error"
        output.get_prediction = mock.Mock(return_value="")
        
        # Mock get_prediction_result to return failed result
        handler.get_prediction_result = mock.Mock(return_value={
            "success": False,
            "uuid": "test_uuid",
            "prediction": "",
            "origin_prompt": "input",
            "gold": "gold"
        })
        
        result = handler.get_result(conn, "data_abbr", "input", output, "gold")
        self.assertEqual(result["success"], False)
        self.assertIn("error_info", result)
        self.assertEqual(result["error_info"], "Test error")
        self.assertFalse(handler.all_success)
        
        conn.close()

    def test_get_result_with_failure_no_error_info(self):
        """Test get_result with failed output but no error_info (lines 87-90)"""
        handler = GenInferencerOutputHandler(perf_mode=False)
        conn = sqlite3.connect(":memory:")
        
        output = Output()
        output.success = False
        output.uuid = "test_uuid"
        # No error_info attribute
        output.get_prediction = mock.Mock(return_value="")
        
        # Mock get_prediction_result to return failed result
        handler.get_prediction_result = mock.Mock(return_value={
            "success": False,
            "uuid": "test_uuid",
            "prediction": "",
            "origin_prompt": "input",
            "gold": "gold"
        })
        
        result = handler.get_result(conn, "data_abbr", "input", output, "gold")
        self.assertEqual(result["success"], False)
        self.assertFalse(handler.all_success)
        
        conn.close()

    def test_get_result_without_gold(self):
        """Test get_result without gold parameter"""
        handler = GenInferencerOutputHandler(perf_mode=False)
        conn = sqlite3.connect(":memory:")
        
        output = Output()
        output.success = True
        output.uuid = "test_uuid"
        output.get_prediction = mock.Mock(return_value="predicted_text")
        
        # Mock get_prediction_result to return result without gold
        handler.get_prediction_result = mock.Mock(return_value={
            "success": True,
            "uuid": "test_uuid",
            "prediction": "predicted_text",
            "origin_prompt": "input"
        })
        
        result = handler.get_result(conn, "data_abbr", "input", output, None)
        self.assertNotIn("gold", result)
        
        conn.close()
    
    def test_get_result_string_output_without_gold(self):
        """Test get_result with string output and without gold parameter"""
        handler = GenInferencerOutputHandler(perf_mode=False)
        conn = sqlite3.connect(":memory:")
        
        string_output = "predicted_text"
        result = handler.get_result(conn, "data_abbr", "input", string_output, None)
        
        self.assertEqual(result["success"], True)
        self.assertIn("uuid", result)
        self.assertEqual(len(result["uuid"]), 8)
        self.assertEqual(result["prediction"], "predicted_text")
        self.assertEqual(result["origin_prompt"], "input")
        self.assertNotIn("gold", result)
        
        conn.close()
    
    def test_get_result_string_output_uuid_uniqueness(self):
        """Test that string outputs generate unique UUIDs"""
        handler = GenInferencerOutputHandler(perf_mode=False)
        conn = sqlite3.connect(":memory:")
        
        string_output = "predicted_text"
        result1 = handler.get_result(conn, "data_abbr", "input1", string_output, "gold1")
        result2 = handler.get_result(conn, "data_abbr", "input2", string_output, "gold2")
        
        # UUIDs should be different for different calls
        self.assertNotEqual(result1["uuid"], result2["uuid"])
        # But both should be 8 characters
        self.assertEqual(len(result1["uuid"]), 8)
        self.assertEqual(len(result2["uuid"]), 8)
        
        conn.close()

    def test_get_result_perf_mode_with_output_object(self):
        """Test get_result perf_mode with Output object (lines 57-61)"""
        handler = GenInferencerOutputHandler(perf_mode=True)
        conn = sqlite3.connect(":memory:")
        
        output = Output()
        output.success = True
        output.uuid = "test_uuid"
        output.get_metrics = mock.Mock(return_value={"latency": 0.1, "throughput": 100})
        
        handler._extract_and_write_arrays = mock.Mock(return_value={"latency": 0.1, "throughput": 100})
        
        result = handler.get_result(conn, "data_abbr", "input", output, "gold")
        self.assertIn("latency", result)
        handler._extract_and_write_arrays.assert_called_once()
        
        conn.close()

    def test_get_result_perf_mode_with_string_output(self):
        """Test get_result perf_mode with string output (lines 64-77)
        
        Note: In perf_mode, if output is a string (not Output object), the code
        goes to the else branch (line 64) and now properly handles string output
        by generating a UUID (line 70).
        """
        handler = GenInferencerOutputHandler(perf_mode=True)
        conn = sqlite3.connect(":memory:")
        
        # Test with string output in perf_mode
        # Since output is not Output, it goes to else branch (line 64)
        # and now properly generates UUID for string output
        string_output = "predicted_text"
        
        result = handler.get_result(conn, "data_abbr", "input", string_output, "gold")
        self.assertEqual(result["success"], True)
        self.assertIn("uuid", result)
        # UUID should be generated (8 characters)
        self.assertEqual(len(result["uuid"]), 8)
        self.assertEqual(result["prediction"], "predicted_text")
        self.assertEqual(result["origin_prompt"], "input")
        self.assertEqual(result["gold"], "gold")
        
        conn.close()

    def test_get_result_accuracy_mode_failure_no_error_info(self):
        """Test get_result accuracy mode failure without error_info (lines 82-90)"""
        handler = GenInferencerOutputHandler(perf_mode=False)
        conn = sqlite3.connect(":memory:")

        output = Output()
        output.success = False
        output.uuid = "test_uuid"
        output.get_prediction = mock.Mock(return_value="")

        # Delete error_info if it exists (Output may have it as default attribute)
        if hasattr(output, "error_info"):
            delattr(output, "error_info")

        # Mock get_prediction_result to return failed result
        handler.get_prediction_result = mock.Mock(return_value={
            "success": False,
            "uuid": "test_uuid",
            "prediction": "",
            "origin_prompt": "input",
            "gold": "gold"
        })

        result = handler.get_result(conn, "data_abbr", "input", output, "gold")
        self.assertEqual(result["success"], False)
        self.assertFalse(handler.all_success)
        # Should not have error_info when it doesn't exist
        self.assertNotIn("error_info", result)

        conn.close()

    def test_get_prediction_result_with_tokens(self):
        """测试 get_prediction_result 始终输出 input_tokens 和 output_tokens"""
        handler = GenInferencerOutputHandler(perf_mode=False)

        output = Output()
        output.success = True
        output.uuid = "test_uuid"
        output.input_tokens = 100
        output.output_tokens = 50
        output.get_prediction = mock.Mock(return_value="predicted_text")

        result = handler.get_prediction_result(output, "gold", "input", "data_abbr")

        self.assertEqual(result["input_tokens"], 100)
        self.assertEqual(result["output_tokens"], 50)

    def test_get_prediction_result_with_logprobs(self):
        """测试 get_prediction_result 在有 logprobs 数据时输出"""
        handler = GenInferencerOutputHandler(perf_mode=False)

        output = Output()
        output.success = True
        output.uuid = "test_uuid"
        output.input_tokens = 100
        output.output_tokens = 50
        output.origin_logprobs = [{"token": "B", "logprob": -0.5, "top_logprobs": [{"token": "B", "logprob": -0.5}]}]
        output.get_prediction = mock.Mock(return_value="predicted_text")

        result = handler.get_prediction_result(output, "gold", "input", "data_abbr")

        self.assertEqual(result["origin_logprobs"], [{"token": "B", "logprob": -0.5, "top_logprobs": [{"token": "B", "logprob": -0.5}]}])

    def test_get_prediction_result_without_logprobs(self):
        """测试 get_prediction_result 在无 logprobs 数据时不输出 logprobs 字段"""
        handler = GenInferencerOutputHandler(perf_mode=False)

        output = Output()
        output.success = True
        output.uuid = "test_uuid"
        output.input_tokens = 100
        output.output_tokens = 50
        output.origin_logprobs = []
        output.get_prediction = mock.Mock(return_value="predicted_text")

        result = handler.get_prediction_result(output, "gold", "input", "data_abbr")

        self.assertNotIn("origin_logprobs", result)

    def test_get_prediction_result_string_output_no_tokens(self):
        """测试 get_prediction_result 在 string output 时不输出 token 字段"""
        handler = GenInferencerOutputHandler(perf_mode=False)

        result = handler.get_prediction_result("predicted_text", "gold", "input", "data_abbr")

        self.assertNotIn("input_tokens", result)
        self.assertNotIn("output_tokens", result)
        self.assertNotIn("origin_logprobs", result)


if __name__ == '__main__':
    unittest.main()
