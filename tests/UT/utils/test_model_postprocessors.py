import unittest
import sys
from unittest.mock import patch, MagicMock, Mock

# Create mock modules and patch sys.modules before importing
mock_naive = MagicMock()
mock_naive.NaiveExtractor = MagicMock
mock_naive.format_input_naive = lambda x: x

mock_xfinder_extractor = MagicMock()
mock_xfinder_extractor.Extractor = MagicMock

mock_xfinder_utils = MagicMock()
mock_xfinder_utils.DataProcessor = MagicMock
mock_xfinder_utils.convert_to_xfinder_format = lambda x, y: y

# Patch sys.modules
sys.modules['ais_bench.benchmark.utils.postprocess.postprocessors.naive'] = mock_naive
sys.modules['ais_bench.benchmark.utils.postprocess.postprocessors.xfinder.extractor'] = mock_xfinder_extractor
sys.modules['ais_bench.benchmark.utils.postprocess.postprocessors.xfinder.xfinder_utils'] = mock_xfinder_utils

# Now import after patching
from ais_bench.benchmark.utils import model_postprocessors as mp
from ais_bench.benchmark.registry import TEXT_POSTPROCESSORS


class TestModelPostprocessors(unittest.TestCase):
    """Tests for model_postprocessors module."""

    def setUp(self):
        self.test_preds = [
            {
                'question': 'What is 2+2?',
                'reference_answer': '4',
                'model_prediction': 'The answer is 4'
            },
            {
                'question': 'What is 3+3?',
                'reference_answer': '6',
                'model_prediction': 'The answer is 6'
            }
        ]

    @patch('ais_bench.benchmark.utils.model_postprocessors.Pool')
    @patch('ais_bench.benchmark.utils.model_postprocessors.NaiveExtractor')
    @patch('ais_bench.benchmark.utils.model_postprocessors.format_input_naive')
    def test_naive_model_postprocess(self, mock_format, mock_extractor_class, mock_pool):
        """Test naive_model_postprocess function."""
        # Setup mocks
        mock_format.return_value = self.test_preds
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor

        # Mock the extractor's methods
        def mock_gen_output(ori_data, extractor):
            results = []
            for item in ori_data:
                extracted = extractor.gen_output(item)
                results.append(extracted)
            return results

        mock_extractor.prepare_input.return_value = "prepared_input"
        mock_extractor.gen_output.return_value = "extracted_answer"

        # Mock Pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool.return_value.__exit__.return_value = False

        # Mock map to call the function directly
        def mock_map(func, batches):
            results = []
            for batch in batches:
                batch_result = func(batch)
                results.append(batch_result)
            return results

        mock_pool_instance.map = mock_map

        # Call the function
        result = mp.naive_model_postprocess(
            preds=self.test_preds,
            model_name="test_model",
            custom_instruction="Extract the answer",
            api_url="http://test.api",
            num_processes=2
        )

        # Verify
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        mock_format.assert_called_once_with(self.test_preds)
        mock_extractor_class.assert_called_once()

    @patch('ais_bench.benchmark.utils.model_postprocessors.Pool')
    @patch('ais_bench.benchmark.utils.model_postprocessors.Extractor')
    @patch('ais_bench.benchmark.utils.model_postprocessors.DataProcessor')
    @patch('ais_bench.benchmark.utils.model_postprocessors.convert_to_xfinder_format')
    def test_xfinder_postprocess(self, mock_convert, mock_data_processor_class,
                                 mock_extractor_class, mock_pool):
        """Test xfinder_postprocess function."""
        # Setup mocks
        mock_convert.return_value = self.test_preds
        mock_data_processor = MagicMock()
        mock_data_processor_class.return_value = mock_data_processor
        mock_data_processor.read_data.return_value = [
            {
                'key_answer_type': 'number',
                'standard_answer_range': '0-10',
                'correct_answer': '4'
            }
        ]

        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.prepare_input.return_value = "prepared_input"
        mock_extractor.gen_output.return_value = "extracted_answer"

        # Mock Pool
        mock_pool_instance = MagicMock()
        mock_pool.return_value.__enter__.return_value = mock_pool_instance
        mock_pool.return_value.__exit__.return_value = False

        def mock_map(func, batches):
            results = []
            for batch in batches:
                batch_result = func(batch)
                results.append(batch_result)
            return results

        mock_pool_instance.map = mock_map

        # Call the function
        result = mp.xfinder_postprocess(
            preds=self.test_preds,
            question_type="math",
            model_name="test_model",
            api_url="http://test.api"
        )

        # Verify
        self.assertIsInstance(result, list)
        mock_convert.assert_called_once()
        mock_data_processor_class.assert_called_once()
        mock_extractor_class.assert_called_once()

    def test_naive_model_postprocess_no_api_url(self):
        """Test naive_model_postprocess raises error when api_url is None."""
        with self.assertRaises(AssertionError) as cm:
            mp.naive_model_postprocess(
                preds=self.test_preds,
                model_name="test_model",
                custom_instruction="Extract",
                api_url=None
            )
        self.assertIn("api url", str(cm.exception).lower())

    def test_xfinder_postprocess_no_api_url(self):
        """Test xfinder_postprocess raises error when api_url is None."""
        with self.assertRaises(AssertionError) as cm:
            mp.xfinder_postprocess(
                preds=self.test_preds,
                question_type="math",
                model_name="test_model",
                api_url=None
            )
        self.assertIn("api url", str(cm.exception).lower())

    def test_list_decorator_with_list(self):
        """Test list_decorator with list input."""
        @mp.list_decorator
        def test_func(text):
            return text.upper()

        result = test_func(["hello", "world"])
        self.assertEqual(result, ["HELLO", "WORLD"])

    def test_list_decorator_with_string(self):
        """Test list_decorator with string input."""
        @mp.list_decorator
        def test_func(text):
            return text.upper()

        result = test_func("hello")
        self.assertEqual(result, "HELLO")

    def test_extract_non_reasoning_content_only_end_token(self):
        """Test extract_non_reasoning_content with only end token."""
        text = "This is a test.</think> How are you?"
        result = mp.extract_non_reasoning_content(text)
        self.assertEqual(result, "How are you?")

    def test_extract_non_reasoning_content_both_tokens(self):
        """Test extract_non_reasoning_content with both tokens."""
        text = "Start<think>reasoning here</think> End"
        result = mp.extract_non_reasoning_content(text)
        self.assertEqual(result, "Start End")

    def test_extract_non_reasoning_content_no_tokens(self):
        """Test extract_non_reasoning_content with no tokens."""
        text = "Plain text without tokens"
        result = mp.extract_non_reasoning_content(text)
        self.assertEqual(result, "Plain text without tokens")

    def test_extract_non_reasoning_content_list_input(self):
        """Test extract_non_reasoning_content with list input."""
        texts = [
            "Start<think>reasoning</think> End",
            "Test</think> Result"
        ]
        result = mp.extract_non_reasoning_content(texts)
        self.assertEqual(result, ["Start End", "Result"])

    def test_extract_non_reasoning_content_custom_tokens(self):
        """Test extract_non_reasoning_content with custom tokens."""
        text = "A <r>reason</r> B"
        result = mp.extract_non_reasoning_content(
            text,
            think_start_token="<r>",
            think_end_token="</r>"
        )
        self.assertEqual(result, "A  B")

    def test_extract_non_reasoning_content_multiple_pairs(self):
        """Test extract_non_reasoning_content with multiple tag pairs."""
        text = "A<think>x</think>B<think>y</think>C"
        result = mp.extract_non_reasoning_content(text)
        self.assertEqual(result, "ABC")

    def test_extract_non_reasoning_content_non_string(self):
        """Test extract_non_reasoning_content with non-string input."""
        result = mp.extract_non_reasoning_content(123)
        self.assertEqual(result, 123)

    def test_extract_non_reasoning_content_registered(self):
        """Test that extract_non_reasoning_content is registered."""
        # Check if the function is registered in TEXT_POSTPROCESSORS
        # The registry uses 'in' operator or get() method
        self.assertIn('extract-non-reasoning-content', TEXT_POSTPROCESSORS)


if __name__ == "__main__":
    unittest.main()

