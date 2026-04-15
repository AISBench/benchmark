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

    def test_deprecated_module_exports_current_api(self):
        """Deprecated module should expose the current public API."""
        self.assertTrue(hasattr(mp, "extract_non_reasoning_content"))
        self.assertTrue(callable(mp.extract_non_reasoning_content))

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

