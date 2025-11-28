import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import importlib

from ais_bench.benchmark.utils.file.load_tokenizer import load_tokenizer
from ais_bench.benchmark.utils.logging.exceptions import FileOperationError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES


# Import the module object to patch module-level attributes reliably
lt_module = importlib.import_module("ais_bench.benchmark.utils.file.load_tokenizer")


class TestLoadTokenizer(unittest.TestCase):
    """Tests for load_tokenizer happy paths and failures."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.tokenizer_path = os.path.join(self.test_dir, "tokenizer")
        os.makedirs(self.tokenizer_path)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_tokenizer_path_not_found(self):
        """Raises FileOperationError when path doesn't exist."""
        nonexistent_path = "/nonexistent/tokenizer/path"

        with self.assertRaises(FileOperationError) as cm:
            load_tokenizer(nonexistent_path)

        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.TOKENIZER_PATH_NOT_FOUND.full_code)
        self.assertIn(nonexistent_path, str(cm.exception))

    @patch.object(lt_module, "AutoTokenizer")
    def test_load_tokenizer_success(self, mock_auto_tokenizer):
        """Successfully loads tokenizer"""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        result = load_tokenizer(self.tokenizer_path)

        self.assertIs(result, mock_tokenizer)
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(self.tokenizer_path)

    @patch.object(lt_module, "AutoTokenizer")
    def test_load_tokenizer_loading_fails_value_error(self, mock_auto_tokenizer):
        """Wraps ValueError from AutoTokenizer in FileOperationError"""
        error_msg = "Invalid tokenizer configuration"
        mock_auto_tokenizer.from_pretrained.side_effect = ValueError(error_msg)

        with self.assertRaises(FileOperationError) as cm:
            load_tokenizer(self.tokenizer_path)

        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.TOKENIZER_LOAD_FAILED.full_code)
        exc_str = str(cm.exception)
        self.assertIn(self.tokenizer_path, exc_str)
        self.assertIn("ValueError", exc_str)
        self.assertIn(error_msg, exc_str)

    @patch.object(lt_module, "AutoTokenizer")
    def test_load_tokenizer_loading_fails_runtime_error(self, mock_auto_tokenizer):
        """Wraps RuntimeError and chains as __cause__."""
        error_msg = "Corrupted tokenizer files"
        mock_auto_tokenizer.from_pretrained.side_effect = RuntimeError(error_msg)

        with self.assertRaises(FileOperationError) as cm:
            load_tokenizer(self.tokenizer_path)

        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.TOKENIZER_LOAD_FAILED.full_code)
        self.assertIsInstance(cm.exception.__cause__, RuntimeError)
        self.assertIn("RuntimeError", str(cm.exception))
        self.assertIn(error_msg, str(cm.exception))

    @patch.object(lt_module, "AutoTokenizer")
    def test_load_tokenizer_loading_fails_os_error(self, mock_auto_tokenizer):
        """Wraps OSError"""
        error_msg = "Permission denied"
        mock_auto_tokenizer.from_pretrained.side_effect = OSError(error_msg)

        with self.assertRaises(FileOperationError) as cm:
            load_tokenizer(self.tokenizer_path)

        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.TOKENIZER_LOAD_FAILED.full_code)
        self.assertIn("OSError", str(cm.exception))

    @patch.object(lt_module, "AutoTokenizer")
    def test_load_tokenizer_loading_fails_generic_exception(self, mock_auto_tokenizer):
        """Wraps generic Exception."""
        error_msg = "Unknown error"
        mock_auto_tokenizer.from_pretrained.side_effect = Exception(error_msg)

        with self.assertRaises(FileOperationError) as cm:
            load_tokenizer(self.tokenizer_path)

        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.TOKENIZER_LOAD_FAILED.full_code)
        self.assertIn("Exception", str(cm.exception))
        self.assertIn(error_msg, str(cm.exception))

    @patch.object(lt_module, "AutoTokenizer")
    def test_load_tokenizer_with_absolute_path(self, mock_auto_tokenizer):
        """Accepts absolute path and forwards it to AutoTokenizer."""
        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        abs_path = os.path.abspath(self.tokenizer_path)
        result = load_tokenizer(abs_path)

        self.assertIs(result, mock_tokenizer)
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(abs_path)

class TestLoadTokenizerEdgeCases(unittest.TestCase):
    """Edge case tests for load_tokenizer."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_load_tokenizer_empty_path(self):
        with self.assertRaises(FileOperationError) as cm:
            load_tokenizer("")

        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.TOKENIZER_PATH_NOT_FOUND.full_code)

    def test_load_tokenizer_relative_path_not_exists(self):
        with self.assertRaises(FileOperationError) as cm:
            load_tokenizer("./nonexistent/tokenizer")

        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.TOKENIZER_PATH_NOT_FOUND.full_code)

    @patch.object(lt_module, "AutoTokenizer")
    def test_load_tokenizer_file_instead_of_directory(self, mock_auto_tokenizer):
        """If AutoTokenizer accepts a file path, we return that tokenizer."""
        # Create a file instead of a directory
        file_path = os.path.join(self.test_dir, "tokenizer_file.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("test")

        mock_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        result = load_tokenizer(file_path)

        self.assertIs(result, mock_tokenizer)


if __name__ == "__main__":
    unittest.main()

