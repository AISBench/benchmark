import unittest

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.utils.core.types import (
    check_type_list,
    check_dataset,
    check_list,
    check_str,
    check_dict,
    check_type,
    check_meta_json_dict,
    check_output_config_from_meta_json,
)
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchConfigError,
    AISBenchInvalidTypeException,
)
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES


class TestTypeValidationUtilities(unittest.TestCase):
    """Test suite for type validation utilities in utils.core.types"""

    # ==================== Test check_type_list ====================

    def test_check_type_list_valid_match(self):
        """Test check_type_list with valid type match"""
        result = check_type_list("hello", [str, int])
        self.assertEqual(result, "hello")

        result = check_type_list(42, [str, int])
        self.assertEqual(result, 42)

    def test_check_type_list_none_match(self):
        """Test check_type_list with None type"""
        result = check_type_list(None, [None, str])
        self.assertIsNone(result)

    def test_check_type_list_invalid_type(self):
        """Test check_type_list raises exception for invalid type"""
        with self.assertRaises(AISBenchInvalidTypeException) as cm:
            check_type_list(42, [str, list])
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.INVALID_TYPE.full_code)

    # ==================== Test check_dataset ====================

    def test_check_dataset_valid_dataset(self):
        """Test check_dataset with valid Dataset"""
        mock_dataset = Dataset.from_dict({"col": [1, 2, 3]})
        result = check_dataset(mock_dataset)
        self.assertIsInstance(result, Dataset)

    def test_check_dataset_valid_dataset_dict(self):
        """Test check_dataset with valid DatasetDict"""
        mock_dataset_dict = DatasetDict({
            "train": Dataset.from_dict({"col": [1, 2]}),
            "test": Dataset.from_dict({"col": [3, 4]})
        })
        result = check_dataset(mock_dataset_dict)
        self.assertIsInstance(result, DatasetDict)

    def test_check_dataset_invalid_type(self):
        """Test check_dataset raises exception for invalid type"""
        with self.assertRaises(AISBenchInvalidTypeException) as cm:
            check_dataset({"not": "a dataset"})
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.INVALID_DATASET_TYPE.full_code)

    # ==================== Test check_list ====================

    def test_check_list_valid(self):
        """Test check_list with valid list"""
        result = check_list([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])

    def test_check_list_invalid_type(self):
        """Test check_list raises exception for non-list"""
        with self.assertRaises(AISBenchInvalidTypeException) as cm:
            check_list((1, 2, 3))
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.INVALID_LIST_TYPE.full_code)

    # ==================== Test check_str ====================

    def test_check_str_valid(self):
        """Test check_str with valid string"""
        result = check_str("hello")
        self.assertEqual(result, "hello")
    
    def test_check_str_invalid_type(self):
        """Test check_str raises exception for non-string"""
        with self.assertRaises(AISBenchInvalidTypeException) as cm:
            check_str(123)
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.INVALID_STRING_TYPE.full_code)

    # ==================== Test check_dict ====================

    def test_check_dict_valid(self):
        """Test check_dict with valid dictionary"""
        result = check_dict({"key": "value"})
        self.assertEqual(result, {"key": "value"})

    def test_check_dict_invalid_type(self):
        """Test check_dict raises exception for non-dict"""
        with self.assertRaises(AISBenchInvalidTypeException) as cm:
            check_dict([1, 2, 3])
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.INVALID_DICT_TYPE.full_code)

    # ==================== Test check_type ====================

    def test_check_type_valid(self):
        """Test check_type with valid type match"""
        result = check_type("hello", str)
        self.assertEqual(result, "hello")
        
        result = check_type(42, int)
        self.assertEqual(result, 42)
        
        result = check_type([1, 2], list)
        self.assertEqual(result, [1, 2])

    def test_check_type_none_valid(self):
        """Test check_type with None when NoneType expected"""
        result = check_type(None, type(None))
        self.assertIsNone(result)

    def test_check_type_none_invalid(self):
        """Test check_type raises exception when None but expecting value"""
        with self.assertRaises(AISBenchInvalidTypeException) as cm:
            check_type(None, str)
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.INVALID_TYPE.full_code)

    def test_check_type_invalid_type_specifier(self):
        """Test check_type raises exception for invalid type specifier"""
        with self.assertRaises(AISBenchInvalidTypeException) as cm:
            check_type("hello", "not a type")
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.INVALID_TYPE_SPECIFIER.full_code)

    def test_check_type_mismatch(self):
        """Test check_type raises exception for type mismatch"""
        with self.assertRaises(AISBenchInvalidTypeException) as cm:
            check_type("hello", int)
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.TYPE_MISMATCH.full_code)

    def test_check_type_long_repr_truncated(self):
        """Test check_type truncates long object representations"""
        long_string = "x" * 200
        with self.assertRaises(AISBenchInvalidTypeException) as cm:
            check_type(long_string, int)
        
        # Error message should contain truncated repr
        error_msg = str(cm.exception)
        self.assertIn("...", error_msg)

    # ==================== Test check_meta_json_dict ====================

    def test_check_meta_json_dict_valid_minimal(self):
        """Test check_meta_json_dict with minimal valid config"""
        config = {
            "sampling_mode": "random"
        }
        result = check_meta_json_dict(config)
        self.assertEqual(result, config)

    def test_check_meta_json_dict_valid_with_request_count(self):
        """Test check_meta_json_dict with valid request_count"""
        config = {
            "request_count": 100,
            "sampling_mode": "sequential"
        }
        result = check_meta_json_dict(config)
        self.assertEqual(result, config)
        
        # Test string request_count
        config_str = {
            "request_count": "50",
            "sampling_mode": "random"
        }
        result = check_meta_json_dict(config_str)
        self.assertEqual(result, config_str)

    def test_check_meta_json_dict_invalid_request_count(self):
        """Test check_meta_json_dict raises exception for invalid request_count"""
        config = {
            "request_count": -1,
            "sampling_mode": "random"
        }
        with self.assertRaises(AISBenchConfigError) as cm:
            check_meta_json_dict(config)
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.INVALID_REQUEST_COUNT.full_code)

    def test_check_meta_json_dict_illegal_keys(self):
        """Test check_meta_json_dict raises exception for illegal keys"""
        config = {
            "sampling_mode": "random",
            "illegal_key": "value"
        }
        with self.assertRaises(AISBenchConfigError) as cm:
            check_meta_json_dict(config)
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.ILLEGAL_KEYS_IN_CONFIG.full_code)

    def test_check_meta_json_dict_type_mismatch(self):
        """Test check_meta_json_dict raises exception for type mismatch"""
        config = {
            "sampling_mode": 123  # Should be string
        }
        with self.assertRaises(AISBenchInvalidTypeException) as cm:
            check_meta_json_dict(config)
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.TYPE_MISMATCH.full_code)

    def test_check_meta_json_dict_nested_output_config(self):
        """Test check_meta_json_dict with nested output_config"""
        config = {
            "output_config": {
                "method": "uniform",
                "params": {
                    "min_value": 10,
                    "max_value": 100
                }
            },
            "sampling_mode": "random"
        }
        result = check_meta_json_dict(config)
        self.assertEqual(result, config)

    # ==================== Test check_output_config_from_meta_json ====================

    def test_check_output_config_no_config(self):
        """Test check_output_config_from_meta_json returns False when no output_config"""
        config = {}
        result = check_output_config_from_meta_json(config)
        self.assertFalse(result)

        config = {"sampling_mode": "random"}
        result = check_output_config_from_meta_json(config)
        self.assertFalse(result)

    def test_check_output_config_missing_params(self):
        """Test check_output_config_from_meta_json raises exception when params missing"""
        config = {
            "output_config": {
                "method": "uniform"
            }
        }
        with self.assertRaises(AISBenchConfigError) as cm:
            check_output_config_from_meta_json(config)
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.MISSING_PARAMS.full_code)

    def test_check_output_config_uniform_valid(self):
        """Test check_output_config_from_meta_json with valid uniform distribution"""
        config = {
            "output_config": {
                "method": "uniform",
                "params": {
                    "min_value": 10,
                    "max_value": 100
                }
            }
        }
        result = check_output_config_from_meta_json(config)
        self.assertTrue(result)

        # Test with string values
        config_str = {
            "output_config": {
                "method": "uniform",
                "params": {
                    "min_value": "10",
                    "max_value": "100"
                }
            }
        }
        result = check_output_config_from_meta_json(config_str)
        self.assertTrue(result)

    def test_check_output_config_uniform_invalid_values(self):
        """Test check_output_config_from_meta_json with invalid min/max values"""
        config = {
            "output_config": {
                "method": "uniform",
                "params": {
                    "min_value": -1,
                    "max_value": 100
                }
            }
        }
        with self.assertRaises(AISBenchConfigError) as cm:
            check_output_config_from_meta_json(config)
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.INVALID_MIN_MAX_VALUE.full_code)

    def test_check_output_config_uniform_min_greater_than_max(self):
        """Test check_output_config_from_meta_json with min > max"""
        config = {
            "output_config": {
                "method": "uniform",
                "params": {
                    "min_value": 100,
                    "max_value": 10
                }
            }
        }
        with self.assertRaises(AISBenchConfigError) as cm:
            check_output_config_from_meta_json(config)
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.MIN_GREATER_THAN_MAX.full_code)

    def test_check_output_config_uniform_missing_params(self):
        """Test check_output_config_from_meta_json with missing min/max"""
        config = {
            "output_config": {
                "method": "uniform",
                "params": {
                    "min_value": 10
                }
            }
        }
        with self.assertRaises(AISBenchConfigError) as cm:
            check_output_config_from_meta_json(config)
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.MISSING_PARAMS.full_code)

    def test_check_output_config_percentage_valid(self):
        """Test check_output_config_from_meta_json with valid percentage distribution"""
        config = {
            "output_config": {
                "method": "percentage",
                "params": {
                    "percentage_distribute": [[1000, 0.5], [500, 0.5]]
                }
            }
        }
        result = check_output_config_from_meta_json(config)
        self.assertTrue(result)

        # Test single percentage
        config_single = {
            "output_config": {
                "method": "percentage",
                "params": {
                    "percentage_distribute": [[2000, 1.0]]
                }
            }
        }
        result = check_output_config_from_meta_json(config_single)
        self.assertTrue(result)

    def test_check_output_config_percentage_missing_param(self):
        """Test check_output_config_from_meta_json with missing percentage_distribute"""
        config = {
            "output_config": {
                "method": "percentage",
                "params": {}
            }
        }
        with self.assertRaises(AISBenchConfigError) as cm:
            check_output_config_from_meta_json(config)
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.MISSING_PARAMS.full_code)

    def test_check_output_config_percentage_invalid_format(self):
        """Test check_output_config_from_meta_json with invalid percentage format"""
        # Sum not equal to 1
        config = {
            "output_config": {
                "method": "percentage",
                "params": {
                    "percentage_distribute": [[1000, 0.3], [500, 0.2]]
                }
            }
        }
        with self.assertRaises(AISBenchConfigError) as cm:
            check_output_config_from_meta_json(config)
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.INVALID_PERCENTAGE_DISTRIBUTE.full_code)

        # Percentage > 1
        config_over = {
            "output_config": {
                "method": "percentage",
                "params": {
                    "percentage_distribute": [[1000, 1.5]]
                }
            }
        }
        with self.assertRaises(AISBenchConfigError):
            check_output_config_from_meta_json(config_over)

        # max_tokens <= 0
        config_zero = {
            "output_config": {
                "method": "percentage",
                "params": {
                    "percentage_distribute": [[0, 1.0]]
                }
            }
        }
        with self.assertRaises(AISBenchConfigError):
            check_output_config_from_meta_json(config_zero)

        # Invalid structure
        config_bad_struct = {
            "output_config": {
                "method": "percentage",
                "params": {
                    "percentage_distribute": [[1000]]
                }
            }
        }
        with self.assertRaises(AISBenchConfigError):
            check_output_config_from_meta_json(config_bad_struct)

    def test_check_output_config_unsupported_method(self):
        """Test check_output_config_from_meta_json with unsupported method"""
        config = {
            "output_config": {
                "method": "exponential",
                "params": {
                    "some_param": "value"
                }
            }
        }
        with self.assertRaises(AISBenchConfigError) as cm:
            check_output_config_from_meta_json(config)
        
        self.assertEqual(cm.exception.error_code_str, UTILS_CODES.UNSUPPORTED_DISTRIBUTION_METHOD.full_code)


if __name__ == "__main__":
    unittest.main()
