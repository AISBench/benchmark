import unittest
from unittest.mock import patch

from ais_bench.benchmark.utils.core.valid_global_consts import (
    get_max_chunk_size,
    get_request_time_out,
)


class TestValidGlobalConsts(unittest.TestCase):
    """Test suite for global constants validation utilities"""

    # ==================== Test get_max_chunk_size ====================

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_max_chunk_size_valid_default(self, mock_consts):
        """Test get_max_chunk_size with valid default value"""
        mock_consts.MAX_CHUNK_SIZE = 2**16  # 65536
        result = get_max_chunk_size()
        self.assertEqual(result, 65536)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_max_chunk_size_valid_min(self, mock_consts):
        """Test get_max_chunk_size with minimum valid value"""
        mock_consts.MAX_CHUNK_SIZE = 1
        result = get_max_chunk_size()
        self.assertEqual(result, 1)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_max_chunk_size_valid_max(self, mock_consts):
        """Test get_max_chunk_size with maximum valid value"""
        mock_consts.MAX_CHUNK_SIZE = 2**24  # 16777216 (16MB)
        result = get_max_chunk_size()
        self.assertEqual(result, 2**24)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_max_chunk_size_valid_midrange(self, mock_consts):
        """Test get_max_chunk_size with mid-range value"""
        mock_consts.MAX_CHUNK_SIZE = 2**20  # 1MB
        result = get_max_chunk_size()
        self.assertEqual(result, 2**20)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.logger")
    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_max_chunk_size_invalid_type_string(self, mock_consts, mock_logger):
        """Test get_max_chunk_size with invalid string type"""
        mock_consts.MAX_CHUNK_SIZE = "65536"
        result = get_max_chunk_size()
        
        self.assertEqual(result, 2**16)  # Returns default
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("invalid", warning_msg.lower())
        self.assertIn("type: str", warning_msg)
        self.assertIn("64KB", warning_msg)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.logger")
    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_max_chunk_size_invalid_type_float(self, mock_consts, mock_logger):
        """Test get_max_chunk_size with invalid float type"""
        mock_consts.MAX_CHUNK_SIZE = 65536.0
        result = get_max_chunk_size()
        
        self.assertEqual(result, 2**16)  # Returns default
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("type: float", warning_msg)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.logger")
    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_max_chunk_size_invalid_type_none(self, mock_consts, mock_logger):
        """Test get_max_chunk_size with None"""
        mock_consts.MAX_CHUNK_SIZE = None
        result = get_max_chunk_size()
        
        self.assertEqual(result, 2**16)  # Returns default
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("type: NoneType", warning_msg)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.logger")
    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_max_chunk_size_below_range(self, mock_consts, mock_logger):
        """Test get_max_chunk_size with value below minimum"""
        mock_consts.MAX_CHUNK_SIZE = 0
        result = get_max_chunk_size()
        
        self.assertEqual(result, 2**16)  # Returns default
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("out of range", warning_msg)
        self.assertIn("[1, 16777216]", warning_msg)
        self.assertIn("0", warning_msg)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.logger")
    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_max_chunk_size_negative(self, mock_consts, mock_logger):
        """Test get_max_chunk_size with negative value"""
        mock_consts.MAX_CHUNK_SIZE = -1000
        result = get_max_chunk_size()
        
        self.assertEqual(result, 2**16)  # Returns default
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("out of range", warning_msg)
        self.assertIn("-1000", warning_msg)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.logger")
    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_max_chunk_size_above_range(self, mock_consts, mock_logger):
        """Test get_max_chunk_size with value above maximum"""
        mock_consts.MAX_CHUNK_SIZE = 2**24 + 1  # Just over 16MB
        result = get_max_chunk_size()
        
        self.assertEqual(result, 2**16)  # Returns default
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("out of range", warning_msg)
        self.assertIn("16777217", warning_msg)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.logger")
    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_max_chunk_size_extremely_large(self, mock_consts, mock_logger):
        """Test get_max_chunk_size with extremely large value"""
        mock_consts.MAX_CHUNK_SIZE = 2**30  # 1GB
        result = get_max_chunk_size()
        
        self.assertEqual(result, 2**16)  # Returns default
        mock_logger.warning.assert_called_once()

    # ==================== Test get_request_time_out ====================

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_request_time_out_valid_none(self, mock_consts):
        """Test get_request_time_out with None (no timeout)"""
        mock_consts.REQUEST_TIME_OUT = None
        result = get_request_time_out()
        self.assertIsNone(result)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_request_time_out_valid_zero(self, mock_consts):
        """Test get_request_time_out with zero (immediate timeout)"""
        mock_consts.REQUEST_TIME_OUT = 0
        result = get_request_time_out()
        self.assertEqual(result, 0)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_request_time_out_valid_int(self, mock_consts):
        """Test get_request_time_out with valid integer"""
        mock_consts.REQUEST_TIME_OUT = 300  # 5 minutes
        result = get_request_time_out()
        self.assertEqual(result, 300)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_request_time_out_valid_float(self, mock_consts):
        """Test get_request_time_out with valid float"""
        mock_consts.REQUEST_TIME_OUT = 30.5
        result = get_request_time_out()
        self.assertEqual(result, 30.5)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_request_time_out_valid_max(self, mock_consts):
        """Test get_request_time_out with maximum value (24 hours)"""
        mock_consts.REQUEST_TIME_OUT = 3600 * 24  # 86400 seconds
        result = get_request_time_out()
        self.assertEqual(result, 86400)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_request_time_out_valid_one_hour(self, mock_consts):
        """Test get_request_time_out with 1 hour"""
        mock_consts.REQUEST_TIME_OUT = 3600
        result = get_request_time_out()
        self.assertEqual(result, 3600)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.logger")
    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_request_time_out_invalid_type_string(self, mock_consts, mock_logger):
        """Test get_request_time_out with invalid string type"""
        mock_consts.REQUEST_TIME_OUT = "300"
        result = get_request_time_out()
        
        self.assertIsNone(result)  # Returns default
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("invalid", warning_msg.lower())
        self.assertIn("type: str", warning_msg)
        self.assertIn("int/float", warning_msg)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.logger")
    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_request_time_out_invalid_type_list(self, mock_consts, mock_logger):
        """Test get_request_time_out with invalid list type"""
        mock_consts.REQUEST_TIME_OUT = [300]
        result = get_request_time_out()
        
        self.assertIsNone(result)  # Returns default
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("type: list", warning_msg)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.logger")
    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_request_time_out_invalid_type_dict(self, mock_consts, mock_logger):
        """Test get_request_time_out with invalid dict type"""
        mock_consts.REQUEST_TIME_OUT = {"timeout": 300}
        result = get_request_time_out()
        
        self.assertIsNone(result)  # Returns default
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("type: dict", warning_msg)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.logger")
    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_request_time_out_negative(self, mock_consts, mock_logger):
        """Test get_request_time_out with negative value"""
        mock_consts.REQUEST_TIME_OUT = -10
        result = get_request_time_out()
        
        self.assertIsNone(result)  # Returns default
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("out of range", warning_msg)
        self.assertIn("[0, 86400]", warning_msg)
        self.assertIn("-10", warning_msg)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.logger")
    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_request_time_out_above_range(self, mock_consts, mock_logger):
        """Test get_request_time_out with value above maximum (more than 24 hours)"""
        mock_consts.REQUEST_TIME_OUT = 86401  # 1 second over 24 hours
        result = get_request_time_out()
        
        self.assertIsNone(result)  # Returns default
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("out of range", warning_msg)
        self.assertIn("86401", warning_msg)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.logger")
    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_request_time_out_extremely_large(self, mock_consts, mock_logger):
        """Test get_request_time_out with extremely large value"""
        mock_consts.REQUEST_TIME_OUT = 1000000  # Way over 24 hours
        result = get_request_time_out()
        
        self.assertIsNone(result)  # Returns default
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("1000000", warning_msg)

    @patch("ais_bench.benchmark.utils.core.valid_global_consts.logger")
    @patch("ais_bench.benchmark.utils.core.valid_global_consts.global_consts")
    def test_get_request_time_out_negative_float(self, mock_consts, mock_logger):
        """Test get_request_time_out with negative float"""
        mock_consts.REQUEST_TIME_OUT = -5.5
        result = get_request_time_out()
        
        self.assertIsNone(result)  # Returns default
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        self.assertIn("-5.5", warning_msg)


if __name__ == "__main__":
    unittest.main()
