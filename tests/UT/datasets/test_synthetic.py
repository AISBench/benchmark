"""Unit tests for synthetic.py"""
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock

from ais_bench.benchmark.datasets.synthetic import (
    SyntheticDataset,
    _check_keys_equal,
    _ensure_keys_present,
    check_type,
    check_range,
    normalize_file_path,
    NumberRange,
)


class TestHelperFunctions(unittest.TestCase):
    """测试辅助函数"""
    
    def test_check_keys_equal_valid(self):
        """测试_check_keys_equal函数 - 有效情况"""
        got_keys = {'a', 'b', 'c'}
        true_keys = {'a', 'b', 'c'}
        # 应该不抛出异常
        _check_keys_equal(got_keys, true_keys, "test")
    
    def test_check_keys_equal_missing_key(self):
        """测试_check_keys_equal函数 - 缺少键"""
        got_keys = {'a', 'b'}
        true_keys = {'a', 'b', 'c'}
        # 当got_keys != true_keys时，会抛出"Expect keys"异常
        with self.assertRaises(ValueError) as cm:
            _check_keys_equal(got_keys, true_keys, "test")
        self.assertIn("Expect keys", str(cm.exception))
    
    def test_check_keys_equal_extra_key(self):
        """测试_check_keys_equal函数 - 多余键"""
        got_keys = {'a', 'b', 'c', 'd'}
        true_keys = {'a', 'b', 'c'}
        with self.assertRaises(ValueError) as cm:
            _check_keys_equal(got_keys, true_keys, "test")
        self.assertIn("not a valid key", str(cm.exception))
    
    def test_check_keys_equal_different_keys(self):
        """测试_check_keys_equal函数 - 不同的键"""
        got_keys = {'a', 'b'}
        true_keys = {'c', 'd'}
        # 这种情况下会先检查got_keys中的键是否在true_keys中，如果不在会先抛出异常
        with self.assertRaises(ValueError) as cm:
            _check_keys_equal(got_keys, true_keys, "test")
        # 应该包含"not a valid key"或"Expect keys"
        self.assertTrue("not a valid key" in str(cm.exception) or "Expect keys" in str(cm.exception))
    
    def test_ensure_keys_present_valid(self):
        """测试_ensure_keys_present函数 - 有效情况"""
        check_keys = {'a', 'b', 'c'}
        required_keys = {'a', 'b'}
        # 应该不抛出异常
        _ensure_keys_present(check_keys, required_keys, "test")
    
    def test_ensure_keys_present_missing(self):
        """测试_ensure_keys_present函数 - 缺少必需键"""
        check_keys = {'a'}
        required_keys = {'a', 'b'}
        with self.assertRaises(ValueError) as cm:
            _ensure_keys_present(check_keys, required_keys, "test")
        self.assertIn("Missing required key", str(cm.exception))
    
    def test_check_type_valid(self):
        """测试check_type函数 - 有效情况"""
        # 应该不抛出异常
        check_type("test_param", 123, types=(int,))
        check_type("test_param", 123.0, types=(int, float))
    
    def test_check_type_invalid(self):
        """测试check_type函数 - 无效类型"""
        with self.assertRaises(ValueError) as cm:
            check_type("test_param", "123", types=(int,))
        self.assertIn("should have type", str(cm.exception))
    
    def test_check_range_valid(self):
        """测试check_range函数 - 有效情况"""
        param = NumberRange(lower=1, upper=10, lower_inclusive=True, upper_inclusive=True)
        # 应该不抛出异常
        check_range("test_param", 5, param)
    
    def test_check_range_lower_bound_inclusive(self):
        """测试check_range函数 - 下限（包含）"""
        param = NumberRange(lower=1, upper=10, lower_inclusive=True, upper_inclusive=True)
        # 边界值应该通过
        check_range("test_param", 1, param)
    
    def test_check_range_lower_bound_exclusive(self):
        """测试check_range函数 - 下限（不包含）"""
        param = NumberRange(lower=1, upper=10, lower_inclusive=False, upper_inclusive=True)
        # 边界值应该失败
        with self.assertRaises(ValueError):
            check_range("test_param", 1, param)
        # 大于边界值应该通过
        check_range("test_param", 2, param)
    
    def test_check_range_upper_bound_inclusive(self):
        """测试check_range函数 - 上限（包含）"""
        param = NumberRange(lower=1, upper=10, lower_inclusive=True, upper_inclusive=True)
        # 边界值应该通过
        check_range("test_param", 10, param)
    
    def test_check_range_upper_bound_exclusive(self):
        """测试check_range函数 - 上限（不包含）"""
        param = NumberRange(lower=1, upper=10, lower_inclusive=True, upper_inclusive=False)
        # 边界值应该失败
        with self.assertRaises(ValueError):
            check_range("test_param", 10, param)
        # 小于边界值应该通过
        check_range("test_param", 9, param)
    
    def test_check_range_no_lower(self):
        """测试check_range函数 - 无下限"""
        param = NumberRange(lower=None, upper=10, upper_inclusive=True)
        check_range("test_param", -100, param)
    
    def test_check_range_no_upper(self):
        """测试check_range函数 - 无上限"""
        param = NumberRange(lower=1, upper=None, lower_inclusive=True)
        check_range("test_param", 1000, param)
    
    def test_normalize_file_path(self):
        """测试normalize_file_path函数"""
        # 测试相对路径
        result = normalize_file_path("./test")
        self.assertTrue(os.path.isabs(result))
        
        # 测试绝对路径
        abs_path = "/tmp/test"
        result = normalize_file_path(abs_path)
        self.assertEqual(result, abs_path)
        
        # 测试~路径
        result = normalize_file_path("~/test")
        self.assertTrue(os.path.isabs(result))


class TestSyntheticDataset(unittest.TestCase):
    """测试SyntheticDataset类"""
    
    def test_check_synthetic_string_config_uniform(self):
        """测试check_synthetic_string_config - uniform方法"""
        config = {
            "Input": {
                "Method": "uniform",
                "Params": {
                    "MinValue": 10,
                    "MaxValue": 100
                }
            },
            "Output": {
                "Method": "uniform",
                "Params": {
                    "MinValue": 5,
                    "MaxValue": 50
                }
            }
        }
        # 应该不抛出异常
        SyntheticDataset.check_synthetic_string_config(config)
    
    def test_check_synthetic_string_config_gaussian(self):
        """测试check_synthetic_string_config - gaussian方法"""
        config = {
            "Input": {
                "Method": "gaussian",
                "Params": {
                    "Mean": 50.0,
                    "Var": 10.0,
                    "MinValue": 10,
                    "MaxValue": 100
                }
            },
            "Output": {
                "Method": "gaussian",
                "Params": {
                    "Mean": 25.0,
                    "Var": 5.0,
                    "MinValue": 5,
                    "MaxValue": 50
                }
            }
        }
        # 应该不抛出异常
        SyntheticDataset.check_synthetic_string_config(config)
    
    def test_check_synthetic_string_config_zipf(self):
        """测试check_synthetic_string_config - zipf方法"""
        config = {
            "Input": {
                "Method": "zipf",
                "Params": {
                    "Alpha": 2.0,
                    "MinValue": 10,
                    "MaxValue": 100
                }
            },
            "Output": {
                "Method": "zipf",
                "Params": {
                    "Alpha": 1.5,
                    "MinValue": 5,
                    "MaxValue": 50
                }
            }
        }
        # 应该不抛出异常
        SyntheticDataset.check_synthetic_string_config(config)
    
    def test_check_synthetic_string_config_missing_keys(self):
        """测试check_synthetic_string_config - 缺少必需键"""
        config = {
            "Input": {
                "Method": "uniform",
                "Params": {
                    "MinValue": 10,
                    "MaxValue": 100
                }
            }
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset.check_synthetic_string_config(config)
        self.assertIn("Missing required key", str(cm.exception))
    
    def test_check_synthetic_string_config_invalid_method(self):
        """测试check_synthetic_string_config - 无效方法"""
        config = {
            "Input": {
                "Method": "invalid",
                "Params": {
                    "MinValue": 10,
                    "MaxValue": 100
                }
            },
            "Output": {
                "Method": "uniform",
                "Params": {
                    "MinValue": 5,
                    "MaxValue": 50
                }
            }
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset.check_synthetic_string_config(config)
        self.assertIn("Method should be one of", str(cm.exception))
    
    def test_check_synthetic_string_config_min_greater_than_max(self):
        """测试check_synthetic_string_config - MinValue大于MaxValue"""
        config = {
            "Input": {
                "Method": "uniform",
                "Params": {
                    "MinValue": 100,
                    "MaxValue": 10
                }
            },
            "Output": {
                "Method": "uniform",
                "Params": {
                    "MinValue": 5,
                    "MaxValue": 50
                }
            }
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset.check_synthetic_string_config(config)
        self.assertIn("MinValue should less than MaxValue", str(cm.exception))
    
    def test_check_synthetic_string_config_alpha_range(self):
        """测试check_synthetic_string_config - Alpha范围检查"""
        config = {
            "Input": {
                "Method": "zipf",
                "Params": {
                    "Alpha": 0.5,  # 小于1.0，应该失败
                    "MinValue": 10,
                    "MaxValue": 100
                }
            },
            "Output": {
                "Method": "zipf",
                "Params": {
                    "Alpha": 1.5,
                    "MinValue": 5,
                    "MaxValue": 50
                }
            }
        }
        with self.assertRaises(ValueError):
            SyntheticDataset.check_synthetic_string_config(config)
    
    def test_check_synthetic_tokenid_config(self):
        """测试check_synthetic_tokenid_config"""
        config = {
            "RequestSize": 1000
        }
        # 应该不抛出异常
        SyntheticDataset.check_synthetic_tokenid_config(config)
    
    def test_check_synthetic_tokenid_config_missing_key(self):
        """测试check_synthetic_tokenid_config - 缺少键"""
        config = {}
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset.check_synthetic_tokenid_config(config)
        self.assertIn("Missing required key", str(cm.exception))
    
    def test_check_synthetic_tokenid_config_invalid_type(self):
        """测试check_synthetic_tokenid_config - 无效类型"""
        config = {
            "RequestSize": "1000"  # 应该是int
        }
        with self.assertRaises(ValueError):
            SyntheticDataset.check_synthetic_tokenid_config(config)
    
    def test_check_synthetic_config_string_type(self):
        """测试_check_synthetic_config - string类型"""
        config = {
            "Type": "string",
            "RequestCount": 100,
            "StringConfig": {
                "Input": {
                    "Method": "uniform",
                    "Params": {
                        "MinValue": 10,
                        "MaxValue": 100
                    }
                },
                "Output": {
                    "Method": "uniform",
                    "Params": {
                        "MinValue": 5,
                        "MaxValue": 50
                    }
                }
            }
        }
        # 应该不抛出异常
        SyntheticDataset._check_synthetic_config(config)
    
    def test_check_synthetic_config_tokenid_type(self):
        """测试_check_synthetic_config - tokenid类型"""
        config = {
            "Type": "tokenid",
            "RequestCount": 100,
            "TokenIdConfig": {
                "RequestSize": 1000
            }
        }
        # 应该不抛出异常
        SyntheticDataset._check_synthetic_config(config)
    
    def test_check_synthetic_config_case_insensitive(self):
        """测试_check_synthetic_config - 大小写不敏感"""
        config = {
            "Type": "STRING",  # 大写
            "RequestCount": 100,
            "StringConfig": {
                "Input": {
                    "Method": "uniform",
                    "Params": {
                        "MinValue": 10,
                        "MaxValue": 100
                    }
                },
                "Output": {
                    "Method": "uniform",
                    "Params": {
                        "MinValue": 5,
                        "MaxValue": 50
                    }
                }
            }
        }
        # 应该不抛出异常
        SyntheticDataset._check_synthetic_config(config)
    
    def test_check_synthetic_config_invalid_type(self):
        """测试_check_synthetic_config - 无效类型"""
        config = {
            "Type": "invalid",
            "RequestCount": 100
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset._check_synthetic_config(config)
        self.assertIn("Expect type should from", str(cm.exception))
    
    def test_sample_one_value_uniform(self):
        """测试sample_one_value - uniform方法"""
        method = "uniform"
        params = {"MinValue": 10, "MaxValue": 100}
        value = SyntheticDataset.sample_one_value(method, params)
        self.assertIsInstance(value, int)
        self.assertGreaterEqual(value, 10)
        self.assertLessEqual(value, 100)
    
    def test_sample_one_value_gaussian(self):
        """测试sample_one_value - gaussian方法"""
        method = "gaussian"
        params = {"Mean": 50.0, "Var": 10.0, "MinValue": 10, "MaxValue": 100}
        value = SyntheticDataset.sample_one_value(method, params)
        self.assertIsInstance(value, int)
        self.assertGreaterEqual(value, 10)
        self.assertLessEqual(value, 100)
    
    def test_sample_one_value_zipf(self):
        """测试sample_one_value - zipf方法"""
        method = "zipf"
        params = {"Alpha": 2.0, "MinValue": 10, "MaxValue": 100}
        value = SyntheticDataset.sample_one_value(method, params)
        self.assertIsInstance(value, int)
        self.assertGreaterEqual(value, 10)
        self.assertLessEqual(value, 100)
    
    def test_sample_one_value_invalid_method(self):
        """测试sample_one_value - 无效方法"""
        method = "invalid"
        params = {"MinValue": 10, "MaxValue": 100}
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset.sample_one_value(method, params)
        self.assertIn("Unknown method", str(cm.exception))
    
    def test_read_line_valid(self):
        """测试read_line - 有效情况"""
        line = [10, 5]
        data, num_tokens = SyntheticDataset.read_line(None, line)
        self.assertIsInstance(data, str)
        self.assertEqual(num_tokens, 5)
        # 验证数据包含10个"A"
        self.assertEqual(len(data.split()), 10)
    
    def test_read_line_invalid_length(self):
        """测试read_line - 无效长度"""
        line = [10]  # 只有1个元素
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset.read_line(None, line)
        self.assertIn("should be a list with 2 integral elements", str(cm.exception))
    
    def test_read_line_not_list(self):
        """测试read_line - 不是列表"""
        line = "not a list"
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset.read_line(None, line)
        self.assertIn("should be a list with 2 integral elements", str(cm.exception))
    
    def test_find_first_file_path_valid(self):
        """测试find_first_file_path - 有效情况"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试文件
            test_file = os.path.join(tmpdir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            
            # 查找文件
            result = SyntheticDataset.find_first_file_path(tmpdir, "test.txt")
            self.assertEqual(result, test_file)
    
    def test_find_first_file_path_not_found(self):
        """测试find_first_file_path - 文件不存在"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                SyntheticDataset.find_first_file_path(tmpdir, "nonexistent.txt")
    
    def test_find_first_file_path_invalid_path(self):
        """测试find_first_file_path - 无效路径"""
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset.find_first_file_path("/nonexistent/path", "test.txt")
        self.assertIn("Path does not exist", str(cm.exception))
    
    def test_find_first_file_path_not_directory(self):
        """测试find_first_file_path - 不是目录"""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(b"test")
            tmpfile_path = tmpfile.name
        
        try:
            with self.assertRaises(ValueError) as cm:
                SyntheticDataset.find_first_file_path(tmpfile_path, "test.txt")
            self.assertIn("Not a directory", str(cm.exception))
        finally:
            os.unlink(tmpfile_path)
    
    def test_generate_valid_random_ids(self):
        """测试generate_valid_random_ids函数"""
        import torch
        # 创建有效的索引
        valid_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        request_size = 5
        
        result = SyntheticDataset.generate_valid_random_ids(valid_indices, request_size)
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(len(result), request_size)
        self.assertEqual(result.dtype, torch.int64)
        # 验证所有值都在valid_indices中
        for val in result:
            self.assertIn(val.item(), valid_indices.tolist())
    
    def test_load_string_type(self):
        """测试load方法 - string类型（覆盖265-286行）"""
        from datasets import Dataset
        from ais_bench.benchmark.utils.logging import get_logger
        
        config = {
            "Type": "string",
            "RequestCount": 5,
            "StringConfig": {
                "Input": {
                    "Method": "uniform",
                    "Params": {
                        "MinValue": 10,
                        "MaxValue": 100
                    }
                },
                "Output": {
                    "Method": "uniform",
                    "Params": {
                        "MinValue": 5,
                        "MaxValue": 50
                    }
                }
            }
        }
        
        # 创建一个实例，但绕过BaseDataset的__init__，直接设置logger
        dataset_instance = object.__new__(SyntheticDataset)
        dataset_instance.logger = get_logger()
        
        dataset = dataset_instance.load(config)
        
        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(len(dataset), 5)
        # 验证每个样本都有必需的字段
        for item in dataset:
            self.assertIn("question", item)
            self.assertIn("answer", item)
            self.assertIn("max_out_len", item)
    
    @patch('ais_bench.benchmark.datasets.synthetic.AutoTokenizer')
    @patch('ais_bench.benchmark.datasets.synthetic.SyntheticDataset.find_first_file_path')
    def test_load_tokenid_type(self, mock_find_file, mock_tokenizer):
        """测试load方法 - tokenid类型（覆盖287-326行）"""
        from datasets import Dataset
        import tempfile
        from ais_bench.benchmark.utils.logging import get_logger
        
        # 创建临时目录和tokenizer配置文件
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_config_path = os.path.join(tmpdir, "tokenizer_config.json")
            with open(tokenizer_config_path, 'w') as f:
                f.write('{"vocab_size": 1000}')
            
            # Mock tokenizer
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.vocab_size = 1000
            mock_tokenizer_instance.all_special_ids = [0, 1, 2]
            mock_tokenizer_instance.decode.return_value = "decoded text"
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock find_first_file_path
            mock_find_file.return_value = tokenizer_config_path
            
            config = {
                "Type": "tokenid",
                "RequestCount": 3,
                "TokenIdConfig": {
                    "RequestSize": 10,
                    "VocabSize": 1000
                },
                "ModelPath": tmpdir,
                "TrustRemoteCode": False
            }
            
            # 创建一个实例，但绕过BaseDataset的__init__，直接设置logger
            dataset_instance = object.__new__(SyntheticDataset)
            dataset_instance.logger = get_logger()
            
            dataset = dataset_instance.load(config, model_path=tmpdir)
            
            self.assertIsInstance(dataset, Dataset)
            self.assertEqual(len(dataset), 3)
            # 验证每个样本都有必需的字段
            for item in dataset:
                self.assertIn("question", item)
                self.assertIn("answer", item)
    
    def test_load_invalid_type(self):
        """测试load方法 - 无效类型（覆盖328-329行）"""
        from ais_bench.benchmark.utils.logging import get_logger
        
        # 创建一个mock来绕过_check_synthetic_config，直接测试else分支
        with patch.object(SyntheticDataset, '_check_synthetic_config'):
            dataset_instance = object.__new__(SyntheticDataset)
            dataset_instance.logger = get_logger()
            invalid_config = {
                "Type": "invalid",
                "RequestCount": 100
            }
            with self.assertRaises(ValueError) as cm:
                dataset_instance.load(invalid_config)
            # 错误信息来自load方法的else分支
            error_msg = str(cm.exception)
            self.assertIn("Invalid type", error_msg)
    
    def test_check_synthetic_string_config_missing_method_key(self):
        """测试check_synthetic_string_config - 缺少Method键（覆盖78-100行）"""
        config = {
            "Input": {
                "Params": {  # 缺少Method键
                    "MinValue": 10,
                    "MaxValue": 100
                }
            },
            "Output": {
                "Method": "uniform",
                "Params": {
                    "MinValue": 5,
                    "MaxValue": 50
                }
            }
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset.check_synthetic_string_config(config)
        self.assertIn("Expect keys", str(cm.exception))
    
    def test_check_synthetic_string_config_missing_params_key(self):
        """测试check_synthetic_string_config - 缺少Params键（覆盖78-100行）"""
        config = {
            "Input": {
                "Method": "uniform"
                # 缺少Params键
            },
            "Output": {
                "Method": "uniform",
                "Params": {
                    "MinValue": 5,
                    "MaxValue": 50
                }
            }
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset.check_synthetic_string_config(config)
        self.assertIn("Expect keys", str(cm.exception))
    
    def test_check_synthetic_string_config_invalid_param_type(self):
        """测试check_synthetic_string_config - 无效参数类型（覆盖103-119行）"""
        config = {
            "Input": {
                "Method": "uniform",
                "Params": {
                    "MinValue": "10",  # 应该是int
                    "MaxValue": 100
                }
            },
            "Output": {
                "Method": "uniform",
                "Params": {
                    "MinValue": 5,
                    "MaxValue": 50
                }
            }
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset.check_synthetic_string_config(config)
        self.assertIn("should have type", str(cm.exception))
    
    def test_check_synthetic_string_config_param_out_of_range(self):
        """测试check_synthetic_string_config - 参数超出范围（覆盖103-119行）"""
        config = {
            "Input": {
                "Method": "uniform",
                "Params": {
                    "MinValue": 0,  # 小于1，应该失败
                    "MaxValue": 100
                }
            },
            "Output": {
                "Method": "uniform",
                "Params": {
                    "MinValue": 5,
                    "MaxValue": 50
                }
            }
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset.check_synthetic_string_config(config)
        self.assertIn("not within the required range", str(cm.exception))
    
    def test_check_synthetic_string_config_gaussian_params(self):
        """测试check_synthetic_string_config - gaussian方法的参数检查（覆盖111-116行）"""
        config = {
            "Input": {
                "Method": "gaussian",
                "Params": {
                    "Mean": 50.0,
                    "Var": -10.0,  # 负数，应该失败
                    "MinValue": 10,
                    "MaxValue": 100
                }
            },
            "Output": {
                "Method": "gaussian",
                "Params": {
                    "Mean": 25.0,
                    "Var": 5.0,
                    "MinValue": 5,
                    "MaxValue": 50
                }
            }
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset.check_synthetic_string_config(config)
        self.assertIn("not within the required range", str(cm.exception))
    
    def test_check_synthetic_string_config_alpha_at_boundary(self):
        """测试check_synthetic_string_config - Alpha边界值（覆盖117-119行）"""
        # Alpha = 1.0 应该失败（lower_inclusive=False）
        config = {
            "Input": {
                "Method": "zipf",
                "Params": {
                    "Alpha": 1.0,  # 等于下界，但不包含
                    "MinValue": 10,
                    "MaxValue": 100
                }
            },
            "Output": {
                "Method": "zipf",
                "Params": {
                    "Alpha": 1.5,
                    "MinValue": 5,
                    "MaxValue": 50
                }
            }
        }
        with self.assertRaises(ValueError):
            SyntheticDataset.check_synthetic_string_config(config)
        
        # Alpha = 10.0 应该失败（upper_inclusive=True，但可能超过）
        config2 = {
            "Input": {
                "Method": "zipf",
                "Params": {
                    "Alpha": 10.1,  # 超过上界
                    "MinValue": 10,
                    "MaxValue": 100
                }
            },
            "Output": {
                "Method": "zipf",
                "Params": {
                    "Alpha": 1.5,
                    "MinValue": 5,
                    "MaxValue": 50
                }
            }
        }
        with self.assertRaises(ValueError):
            SyntheticDataset.check_synthetic_string_config(config2)
    
    def test_check_synthetic_config_missing_type_key(self):
        """测试_check_synthetic_config - 缺少Type键（覆盖136-149行）"""
        config = {
            "RequestCount": 100
            # 缺少Type键
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset._check_synthetic_config(config)
        self.assertIn("Missing required key", str(cm.exception))
    
    def test_check_synthetic_config_missing_request_count_key(self):
        """测试_check_synthetic_config - 缺少RequestCount键（覆盖136-149行）"""
        config = {
            "Type": "string"
            # 缺少RequestCount键
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset._check_synthetic_config(config)
        self.assertIn("Missing required key", str(cm.exception))
    
    def test_check_synthetic_config_invalid_type_value(self):
        """测试_check_synthetic_config - 无效Type值类型（覆盖143-145行）"""
        config = {
            "Type": 123,  # 应该是str
            "RequestCount": 100
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset._check_synthetic_config(config)
        self.assertIn("should have type", str(cm.exception))
    
    def test_check_synthetic_config_invalid_request_count_type(self):
        """测试_check_synthetic_config - 无效RequestCount类型（覆盖147-149行）"""
        config = {
            "Type": "string",
            "RequestCount": "100"  # 应该是int
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset._check_synthetic_config(config)
        self.assertIn("should have type", str(cm.exception))
    
    def test_check_synthetic_config_request_count_out_of_range(self):
        """测试_check_synthetic_config - RequestCount超出范围（覆盖147-149行）"""
        config = {
            "Type": "string",
            "RequestCount": 0  # 小于1，应该失败
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset._check_synthetic_config(config)
        self.assertIn("not within the required range", str(cm.exception))
    
    def test_check_synthetic_config_missing_string_config(self):
        """测试_check_synthetic_config - 缺少StringConfig键（覆盖161-166行）"""
        config = {
            "Type": "string",
            "RequestCount": 100
            # 缺少StringConfig键
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset._check_synthetic_config(config)
        self.assertIn("Missing required key", str(cm.exception))
    
    def test_check_synthetic_config_missing_tokenid_config(self):
        """测试_check_synthetic_config - 缺少TokenIdConfig键（覆盖161-166行）"""
        config = {
            "Type": "tokenid",
            "RequestCount": 100
            # 缺少TokenIdConfig键
        }
        with self.assertRaises(ValueError) as cm:
            SyntheticDataset._check_synthetic_config(config)
        self.assertIn("Missing required key", str(cm.exception))
    
    def test_find_first_file_path_subdirectory(self):
        """测试find_first_file_path - 在子目录中查找（覆盖236-241行）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建子目录和文件
            subdir = os.path.join(tmpdir, "subdir")
            os.makedirs(subdir)
            test_file = os.path.join(subdir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            
            # 查找文件
            result = SyntheticDataset.find_first_file_path(tmpdir, "test.txt")
            self.assertEqual(result, test_file)
    
    def test_load_tokenid_type_model_path_not_exist(self):
        """测试load方法 - tokenid类型，ModelPath不存在（覆盖292-294行）"""
        from ais_bench.benchmark.utils.logging import get_logger
        
        config = {
            "Type": "tokenid",
            "RequestCount": 3,
            "TokenIdConfig": {
                "RequestSize": 10
            }
            # ModelPath会在load方法中从kwargs获取
        }
        
        dataset_instance = object.__new__(SyntheticDataset)
        dataset_instance.logger = get_logger()
        
        # 需要绕过_check_synthetic_config，并传入不存在的model_path
        with patch.object(SyntheticDataset, '_check_synthetic_config'):
            with self.assertRaises(ValueError) as cm:
                dataset_instance.load(config, model_path="/nonexistent/path")
            self.assertIn("ModelPath does not exist", str(cm.exception))
    
    @patch('ais_bench.benchmark.datasets.synthetic.AutoTokenizer')
    @patch('ais_bench.benchmark.datasets.synthetic.SyntheticDataset.find_first_file_path')
    def test_load_tokenid_type_no_vocab_size(self, mock_find_file, mock_tokenizer):
        """测试load方法 - tokenid类型，没有vocab_size（覆盖304-308行）"""
        from ais_bench.benchmark.utils.logging import get_logger
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_config_path = os.path.join(tmpdir, "tokenizer_config.json")
            with open(tokenizer_config_path, 'w') as f:
                f.write('{}')
            
            # Mock tokenizer - 没有vocab_size
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.vocab_size = None
            mock_tokenizer_instance.all_special_ids = [0, 1, 2]
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            mock_find_file.return_value = tokenizer_config_path
            
            config = {
                "Type": "tokenid",
                "RequestCount": 3,
                "TokenIdConfig": {
                    "RequestSize": 10
                    # 没有VocabSize
                },
                "ModelPath": tmpdir
            }
            
            dataset_instance = object.__new__(SyntheticDataset)
            dataset_instance.logger = get_logger()
            
            # 需要传入model_path参数，因为load方法会从kwargs获取
            with self.assertRaises(ValueError) as cm:
                dataset_instance.load(config, model_path=tmpdir)
            self.assertIn("vocab_size was not found", str(cm.exception))
    
    @patch('ais_bench.benchmark.datasets.synthetic.AutoTokenizer')
    @patch('ais_bench.benchmark.datasets.synthetic.SyntheticDataset.find_first_file_path')
    def test_load_tokenid_type_with_vocab_size_in_config(self, mock_find_file, mock_tokenizer):
        """测试load方法 - tokenid类型，使用配置中的vocab_size（覆盖305行）"""
        from datasets import Dataset
        from ais_bench.benchmark.utils.logging import get_logger
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_config_path = os.path.join(tmpdir, "tokenizer_config.json")
            with open(tokenizer_config_path, 'w') as f:
                f.write('{}')
            
            # Mock tokenizer - 没有vocab_size，但配置中有
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.vocab_size = None
            mock_tokenizer_instance.all_special_ids = [0, 1, 2]
            mock_tokenizer_instance.decode.return_value = "decoded text"
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            mock_find_file.return_value = tokenizer_config_path
            
            config = {
                "Type": "tokenid",
                "RequestCount": 2,
                "TokenIdConfig": {
                    "RequestSize": 5,
                    "VocabSize": 1000  # 配置中有vocab_size
                },
                "ModelPath": tmpdir
            }
            
            dataset_instance = object.__new__(SyntheticDataset)
            dataset_instance.logger = get_logger()
            
            # 需要传入model_path参数
            dataset = dataset_instance.load(config, model_path=tmpdir)
            
            self.assertIsInstance(dataset, Dataset)
            self.assertEqual(len(dataset), 2)


if __name__ == "__main__":
    unittest.main()

