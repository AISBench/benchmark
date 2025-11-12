"""Unit tests for humanevalx/humanevalx.py"""
import unittest
import tempfile
import gzip
import json
import os
from unittest.mock import patch, mock_open, MagicMock

from datasets import Dataset

from ais_bench.benchmark.datasets.humanevalx import humanevalx
from ais_bench.benchmark.datasets.humanevalx.humanevalx import (
    HumanevalXDataset,
    generate_predictions_from_file,
    HumanevalXEvaluator,
    _clean_up_code,
    _LANGUAGE_NAME_DICT,
)
from ais_bench.benchmark.datasets.humanevalx import humaneval_x_eval
from ais_bench.benchmark.datasets.utils import datasets as datasets_utils


class TestHumanevalXDataset(unittest.TestCase):
    """测试 HumanevalXDataset"""
    
    def test_load_invalid_language(self):
        """测试load - 无效语言（覆盖35-37行）"""
        with patch.object(datasets_utils, 'get_data_path', return_value='/fake/path'):
            with self.assertRaises(AssertionError) as cm:
                HumanevalXDataset.load('/input', language='invalid')
            self.assertIn('language must be in', str(cm.exception))
    
    def test_load_valid_language(self):
        """测试load - 有效语言（覆盖34-40行）"""
        # 创建模拟的gzip文件内容
        test_data = [{"task_id": "Python/0", "prompt": "def hello():"}]
        gzip_content = '\n'.join(json.dumps(item) for item in test_data)
        
        # 需要mock gzip.open和内置的open
        # gzip.open需要接收一个文件对象，所以先mock内置的open
        # gzip.open返回一个文件对象，需要支持迭代
        mock_gzip_file = MagicMock()
        mock_gzip_file.__iter__.return_value = iter(gzip_content.split('\n'))
        mock_gzip_file.__enter__.return_value = mock_gzip_file
        mock_gzip_file.__exit__.return_value = None
        
        mock_file_obj = MagicMock()
        mock_file_obj.__enter__.return_value = mock_file_obj
        mock_file_obj.__exit__.return_value = None
        
        with patch.object(datasets_utils, 'get_data_path', return_value='/fake/path'):
            with patch('builtins.open', return_value=mock_file_obj):
                with patch('gzip.open', return_value=mock_gzip_file):
                    ds = HumanevalXDataset.load('/input', language='python')
                    self.assertIsInstance(ds, Dataset)
    
    def test_stream_jsonl_all_gz(self):
        """测试_stream_jsonl_all - gzip文件（覆盖42-54行）"""
        test_data = [
            {"task_id": "Python/0", "prompt": "def hello():"},
            {"task_id": "Python/1", "prompt": "def world():"}
        ]
        gzip_content = '\n'.join(json.dumps(item) for item in test_data)
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.gz', delete=False) as tmpfile:
            with gzip.open(tmpfile.name, 'wb') as gz:
                gz.write(gzip_content.encode('utf-8'))
            tmpfile_path = tmpfile.name
        
        try:
            result = HumanevalXDataset._stream_jsonl_all(tmpfile_path)
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]['task_id'], "Python/0")
        finally:
            os.unlink(tmpfile_path)
    
    def test_stream_jsonl_all_regular_file(self):
        """测试_stream_jsonl_all - 普通文件（覆盖42-54行）"""
        test_data = [
            {"task_id": "Python/0", "prompt": "def hello():"}
        ]
        content = '\n'.join(json.dumps(item) for item in test_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmpfile:
            tmpfile.write(content)
            tmpfile_path = tmpfile.name
        
        try:
            result = HumanevalXDataset._stream_jsonl_all(tmpfile_path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['task_id'], "Python/0")
        finally:
            os.unlink(tmpfile_path)
    
    def test_stream_jsonl_all_empty_lines(self):
        """测试_stream_jsonl_all - 空行处理（覆盖49-51行）"""
        test_data = [
            {"task_id": "Python/0", "prompt": "def hello():"}
        ]
        content = '\n'.join(json.dumps(item) for item in test_data) + '\n\n   \n'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmpfile:
            tmpfile.write(content)
            tmpfile_path = tmpfile.name
        
        try:
            result = HumanevalXDataset._stream_jsonl_all(tmpfile_path)
            # 空行应该被跳过
            self.assertEqual(len(result), 1)
        finally:
            os.unlink(tmpfile_path)


class TestGeneratePredictionsFromFile(unittest.TestCase):
    """测试 generate_predictions_from_file 函数"""
    
    @patch('jsonlines.open')
    def test_generate_predictions_from_file(self, mock_jsonlines_open):
        """测试generate_predictions_from_file（覆盖60-72行）"""
        mock_reader = MagicMock()
        mock_reader.__iter__.return_value = [
            {'task_id': 'Python/0', 'generation': 'def hello(): return "world"'},
            {'task_id': 'Python/1', 'generation': 'def world(): return "hello"'}
        ]
        mock_jsonlines_open.return_value.__enter__.return_value = mock_reader
        
        result = generate_predictions_from_file('/fake/path', 'python')
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['task_id'], 'Python/0')
        self.assertEqual(result[0]['generation'], 'def hello(): return "world"')
        self.assertEqual(result[1]['task_id'], 'Python/1')


class TestHumanevalXEvaluator(unittest.TestCase):
    """测试 HumanevalXEvaluator"""
    
    def test_init_invalid_language(self):
        """测试__init__ - 无效语言（覆盖95-103行）"""
        with self.assertRaises(AssertionError) as cm:
            HumanevalXEvaluator(language='invalid')
        self.assertIn('language must be in', str(cm.exception))
    
    def test_init_valid_language(self):
        """测试__init__ - 有效语言（覆盖95-111行）"""
        evaluator = HumanevalXEvaluator(language='python')
        self.assertEqual(evaluator.language, 'python')
        self.assertEqual(evaluator.ip_address, 'localhost')
        self.assertEqual(evaluator.timeout, 6)
    
    def test_init_rust_timeout(self):
        """测试__init__ - rust语言超时时间（覆盖104-105行）"""
        evaluator = HumanevalXEvaluator(language='rust')
        # rust的超时时间应该是默认值的10倍
        self.assertEqual(evaluator.timeout, 60)  # 6 * 10
    
    def test_init_with_custom_params(self):
        """测试__init__ - 自定义参数（覆盖95-111行）"""
        evaluator = HumanevalXEvaluator(
            language='python',
            ip_address='192.168.1.1',
            port='8080',
            retry=3,
            timeout=10
        )
        self.assertEqual(evaluator.ip_address, '192.168.1.1')
        self.assertEqual(evaluator.port, '8080')
        self.assertEqual(evaluator.retry, 3)
        self.assertEqual(evaluator.timeout, 10)
    
    def test_score(self):
        """测试score方法（覆盖113-146行）"""
        # 由于evaluate_functional_correctness需要复杂的依赖（需要读取文件），我们简化测试
        # 直接mock整个evaluate_functional_correctness函数
        evaluator = HumanevalXEvaluator(language='python')
        predictions = ['def hello(): return "world"']
        references = ['Python/0']
        test_set = [{'prompt': 'def hello():\n    """Return hello world"""'}]
        
        # 使用patch.object直接patch已导入的模块对象
        # 需要patch humanevalx模块中的evaluate_functional_correctness和_clean_up_code
        # 同时需要mock os.path.abspath和os.path.join来避免文件路径问题
        with patch.object(humanevalx, 'evaluate_functional_correctness', return_value={'pass@1': 0.5, 'pass@10': 0.8}) as mock_eval, \
             patch.object(humanevalx, '_clean_up_code', return_value='def hello(): return "world"') as mock_clean_up, \
             patch('os.path.abspath', side_effect=lambda x: '/fake/path' if 'humanevalx' in str(x) else x), \
             patch('os.path.join', side_effect=lambda *args: '/'.join(str(a) for a in args)):
            # 使用真实的tempfile.TemporaryDirectory
            result = evaluator.score(predictions, references, test_set)
            
            self.assertIsNotNone(result)
            self.assertEqual(result, {'pass@1': 0.5, 'pass@10': 0.8})
            mock_eval.assert_called_once()
    
    @patch('subprocess.run')
    def test_code_eval_service_with_port(self, mock_subprocess):
        """测试_code_eval_service - 带端口（覆盖149-175行）"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        # 注意：需要匹配正则表达式 "\"{.*:.*}\""，所以需要双引号包裹的JSON字符串
        # 正则表达式匹配的是被双引号包裹的JSON字符串，但json.loads会解析内层的JSON
        # 所以需要返回被双引号包裹的JSON字符串
        mock_result.stdout = b'"{\\"result\\": \\"passed\\"}"'
        mock_result.stderr = b''
        mock_subprocess.return_value = mock_result
        
        evaluator = HumanevalXEvaluator(language='python', ip_address='localhost', port='5000')
        
        success, result = evaluator._code_eval_service('/fake/path')
        
        self.assertTrue(success)
        # 注意：json.loads会解析被双引号包裹的JSON字符串，返回的是内层的JSON字符串
        # 所以result可能是字符串而不是字典
        self.assertIsInstance(result, (dict, str))
        mock_subprocess.assert_called_once()
        # 验证URL包含端口
        call_args = mock_subprocess.call_args[0][0]
        self.assertIn('localhost:5000/evaluate', ' '.join(call_args))
    
    @patch('subprocess.run')
    def test_code_eval_service_without_port(self, mock_subprocess):
        """测试_code_eval_service - 不带端口（覆盖149-175行）"""
        mock_result = MagicMock()
        mock_result.returncode = 0
        # 注意：需要匹配正则表达式 "\"{.*:.*}\""，所以需要双引号包裹的JSON字符串
        mock_result.stdout = b'"{\\"result\\": \\"passed\\"}"'
        mock_result.stderr = b''
        mock_subprocess.return_value = mock_result
        
        evaluator = HumanevalXEvaluator(language='python', ip_address='localhost', port='')
        
        success, result = evaluator._code_eval_service('/fake/path')
        
        self.assertTrue(success)
        # 验证URL不包含端口
        call_args = mock_subprocess.call_args[0][0]
        self.assertIn('localhost/evaluate', ' '.join(call_args))
        self.assertNotIn('localhost:', ' '.join(call_args))
    
    @patch('subprocess.run')
    def test_code_eval_service_failure(self, mock_subprocess):
        """测试_code_eval_service - 失败情况（覆盖161-175行）"""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = b'invalid output'
        mock_result.stderr = b'error message'
        mock_subprocess.return_value = mock_result
        
        evaluator = HumanevalXEvaluator(language='python')
        
        success, error = evaluator._code_eval_service('/fake/path')
        
        self.assertFalse(success)
        self.assertIsInstance(error, str)


class TestCleanUpCode(unittest.TestCase):
    """测试 _clean_up_code 函数"""
    
    def test_clean_up_code_python_basic(self):
        """测试_clean_up_code - Python基本清理（覆盖178-253行）"""
        text = "```python\ndef hello():\n    return 'world'\n```"
        result = _clean_up_code(text, 'python', 'def hello():')
        # 验证函数能够执行并返回字符串
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_clean_up_code_python_with_eval_string(self):
        """测试_clean_up_code - Python eval字符串（覆盖180-187行）"""
        # 测试eval可以解析的字符串
        import json
        text = json.dumps("def hello():\n    return 'world'")
        result = _clean_up_code(text, 'python', 'def hello():')
        # 如果eval成功，应该包含函数定义
        self.assertIsInstance(result, str)
    
    def test_clean_up_code_python_empty_line_detection(self):
        """测试_clean_up_code - Python空行检测（覆盖204-213行）"""
        text = "    def hello():\n    return 'world'\ndef other():\n    pass"
        result = _clean_up_code(text, 'python', 'def hello():')
        # 验证函数能够执行
        self.assertIsInstance(result, str)
    
    def test_clean_up_code_python_end_words(self):
        """测试_clean_up_code - Python结束词（覆盖214-221行）"""
        text = "    def hello():\n    return 'world'\n\ndef other():\n    pass"
        result = _clean_up_code(text, 'python', 'def hello():')
        # 验证函数能够执行
        self.assertIsInstance(result, str)
    
    def test_clean_up_code_java(self):
        """测试_clean_up_code - Java（覆盖228-235行）"""
        text = "public class Test {\n    public static void main(String[] args) {\n        System.out.println('test');\n    }\n}"
        result = _clean_up_code(text, 'java', 'public class Test {')
        # 验证函数能够执行
        self.assertIsInstance(result, str)
    
    def test_clean_up_code_go(self):
        """测试_clean_up_code - Go（覆盖236-240行）"""
        text = "func hello() {\n    return 'world'\n}\nfunc main() {\n    fmt.Println('test')\n}"
        result = _clean_up_code(text, 'go', 'func hello() {')
        # 验证函数能够执行
        self.assertIsInstance(result, str)
    
    def test_clean_up_code_cpp(self):
        """测试_clean_up_code - C++（覆盖241-245行）"""
        text = "int hello() {\n    return 1;\n}\nint main() {\n    return 0;\n}"
        result = _clean_up_code(text, 'cpp', 'int hello() {')
        # 验证函数能够执行
        self.assertIsInstance(result, str)
    
    def test_clean_up_code_js(self):
        """测试_clean_up_code - JavaScript（覆盖246-248行）"""
        text = "function hello() {\n    return 'world';\n}\nfunction other() {\n    return 'test';\n}"
        result = _clean_up_code(text, 'js', 'function hello() {')
        # 验证函数能够执行
        self.assertIsInstance(result, str)
    
    def test_clean_up_code_rust(self):
        """测试_clean_up_code - Rust（覆盖249-251行）"""
        text = "fn hello() -> String {\n    \"world\".to_string()\n}\nfn other() {\n    println!(\"test\");\n}"
        result = _clean_up_code(text, 'rust', 'fn hello() -> String {')
        # 验证函数能够执行
        self.assertIsInstance(result, str)
    
    def test_clean_up_code_with_reference(self):
        """测试_clean_up_code - 带reference参数（覆盖222-227行）"""
        text = "def hello():\n    return 'world'\ndef other():\n    return 'test'"
        reference = "def hello():\n    pass"
        result = _clean_up_code(text, 'python', reference)
        # 验证函数能够执行
        self.assertIsInstance(result, str)
    
    def test_clean_up_code_no_code_block(self):
        """测试_clean_up_code - 无代码块（覆盖189-197行）"""
        text = "def hello():\n    return 'world'"
        result = _clean_up_code(text, 'python', 'def hello():')
        # 验证函数能够执行
        self.assertIsInstance(result, str)
    
    def test_clean_up_code_code_block_fallback(self):
        """测试_clean_up_code - 代码块回退策略（覆盖190-197行）"""
        text = "```\ndef hello():\n    return 'world'\n```"
        result = _clean_up_code(text, 'python', 'def hello():')
        # 验证函数能够执行
        self.assertIsInstance(result, str)


if __name__ == '__main__':
    unittest.main()

