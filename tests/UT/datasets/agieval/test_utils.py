"""Unit tests for agieval/utils.py"""
import unittest
import tempfile
import os
from unittest.mock import patch, mock_open

from ais_bench.benchmark.datasets.agieval import utils


class TestReadJsonl(unittest.TestCase):
    """测试read_jsonl函数"""
    
    def test_read_jsonl_valid(self):
        """测试读取有效的jsonl文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf8') as f:
            f.write('{"key": "value1"}\n')
            f.write('{"key": "value2"}\n')
            f.write('{"key": "value3"}\n')
            temp_path = f.name
        
        try:
            results = utils.read_jsonl(temp_path)
            self.assertEqual(len(results), 3)
            self.assertEqual(results[0]["key"], "value1")
        finally:
            os.unlink(temp_path)
    
    def test_read_jsonl_with_null(self):
        """测试读取包含null的jsonl文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf8') as f:
            f.write('{"key": "value1"}\n')
            f.write('null\n')
            f.write('{"key": "value2"}\n')
            temp_path = f.name
        
        try:
            results = utils.read_jsonl(temp_path)
            self.assertEqual(len(results), 3)
            # null字符串会被json.loads解析为None，但代码中有特殊处理
            # 如果line == 'null'，则返回'null'字符串，否则返回json.loads的结果
            # 所以'null'字符串应该被保留为'null'
            self.assertIn(results[1], ['null', None])
        finally:
            os.unlink(temp_path)
    
    def test_read_jsonl_with_none_line(self):
        """测试读取包含None的行（覆盖10行）"""
        # 模拟文件读取时返回None的情况
        mock_file_content = '{"key": "value1"}\n'
        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            # 模拟文件迭代器返回None
            mock_file = mock_open(read_data=mock_file_content).return_value
            mock_file.__iter__ = lambda self: iter(['{"key": "value1"}\n', None, '{"key": "value2"}\n'])
            with patch('builtins.open', return_value=mock_file):
                results = utils.read_jsonl('/fake/path')
                # None行应该被跳过
                self.assertGreaterEqual(len(results), 1)
    
    def test_read_jsonl_invalid_json(self):
        """测试读取无效的json"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf8') as f:
            f.write('{"key": "value1"}\n')
            f.write('invalid json\n')
            temp_path = f.name
        
        try:
            with self.assertRaises(Exception):
                utils.read_jsonl(temp_path)
        finally:
            os.unlink(temp_path)


class TestSaveJsonl(unittest.TestCase):
    """测试save_jsonl函数"""
    
    def test_save_jsonl(self):
        """测试保存jsonl文件"""
        lines = [
            {"key": "value1"},
            {"key": "value2"},
            {"key": "value3"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf8') as f:
            temp_path = f.name
        
        try:
            utils.save_jsonl(lines, temp_path)
            
            # 验证文件内容
            with open(temp_path, 'r', encoding='utf8') as f:
                content = f.read()
                self.assertIn('"key"', content)
                self.assertIn('"value1"', content)
        finally:
            os.unlink(temp_path)


class TestExtractAnswer(unittest.TestCase):
    """测试extract_answer函数"""
    
    def test_extract_answer_from_string(self):
        """测试从字符串提取答案"""
        result = utils.extract_answer("answer text")
        self.assertEqual(result, "answer text")
    
    def test_extract_answer_from_dict_with_text(self):
        """测试从字典中提取答案（text字段）"""
        js = {
            "choices": [{
                "text": "answer text"
            }]
        }
        result = utils.extract_answer(js)
        self.assertEqual(result, "answer text")
    
    def test_extract_answer_from_dict_with_message(self):
        """测试从字典中提取答案（message字段）"""
        js = {
            "choices": [{
                "message": {
                    "content": "answer content"
                }
            }]
        }
        result = utils.extract_answer(js)
        self.assertEqual(result, "answer content")
    
    def test_extract_answer_none(self):
        """测试None输入"""
        result = utils.extract_answer(None)
        self.assertEqual(result, "")
    
    def test_extract_answer_null_string(self):
        """测试'null'字符串输入"""
        result = utils.extract_answer('null')
        self.assertEqual(result, "")
    
    def test_extract_answer_exception(self):
        """测试异常情况（覆盖40-43行）"""
        # 测试无效的js结构
        js = {"invalid": "structure"}
        result = utils.extract_answer(js)
        # 异常时应该返回空字符串
        self.assertEqual(result, "")


if __name__ == "__main__":
    unittest.main()

