import unittest
from unittest.mock import patch, call
import os
from ais_bench.benchmark.utils.file import match_cfg_file
from ais_bench.benchmark.utils.logging import FileMatchError
from ais_bench.benchmark.utils.logging import UTILS_CODES


class TestMatchCfgFile(unittest.TestCase):
    def setUp(self):
        # 保存原始的os.walk函数
        self.original_os_walk = os.walk

    def tearDown(self):
        # 恢复原始的os.walk函数
        os.walk = self.original_os_walk

    @patch('ais_bench.benchmark.utils.file.logger')
    @patch('os.walk')
    def test_match_cfg_file_single_pattern_single_match(self, mock_os_walk, mock_logger):
        """测试单个模式匹配单个文件的情况"""
        # 模拟os.walk返回值
        mock_os_walk.return_value = [
            ('/test/dir', [], ['config1.py'])
        ]

        # 调用函数
        result = match_cfg_file('/test/dir', 'config1')

        # 验证结果
        expected = [('config1', '/test/dir/config1.py')]
        self.assertEqual(result, expected)

        # 验证os.walk被正确调用
        mock_os_walk.assert_called_once_with('/test/dir')

        # 验证logger.warning没有被调用
        mock_logger.warning.assert_not_called()

    @patch('ais_bench.benchmark.utils.file.logger')
    @patch('os.walk')
    def test_match_cfg_file_multiple_patterns_multiple_matches(self, mock_os_walk, mock_logger):
        """测试多个模式匹配多个文件的情况"""
        # 模拟os.walk返回值
        mock_os_walk.return_value = [
            ('/test/dir', [], ['config1.py', 'config2.py'])
        ]

        # 调用函数
        result = match_cfg_file('/test/dir', ['config1', 'config2'])

        # 验证结果
        expected = [('config1', '/test/dir/config1.py'), ('config2', '/test/dir/config2.py')]
        self.assertEqual(result, expected)

        # 验证logger.warning没有被调用
        mock_logger.warning.assert_not_called()

    @patch('ais_bench.benchmark.utils.file.logger')
    @patch('os.walk')
    def test_match_cfg_file_pattern_no_match(self, mock_os_walk, mock_logger):
        """测试模式不匹配任何文件的情况"""
        # 模拟os.walk返回值
        mock_os_walk.return_value = [
            ('/test/dir', [], ['other_config.py'])
        ]

        # 验证异常
        with self.assertRaises(FileMatchError) as context:
            match_cfg_file('/test/dir', 'config1')

        # 验证错误代码
        self.assertEqual(context.exception.error_code_str, UTILS_CODES.MATCH_CONFIG_FILE_FAILED.full_code)

    @patch('ais_bench.benchmark.utils.file.file.logger')
    @patch('os.walk')
    def test_match_cfg_file_pattern_multiple_matches(self, mock_os_walk, mock_logger):
        """测试模式匹配多个文件的情况（模糊匹配）"""
        # 模拟os.walk返回值
        mock_os_walk.return_value = [
            ('/test/dir', [], ['config1.py', 'config1_v2.py'])
        ]

        # 调用函数（使用'config1'作为模式，会匹配到两个文件）
        with patch('ais_bench.benchmark.utils.file.fnmatch.fnmatch') as mock_fnmatch:
            # 模拟fnmatch返回True，这样每个文件都会匹配
            mock_fnmatch.return_value = True
            result = match_cfg_file('/test/dir', 'config1')

        # 验证结果 - 应该返回第一个匹配的文件
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 'config1')

        # 验证logger.warning被调用
        mock_logger.warning.assert_called_once()

    @patch('ais_bench.benchmark.utils.file.file.logger')
    @patch('os.walk')
    def test_match_cfg_file_mixed_scenarios(self, mock_os_walk, mock_logger):
        """测试混合场景：有些模式匹配，有些匹配多个"""
        # 模拟os.walk返回值
        mock_os_walk.return_value = [
            ('/test/dir', [], ['config1.py', 'config2.py', 'config2_v2.py'])
        ]

        # 设置不同的fnmatch行为
        def side_effect(name, pattern):
            if 'config1.py' in name and 'config1.py' in pattern:
                return True
            elif 'config2' in name and 'config2.py' in pattern:
                return True
            return False

        with patch('ais_bench.benchmark.utils.file.fnmatch.fnmatch') as mock_fnmatch:
            mock_fnmatch.side_effect = side_effect

            # 调用函数
            result = match_cfg_file('/test/dir', ['config1', 'config2'])

        # 验证结果 - 对于模糊匹配的情况，结果可能是第一个匹配的文件
        self.assertEqual(len(result), 1)

        # 验证logger.warning被调用
        mock_logger.warning.assert_called_once()

    @patch('ais_bench.benchmark.utils.file.logger')
    @patch('os.walk')
    def test_match_cfg_file_auto_add_py_suffix(self, mock_os_walk, mock_logger):
        """测试自动添加.py后缀的功能"""
        # 模拟os.walk返回值
        mock_os_walk.return_value = [
            ('/test/dir', [], ['config1.py'])
        ]

        # 调用函数，不提供.py后缀
        result = match_cfg_file('/test/dir', 'config1')

        # 验证结果
        expected = [('config1', '/test/dir/config1.py')]
        self.assertEqual(result, expected)

    @patch('ais_bench.benchmark.utils.file.logger')
    @patch('os.walk')
    def test_match_cfg_file_with_py_suffix(self, mock_os_walk, mock_logger):
        """测试已经有.py后缀的情况"""
        # 模拟os.walk返回值
        mock_os_walk.return_value = [
            ('/test/dir', [], ['config1.py'])
        ]

        # 调用函数，已经提供.py后缀
        result = match_cfg_file('/test/dir', 'config1.py')

        # 验证结果
        expected = [('config1', '/test/dir/config1.py')]
        self.assertEqual(result, expected)

    @patch('ais_bench.benchmark.utils.file.logger')
    @patch('os.walk')
    def test_match_cfg_file_multiple_workdirs(self, mock_os_walk, mock_logger):
        """测试多个工作目录的情况"""
        # 模拟os.walk返回值 - 每个调用返回正确的3元组结构
        def walk_side_effect(path):
            if path == '/test/dir1':
                return [('/test/dir1', [], ['config1.py'])]
            elif path == '/test/dir2':
                return [('/test/dir2', [], ['config2.py'])]
            return []

        mock_os_walk.side_effect = walk_side_effect

        # 调用函数，提供多个工作目录
        result = match_cfg_file(['/test/dir1', '/test/dir2'], ['config1', 'config2'])

        # 验证结果
        self.assertEqual(len(result), 2)

        # 验证os.walk被调用了两次
        self.assertEqual(mock_os_walk.call_count, 2)
        mock_os_walk.assert_has_calls([call('/test/dir1'), call('/test/dir2')])

    @patch('ais_bench.benchmark.utils.file.logger')
    @patch('os.walk')
    def test_match_cfg_file_some_patterns_no_match(self, mock_os_walk, mock_logger):
        """测试部分模式不匹配的情况"""
        # 模拟os.walk返回值
        mock_os_walk.return_value = [
            ('/test/dir', [], ['config1.py'])
        ]

        # 验证异常
        with self.assertRaises(FileMatchError) as context:
            match_cfg_file('/test/dir', ['config1', 'nonexistent_config'])

        # 验证错误代码
        self.assertEqual(context.exception.error_code_str, UTILS_CODES.MATCH_CONFIG_FILE_FAILED.full_code)

    @patch('ais_bench.benchmark.utils.file.logger')
    def test_match_cfg_file_empty_pattern(self, mock_logger):
        """测试空模式的情况"""
        # 空模式实际会引发异常，而不是返回空列表
        with self.assertRaises(FileMatchError) as context:
            match_cfg_file('/test/dir', '')

        # 验证错误代码
        self.assertEqual(context.exception.error_code_str, UTILS_CODES.MATCH_CONFIG_FILE_FAILED.full_code)

    @patch('ais_bench.benchmark.utils.file.logger')
    @patch('os.walk')
    def test_match_cfg_file_empty_workdir(self, mock_os_walk, mock_logger):
        """测试空工作目录的情况"""
        # 模拟os.walk返回空列表
        mock_os_walk.return_value = []

        # 验证异常
        with self.assertRaises(FileMatchError) as context:
            match_cfg_file('', 'config1')

        # 验证错误代码
        self.assertEqual(context.exception.error_code_str, UTILS_CODES.MATCH_CONFIG_FILE_FAILED.full_code)


if __name__ == '__main__':
    unittest.main()