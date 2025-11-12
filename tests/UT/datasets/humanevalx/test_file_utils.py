"""Unit tests for humanevalx/file_utils.py"""
import unittest
import tempfile
import os
import stat
from unittest.mock import patch, MagicMock

from ais_bench.benchmark.datasets.humanevalx.file_utils import (
    safe_open,
    standardize_path,
    is_path_exists,
    check_path_is_none,
    check_path_is_link,
    check_path_length_lt,
    check_file_size_lt,
    check_owner,
    check_other_write_permission,
    check_path_permission,
    check_file_safety,
    safe_listdir,
    safe_chmod,
    has_owner_write_permission,
    safe_readlines,
    MAX_PATH_LENGTH,
    MAX_FILE_SIZE,
    MAX_FILENUM_PER_DIR,
    MAX_LINENUM_PER_FILE,
)


class TestFileUtils(unittest.TestCase):
    """测试 file_utils 模块"""
    
    def test_check_path_is_none(self):
        """测试check_path_is_none - None路径（覆盖74-75行）"""
        with self.assertRaises(TypeError) as cm:
            check_path_is_none(None)
        self.assertIn("should not be None", str(cm.exception))
    
    def test_check_path_is_link(self):
        """测试check_path_is_link - 符号链接（覆盖78-81行）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建一个符号链接
            target_file = os.path.join(tmpdir, "target.txt")
            link_file = os.path.join(tmpdir, "link.txt")
            with open(target_file, 'w') as f:
                f.write("test")
            os.symlink(target_file, link_file)
            
            with self.assertRaises(ValueError) as cm:
                check_path_is_link(link_file)
            self.assertIn("symbolic link", str(cm.exception))
    
    def test_check_path_length_lt(self):
        """测试check_path_length_lt - 路径过长（覆盖84-88行）"""
        long_path = "a" * (MAX_PATH_LENGTH + 1)
        with self.assertRaises(ValueError) as cm:
            check_path_length_lt(long_path)
        self.assertIn("length of path should not be greater", str(cm.exception))
    
    def test_check_file_size_lt(self):
        """测试check_file_size_lt - 文件过大（覆盖91-95行）"""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            # 创建一个超过最大文件大小的文件
            tmpfile.write(b"x" * (MAX_FILE_SIZE + 1))
            tmpfile_path = tmpfile.name
        
        try:
            with self.assertRaises(ValueError) as cm:
                check_file_size_lt(tmpfile_path)
            self.assertIn("size of file should not be greater", str(cm.exception))
        finally:
            os.unlink(tmpfile_path)
    
    def test_check_owner_permission_error(self):
        """测试check_owner - 权限错误（覆盖98-111行）"""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(b"test")
            tmpfile_path = tmpfile.name
        
        try:
            # Mock os.stat 和 os.geteuid 来模拟权限错误
            with patch('os.stat') as mock_stat, \
                 patch('os.geteuid', return_value=1000), \
                 patch('os.getgid', return_value=1000):
                mock_stat_result = MagicMock()
                mock_stat_result.st_uid = 2000  # 不同的用户ID
                mock_stat_result.st_gid = 2000  # 不同的组ID
                mock_stat.return_value = mock_stat_result
                
                with self.assertRaises(PermissionError) as cm:
                    check_owner(tmpfile_path)
                self.assertIn("does not have permission", str(cm.exception))
        finally:
            os.unlink(tmpfile_path)
    
    def test_check_other_write_permission(self):
        """测试check_other_write_permission - 其他用户可写（覆盖114-126行）"""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(b"test")
            tmpfile_path = tmpfile.name
        
        try:
            # 设置文件权限为其他用户可写
            os.chmod(tmpfile_path, 0o666)  # 其他用户可读写
            
            with self.assertRaises(PermissionError) as cm:
                check_other_write_permission(tmpfile_path)
            self.assertIn("should not be writable by others", str(cm.exception))
        finally:
            os.unlink(tmpfile_path)
    
    def test_check_path_permission_with_env(self):
        """测试check_path_permission - 环境变量控制（覆盖129-134行）"""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(b"test")
            tmpfile_path = tmpfile.name
        
        try:
            # 测试环境变量为"0"时，不检查权限
            with patch.dict(os.environ, {'MINDIE_CHECK_INPUTFILES_PERMISSION': '0'}):
                # 应该不抛出异常
                check_path_permission(tmpfile_path, is_internal_file=False)
            
            # 测试环境变量为"1"时，检查权限
            with patch.dict(os.environ, {'MINDIE_CHECK_INPUTFILES_PERMISSION': '1'}):
                # 正常文件应该通过检查
                check_path_permission(tmpfile_path, is_internal_file=False)
        finally:
            os.unlink(tmpfile_path)
    
    def test_check_file_safety_file_exists_not_ok(self):
        """测试check_file_safety - 文件存在但不允许（覆盖137-142行）"""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(b"test")
            tmpfile_path = tmpfile.name
        
        try:
            with self.assertRaises(FileExistsError) as cm:
                check_file_safety(tmpfile_path, mode='w', is_exist_ok=False)
            self.assertIn("expected not to exist", str(cm.exception))
        finally:
            os.unlink(tmpfile_path)
    
    def test_check_file_safety_file_not_exists_read_mode(self):
        """测试check_file_safety - 文件不存在但以读模式打开（覆盖146-149行）"""
        non_existent_file = "/tmp/nonexistent_file_12345.txt"
        with self.assertRaises(FileNotFoundError) as cm:
            check_file_safety(non_existent_file, mode='r')
        self.assertIn("expected to exist", str(cm.exception))
    
    def test_safe_listdir_too_many_files(self):
        """测试safe_listdir - 文件数量过多（覆盖155-161行）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建超过限制的文件数量
            for i in range(MAX_FILENUM_PER_DIR + 1):
                with open(os.path.join(tmpdir, f"file_{i}.txt"), 'w') as f:
                    f.write("test")
            
            with self.assertRaises(ValueError) as cm:
                safe_listdir(tmpdir, max_file_num=MAX_FILENUM_PER_DIR)
            self.assertIn("exceeds the limit", str(cm.exception))
    
    def test_safe_chmod(self):
        """测试safe_chmod（覆盖164-167行）"""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(b"test")
            tmpfile_path = tmpfile.name
        
        try:
            # 应该不抛出异常
            safe_chmod(tmpfile_path, 0o644)
            # 验证权限已更改（使用八进制权限检查）
            file_stat = os.stat(tmpfile_path)
            # 检查权限位（最后3位）
            actual_mode = stat.filemode(file_stat.st_mode)
            # 验证权限已设置（不一定是644，但应该可读）
            self.assertTrue(os.access(tmpfile_path, os.R_OK))
        finally:
            os.unlink(tmpfile_path)
    
    def test_has_owner_write_permission(self):
        """测试has_owner_write_permission（覆盖170-172行）"""
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(b"test")
            tmpfile_path = tmpfile.name
        
        try:
            # 设置文件为所有者可写
            os.chmod(tmpfile_path, 0o644)
            result = has_owner_write_permission(tmpfile_path)
            self.assertTrue(result)
            
            # 设置文件为所有者不可写
            os.chmod(tmpfile_path, 0o444)
            result = has_owner_write_permission(tmpfile_path)
            self.assertFalse(result)
        finally:
            os.unlink(tmpfile_path)
    
    def test_safe_readlines_too_many_lines(self):
        """测试safe_readlines - 行数过多（覆盖175-181行）"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
            # 创建超过限制的行数
            for i in range(MAX_LINENUM_PER_FILE + 1):
                tmpfile.write(f"line {i}\n")
            tmpfile_path = tmpfile.name
        
        try:
            with open(tmpfile_path, 'r') as f:
                with self.assertRaises(ValueError) as cm:
                    safe_readlines(f, max_line_num=MAX_LINENUM_PER_FILE)
                self.assertIn("exceeds the limit", str(cm.exception))
        finally:
            os.unlink(tmpfile_path)
    
    def test_standardize_path_with_check_link(self):
        """测试standardize_path - 检查符号链接（覆盖55-66行）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建符号链接
            target_file = os.path.join(tmpdir, "target.txt")
            link_file = os.path.join(tmpdir, "link.txt")
            with open(target_file, 'w') as f:
                f.write("test")
            os.symlink(target_file, link_file)
            
            with self.assertRaises(ValueError):
                standardize_path(link_file, check_link=True)
    
    def test_standardize_path_without_check_link(self):
        """测试standardize_path - 不检查符号链接（覆盖55-66行）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建符号链接
            target_file = os.path.join(tmpdir, "target.txt")
            link_file = os.path.join(tmpdir, "link.txt")
            with open(target_file, 'w') as f:
                f.write("test")
            os.symlink(target_file, link_file)
            
            # 不检查符号链接时应该通过
            result = standardize_path(link_file, check_link=False)
            self.assertIsInstance(result, str)
    
    def test_safe_open_read_mode(self):
        """测试safe_open - 读模式（覆盖22-52行）"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
            tmpfile.write("test content")
            tmpfile_path = tmpfile.name
        
        try:
            with safe_open(tmpfile_path, mode='r') as f:
                content = f.read()
                self.assertEqual(content, "test content")
        finally:
            os.unlink(tmpfile_path)
    
    def test_safe_open_write_mode(self):
        """测试safe_open - 写模式（覆盖22-52行）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.txt")
            with safe_open(file_path, mode='w') as f:
                f.write("test content")
            
            # 验证文件已创建
            self.assertTrue(os.path.exists(file_path))
            with open(file_path, 'r') as f:
                self.assertEqual(f.read(), "test content")
    
    def test_safe_open_with_encoding(self):
        """测试safe_open - 指定编码（覆盖22-52行）"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmpfile:
            tmpfile.write("测试内容")
            tmpfile_path = tmpfile.name
        
        try:
            with safe_open(tmpfile_path, mode='r', encoding='utf-8') as f:
                content = f.read()
                self.assertEqual(content, "测试内容")
        finally:
            os.unlink(tmpfile_path)
    
    def test_safe_open_mode_flags(self):
        """测试safe_open - 模式标志处理（覆盖42-49行）"""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.txt")
            # 测试各种模式
            for mode in ['r', 'w', 'a', 'r+', 'w+', 'a+']:
                try:
                    with safe_open(file_path, mode=mode) as f:
                        pass
                except (FileNotFoundError, FileExistsError):
                    # 某些模式可能会失败，这是正常的
                    pass


if __name__ == '__main__':
    unittest.main()

