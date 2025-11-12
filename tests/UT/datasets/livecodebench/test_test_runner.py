import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    # test_runner使用相对导入from testing_util import run_test
    # 需要确保testing_util在路径中，然后导入test_runner
    # 由于test_runner使用相对导入，我们需要先导入testing_util，然后导入test_runner
    import sys
    import os
    import importlib.util
    
    # 获取项目根目录
    # 从当前文件路径计算：tests/UT/datasets/livecodebench/test_test_runner.py
    # 回到项目根目录：../../../../.. -> benchmark目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../../../..'))
    
    # 验证项目根目录是否正确（应该包含 ais_bench 目录）
    if not os.path.exists(os.path.join(project_root, 'ais_bench')):
        # 如果不存在，尝试使用sys.path中已添加的路径
        for path in sys.path:
            if os.path.exists(os.path.join(path, 'ais_bench')):
                project_root = path
                break
        else:
            # 最后尝试使用当前工作目录
            cwd = os.getcwd()
            if os.path.exists(os.path.join(cwd, 'ais_bench')):
                project_root = cwd
    
    livecodebench_dir = os.path.join(project_root, 'ais_bench', 'benchmark', 'datasets', 'livecodebench')
    
    # 验证路径存在
    if not os.path.exists(livecodebench_dir):
        raise ImportError(f"livecodebench directory not found: {livecodebench_dir}")
    
    # 先导入testing_util，这样test_runner的相对导入才能工作
    # 使用绝对导入先导入testing_util
    from ais_bench.benchmark.datasets.livecodebench import testing_util
    
    # 将testing_util添加到sys.modules中，使用相对导入的名称
    # 这样test_runner的相对导入from testing_util import run_test才能工作
    sys.modules['testing_util'] = testing_util
    
    # 使用importlib直接加载test_runner模块
    test_runner_path = os.path.join(livecodebench_dir, 'test_runner.py')
    if not os.path.exists(test_runner_path):
        raise ImportError(f"test_runner.py not found: {test_runner_path}")
    
    spec = importlib.util.spec_from_file_location("test_runner", test_runner_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {test_runner_path}")
    
    test_runner_module = importlib.util.module_from_spec(spec)
    # 在执行模块之前，确保testing_util在sys.modules中
    sys.modules['test_runner'] = test_runner_module
    spec.loader.exec_module(test_runner_module)
    
    test_runner = test_runner_module
    TEST_RUNNER_AVAILABLE = True
except (ImportError, Exception) as e:
    TEST_RUNNER_AVAILABLE = False
    test_runner = None
    import traceback
    print(f"Failed to import test_runner: {e}")
    traceback.print_exc()


class TestRunnerTestBase(unittest.TestCase):
    """TestRunner测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not TEST_RUNNER_AVAILABLE:
            cls.skipTest(cls, "TestRunner modules not available")


class TestTestRunner(TestRunnerTestBase):
    """测试test_runner模块"""
    
    def test_main_success(self):
        """测试main函数成功执行"""
        from unittest.mock import patch
        
        # 模拟stdin输入 - test_runner使用json.load(sys.stdin)
        mock_stdin_json = {
            'sample': {'input_output': '{"inputs": ["test"], "outputs": ["result"]}'},
            'generation': 'def test(): return "result"',
            'debug': False,
            'timeout': 10
        }
        
        # test_runner中json.load会从sys.stdin读取，我们需要patch json.load
        # 由于test_runner使用相对导入from testing_util import run_test，需要patch test_runner模块中的run_test
        with patch.object(test_runner_module, 'json') as mock_json, \
             patch.object(test_runner_module, 'run_test', return_value=([True], {})) as mock_run_test, \
             patch('builtins.print') as mock_print, \
             patch('sys.exit') as mock_exit:
            mock_json.load.return_value = mock_stdin_json
            
            # 执行main函数
            test_runner_module.main()
            
            # 验证run_test被调用
            mock_run_test.assert_called_once_with(
                mock_stdin_json['sample'],
                test=mock_stdin_json['generation'],
                debug=False,
                timeout=10
            )
            # 验证打印了JSON输出
            self.assertTrue(mock_print.called)
            # 验证退出码为0
            mock_exit.assert_called_with(0)
    
    def test_main_with_exception(self):
        """测试main函数异常情况"""
        from unittest.mock import patch
        
        # 模拟JSON加载失败
        with patch.object(test_runner_module, 'json') as mock_json, \
             patch('builtins.print') as mock_print, \
             patch('sys.exit') as mock_exit:
            mock_json.load.side_effect = ValueError('Invalid JSON')
            # 执行main函数
            test_runner_module.main()
            
            # 验证打印了错误信息（两次：一次stdout，一次stderr）
            self.assertTrue(mock_print.called)
            # 验证退出码为2
            mock_exit.assert_called_with(2)
    
    def test_main_with_default_params(self):
        """测试main函数使用默认参数"""
        from unittest.mock import patch
        
        mock_stdin_json = {
            'sample': {'input_output': '{"inputs": ["test"], "outputs": ["result"]}'},
            'generation': 'def test(): return "result"'
            # 没有提供debug和timeout，应该使用默认值
        }
        
        with patch.object(test_runner_module, 'json') as mock_json, \
             patch.object(test_runner_module, 'run_test', return_value=([True], {})) as mock_run_test, \
             patch('builtins.print') as mock_print, \
             patch('sys.exit') as mock_exit:
            mock_json.load.return_value = mock_stdin_json
            
            # 执行main函数
            test_runner_module.main()
            
            # 验证使用了默认值
            mock_run_test.assert_called_once_with(
                mock_stdin_json['sample'],
                test=mock_stdin_json['generation'],
                debug=False,
                timeout=10
            )
            mock_exit.assert_called_with(0)


if __name__ == '__main__':
    unittest.main()

