import sys
import os
import pytest
from unittest.mock import patch, MagicMock, call

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from ais_bench.benchmark.cli.task_manager import TaskManager


class TestTaskManager:

    @pytest.fixture
    def setup_mocks(self):
        """设置所有需要的mock对象"""
        # Mock ArgumentParser
        with patch('ais_bench.benchmark.cli.task_manager.ArgumentParser') as mock_arg_parser:
            # Mock ConfigManager (注意：这是在__init__方法内部导入的)
            with patch('ais_bench.benchmark.cli.config_manager.ConfigManager') as mock_config_manager:
                # Mock WORK_FLOW and WorkFlowExecutor (这些是在run方法内部导入的)
                with patch('ais_bench.benchmark.cli.workers.WORK_FLOW') as mock_work_flow:
                    with patch('ais_bench.benchmark.cli.workers.WorkFlowExecutor') as mock_workflow_executor:
                        # 设置基本的mock行为
                        mock_args = MagicMock()
                        mock_args_parser_instance = mock_arg_parser.return_value
                        mock_args_parser_instance.parse_args.return_value = mock_args

                        mock_config_manager_instance = mock_config_manager.return_value

                        mock_worker_class = MagicMock()
                        mock_worker_instance = MagicMock()
                        mock_worker_class.return_value = mock_worker_instance
                        mock_work_flow.get.return_value = [mock_worker_class]

                        mock_workflow_executor_instance = mock_workflow_executor.return_value

                        yield {
                            'mock_arg_parser': mock_arg_parser,
                            'mock_config_manager': mock_config_manager,
                            'mock_work_flow': mock_work_flow,
                            'mock_workflow_executor': mock_workflow_executor,
                            'mock_args': mock_args,
                            'mock_config_manager_instance': mock_config_manager_instance,
                            'mock_worker_class': mock_worker_class,
                            'mock_worker_instance': mock_worker_instance,
                            'mock_workflow_executor_instance': mock_workflow_executor_instance
                        }

    def test_init(self, setup_mocks):
        """测试TaskManager初始化方法"""
        mocks = setup_mocks

        # 创建TaskManager实例
        task_manager = TaskManager()

        # 验证ArgumentParser被正确初始化和调用
        mocks['mock_arg_parser'].assert_called_once()
        mocks['mock_arg_parser'].return_value.parse_args.assert_called_once()

        # 验证ConfigManager被正确初始化
        mocks['mock_config_manager'].assert_called_once_with(mocks['mock_args'])

        # 验证属性被正确设置
        assert task_manager.args_parser == mocks['mock_arg_parser'].return_value
        assert task_manager.args == mocks['mock_args']
        assert task_manager.config_manager == mocks['mock_config_manager_instance']

    def test_run_search_mode(self, setup_mocks):
        """测试run方法的搜索模式分支"""
        mocks = setup_mocks

        # 设置search参数为True
        mocks['mock_args'].search = True

        # 创建TaskManager实例
        task_manager = TaskManager()

        # 调用run方法
        task_manager.run()

        # 验证config_manager.search_configs_location被调用
        mocks['mock_config_manager_instance'].search_configs_location.assert_called_once()

        # 验证搜索模式下不执行后续操作
        mocks['mock_work_flow'].get.assert_not_called()
        mocks['mock_workflow_executor'].assert_not_called()

    def test_run_normal_mode(self, setup_mocks):
        """测试run方法的正常执行模式分支"""
        mocks = setup_mocks

        # 设置search参数为False，并设置mode参数
        mocks['mock_args'].search = False
        mocks['mock_args'].mode = 'test_mode'

        # 设置config_manager.load_config的返回值
        mock_cfg = MagicMock()
        mocks['mock_config_manager_instance'].load_config.return_value = mock_cfg

        # 创建TaskManager实例
        task_manager = TaskManager()

        # 调用run方法
        task_manager.run()

        # 验证搜索方法未被调用
        mocks['mock_config_manager_instance'].search_configs_location.assert_not_called()

        # 验证WORK_FLOW.get被正确调用
        mocks['mock_work_flow'].get.assert_called_once_with('test_mode')

        # 验证worker类被正确实例化
        mocks['mock_worker_class'].assert_called_once_with(mocks['mock_args'])

        # 验证config_manager.load_config被正确调用
        mocks['mock_config_manager_instance'].load_config.assert_called_once_with([mocks['mock_worker_instance']])

        # 验证WorkFlowExecutor被正确初始化和执行
        mocks['mock_workflow_executor'].assert_called_once_with(mock_cfg, [mocks['mock_worker_instance']])
        mocks['mock_workflow_executor_instance'].execute.assert_called_once()

    def test_run_with_empty_workflow(self, setup_mocks):
        """测试run方法在WORK_FLOW返回空列表的情况"""
        mocks = setup_mocks

        # 设置search参数为False
        mocks['mock_args'].search = False
        mocks['mock_args'].mode = 'empty_mode'

        # 设置WORK_FLOW.get返回空列表
        mocks['mock_work_flow'].get.return_value = []

        # 设置config_manager.load_config的返回值
        mock_cfg = MagicMock()
        mocks['mock_config_manager_instance'].load_config.return_value = mock_cfg

        # 创建TaskManager实例
        task_manager = TaskManager()

        # 调用run方法
        task_manager.run()

        # 验证WORK_FLOW.get被正确调用
        mocks['mock_work_flow'].get.assert_called_once_with('empty_mode')

        # 验证config_manager.load_config被调用时传入空列表
        mocks['mock_config_manager_instance'].load_config.assert_called_once_with([])

        # 验证WorkFlowExecutor被正确初始化，传入空的workflow列表
        mocks['mock_workflow_executor'].assert_called_once_with(mock_cfg, [])
        mocks['mock_workflow_executor_instance'].execute.assert_called_once()

    def test_run_with_multiple_workers(self, setup_mocks):
        """测试run方法在有多个worker的情况"""
        mocks = setup_mocks

        # 设置search参数为False
        mocks['mock_args'].search = False
        mocks['mock_args'].mode = 'multi_mode'

        # 创建多个mock worker类
        mock_worker_class1 = MagicMock()
        mock_worker_instance1 = MagicMock()
        mock_worker_class1.return_value = mock_worker_instance1

        mock_worker_class2 = MagicMock()
        mock_worker_instance2 = MagicMock()
        mock_worker_class2.return_value = mock_worker_instance2

        # 设置WORK_FLOW.get返回多个worker类
        mocks['mock_work_flow'].get.return_value = [mock_worker_class1, mock_worker_class2]

        # 设置config_manager.load_config的返回值
        mock_cfg = MagicMock()
        mocks['mock_config_manager_instance'].load_config.return_value = mock_cfg

        # 创建TaskManager实例
        task_manager = TaskManager()

        # 调用run方法
        task_manager.run()

        # 验证所有worker类被正确实例化
        mock_worker_class1.assert_called_once_with(mocks['mock_args'])
        mock_worker_class2.assert_called_once_with(mocks['mock_args'])

        # 验证config_manager.load_config被调用时传入所有worker实例
        expected_workers = [mock_worker_instance1, mock_worker_instance2]
        mocks['mock_config_manager_instance'].load_config.assert_called_once_with(expected_workers)

        # 验证WorkFlowExecutor被正确初始化，传入所有worker实例
        mocks['mock_workflow_executor'].assert_called_once_with(mock_cfg, expected_workers)
        mocks['mock_workflow_executor_instance'].execute.assert_called_once()