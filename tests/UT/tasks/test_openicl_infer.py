import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

from mmengine.config import ConfigDict

from ais_bench.benchmark.tasks.openicl_infer import OpenICLInferTask
from ais_bench.benchmark.tasks.base import TaskStateManager
from ais_bench.benchmark.utils.logging.error_codes import TINFER_CODES
from ais_bench.benchmark.utils.logging.exceptions import ParameterValueError


class TestOpenICLInferTask(unittest.TestCase):
    """测试OpenICLInferTask类"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.cfg = ConfigDict({
            "models": [{
                "type": "test_model",
                "run_cfg": {
                    "num_gpus": 1,
                    "num_procs": 1,
                    "nnodes": 1,
                    "node_rank": 0,
                    "master_addr": "localhost"
                },
                "generation_kwargs": {
                    "num_return_sequences": 1
                }
            }],
            "datasets": [{
                "type": "test_dataset",
                "infer_cfg": {
                    "inferencer": {"type": "test_inferencer"},
                    "retriever": {"type": "test_retriever"}
                }
            }],
            "work_dir": self.temp_dir,
            "cli_args": {}
        })

    def tearDown(self):
        """清理测试环境"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('ais_bench.benchmark.tasks.openicl_infer.AISLogger')
    def test_init(self, mock_logger_class):
        """测试OpenICLInferTask初始化"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        task = OpenICLInferTask(self.cfg)
        
        self.assertEqual(task.name_prefix, "OpenICLInfer")
        self.assertEqual(task.log_subdir, "logs/infer")
        self.assertEqual(task.output_subdir, "predictions")
        self.assertEqual(task.num_gpus, 1)
        self.assertEqual(task.num_procs, 1)
        self.assertEqual(task.nnodes, 1)
        self.assertEqual(task.node_rank, 0)
        self.assertEqual(task.master_addr, "localhost")

    @patch('ais_bench.benchmark.tasks.openicl_infer.AISLogger')
    def test_init_with_defaults(self, mock_logger_class):
        """测试使用默认值初始化"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        cfg = ConfigDict({
            "models": [{"type": "test_model"}],
            "datasets": [{"infer_cfg": {"inferencer": {"type": "test"}, "retriever": {}}}],
            "work_dir": self.temp_dir,
            "cli_args": {}
        })
        
        task = OpenICLInferTask(cfg)
        
        self.assertEqual(task.num_gpus, 0)
        self.assertEqual(task.num_procs, 1)
        self.assertEqual(task.nnodes, 1)
        self.assertEqual(task.node_rank, 0)
        self.assertEqual(task.master_addr, "localhost")

    @patch('ais_bench.benchmark.tasks.openicl_infer.AISLogger')
    def test_get_command_single_gpu(self, mock_logger_class):
        """测试单GPU命令生成"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        task = OpenICLInferTask(self.cfg)
        
        with patch('ais_bench.benchmark.tasks.openicl_infer.sys.executable', '/usr/bin/python'):
            cmd = task.get_command("/path/to/config.py", "CUDA_VISIBLE_DEVICES=0 {task_cmd}")
        
        self.assertIn("/usr/bin/python", cmd)
        self.assertIn("/path/to/config.py", cmd)

    @patch('ais_bench.benchmark.tasks.openicl_infer.AISLogger')
    def test_get_command_multi_gpu(self, mock_logger_class):
        """测试多GPU命令生成"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        cfg = ConfigDict({
            "models": [{
                "type": "test_model",
                "run_cfg": {
                    "num_gpus": 2,
                    "num_procs": 2,
                    "nnodes": 1
                }
            }],
            "datasets": [{"infer_cfg": {"inferencer": {}, "retriever": {}}}],
            "work_dir": self.temp_dir,
            "cli_args": {}
        })
        
        task = OpenICLInferTask(cfg)
        
        cmd = task.get_command("/path/to/config.py", "CUDA_VISIBLE_DEVICES=0,1 {task_cmd}")
        
        self.assertIn("torchrun", cmd)
        self.assertIn("--nproc_per_node", cmd)
        self.assertIn("2", cmd)

    @patch('ais_bench.benchmark.tasks.openicl_infer.AISLogger')
    def test_get_command_with_backend(self, mock_logger_class):
        """测试使用后端时的命令生成"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        cfg = ConfigDict({
            "models": [{
                "type": "VLLM_model",
                "run_cfg": {
                    "num_gpus": 2,
                    "num_procs": 2
                }
            }],
            "datasets": [{"infer_cfg": {"inferencer": {}, "retriever": {}}}],
            "work_dir": self.temp_dir,
            "cli_args": {}
        })
        
        task = OpenICLInferTask(cfg)
        
        cmd = task.get_command("/path/to/config.py", "CUDA_VISIBLE_DEVICES=0,1 {task_cmd}")
        
        # 使用后端时不应该使用torchrun
        self.assertNotIn("torchrun", cmd)

    @patch('ais_bench.benchmark.tasks.openicl_infer.AISLogger')
    def test_get_command_multi_node(self, mock_logger_class):
        """测试多节点命令生成"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        cfg = ConfigDict({
            "models": [{
                "type": "test_model",
                "run_cfg": {
                    "num_gpus": 2,
                    "num_procs": 2,
                    "nnodes": 2,
                    "node_rank": 0,
                    "master_addr": "192.168.1.1"
                }
            }],
            "datasets": [{"infer_cfg": {"inferencer": {}, "retriever": {}}}],
            "work_dir": self.temp_dir,
            "cli_args": {}
        })
        
        task = OpenICLInferTask(cfg)
        
        cmd = task.get_command("/path/to/config.py", "CUDA_VISIBLE_DEVICES=0,1 {task_cmd}")
        
        self.assertIn("torchrun", cmd)
        self.assertIn("--nnodes", cmd)
        self.assertIn("--node_rank", cmd)
        self.assertIn("--master_addr", cmd)

    @patch('ais_bench.benchmark.tasks.openicl_infer.AISLogger')
    @patch('ais_bench.benchmark.tasks.openicl_infer.ICL_INFERENCERS')
    @patch('ais_bench.benchmark.tasks.openicl_infer.ICL_RETRIEVERS')
    @patch('ais_bench.benchmark.tasks.openicl_infer.build_dataset_from_cfg')
    def test_build_inference(self, mock_build_dataset, mock_retrievers, mock_inferencers, mock_logger_class):
        """测试build_inference方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        mock_inferencer = MagicMock()
        mock_inferencers.build.return_value = mock_inferencer
        
        task = OpenICLInferTask(self.cfg)
        task.task_state_manager = MagicMock()
        task.infer_cfg = {"inferencer": {"type": "test"}}
        task.max_out_len = 512
        task.min_out_len = 1
        task.batch_size = 32
        
        task.build_inference()
        
        mock_inferencers.build.assert_called_once()
        mock_inferencer.set_task_state_manager.assert_called_once_with(task.task_state_manager)

    @patch('ais_bench.benchmark.tasks.openicl_infer.AISLogger')
    def test_set_default_value(self, mock_logger_class):
        """测试_set_default_value方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        task = OpenICLInferTask(self.cfg)
        
        cfg = ConfigDict({})
        task._set_default_value(cfg, "test_key", "test_value")
        
        self.assertEqual(cfg["test_key"], "test_value")

    @patch('ais_bench.benchmark.tasks.openicl_infer.AISLogger')
    def test_set_default_value_existing(self, mock_logger_class):
        """测试_set_default_value方法，键已存在时不覆盖"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        task = OpenICLInferTask(self.cfg)
        
        cfg = ConfigDict({"test_key": "existing_value"})
        task._set_default_value(cfg, "test_key", "new_value")
        
        self.assertEqual(cfg["test_key"], "existing_value")

    @patch('ais_bench.benchmark.tasks.openicl_infer.AISLogger')
    @patch('ais_bench.benchmark.tasks.openicl_infer.ICL_INFERENCERS')
    @patch('ais_bench.benchmark.tasks.openicl_infer.ICL_RETRIEVERS')
    @patch('ais_bench.benchmark.tasks.openicl_infer.build_dataset_from_cfg')
    @patch('ais_bench.benchmark.tasks.openicl_infer.task_abbr_from_cfg')
    def test_run(self, mock_task_abbr, mock_build_dataset, mock_retrievers, mock_inferencers, mock_logger_class):
        """测试run方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        mock_task_abbr.return_value = "test_task"
        
        mock_inferencer = MagicMock()
        mock_inferencers.build.return_value = mock_inferencer
        
        mock_retriever = MagicMock()
        mock_retrievers.build.return_value = mock_retriever
        
        task = OpenICLInferTask(self.cfg)
        # 修复：将task的logger设置为mock，这样才能验证调用
        task.logger = mock_logger
        # 修复：model_cfg需要abbr字段，否则model_abbr_from_cfg会需要path字段
        if "abbr" not in task.model_cfg:
            task.model_cfg["abbr"] = "test_model"
        task_state_manager = MagicMock()
        
        # 修复：dataset_cfgs在BaseTask中被设置为cfg["datasets"][0]（ConfigDict），
        # 但run方法中使用了dataset_cfgs[0]，所以需要将其转换为列表
        if not isinstance(task.dataset_cfgs, list):
            task.dataset_cfgs = [task.dataset_cfgs]
        
        task.run(task_state_manager)
        
        mock_inferencers.build.assert_called()
        mock_retrievers.build.assert_called()

    @patch('ais_bench.benchmark.tasks.openicl_infer.AISLogger')
    @patch('ais_bench.benchmark.tasks.openicl_infer.task_abbr_from_cfg')
    def test_run_with_invalid_num_return_sequences(self, mock_task_abbr, mock_logger_class):
        """测试num_return_sequences无效值"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        mock_task_abbr.return_value = "test_task"
        
        cfg = ConfigDict({
            "models": [{
                "type": "test_model",
                "generation_kwargs": {
                    "num_return_sequences": 0
                }
            }],
            "datasets": [{"infer_cfg": {"inferencer": {}, "retriever": {}}}],
            "work_dir": self.temp_dir,
            "cli_args": {}
        })
        
        task = OpenICLInferTask(cfg)
        # 修复：将task的logger设置为mock
        task.logger = mock_logger
        task_state_manager = MagicMock()
        
        # 修复：dataset_cfgs类型问题
        if not isinstance(task.dataset_cfgs, list):
            task.dataset_cfgs = [task.dataset_cfgs]
        
        with self.assertRaises(ParameterValueError) as context:
            task.run(task_state_manager)
        
        error_code = context.exception.error_code_str
        self.assertEqual(error_code, TINFER_CODES.NUM_RETURN_SEQUENCES_NOT_POSITIVE.full_code)

    @patch('ais_bench.benchmark.tasks.openicl_infer.AISLogger')
    @patch('ais_bench.benchmark.tasks.openicl_infer.ICL_INFERENCERS')
    @patch('ais_bench.benchmark.tasks.openicl_infer.ICL_RETRIEVERS')
    @patch('ais_bench.benchmark.tasks.openicl_infer.build_dataset_from_cfg')
    @patch('ais_bench.benchmark.tasks.openicl_infer.task_abbr_from_cfg')
    def test_run_with_multiple_return_sequences(self, mock_task_abbr, mock_build_dataset, mock_retrievers, mock_inferencers, mock_logger_class):
        """测试num_return_sequences大于1"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        mock_task_abbr.return_value = "test_task"
        
        mock_inferencer = MagicMock()
        mock_inferencers.build.return_value = mock_inferencer
        
        mock_retriever = MagicMock()
        mock_retrievers.build.return_value = mock_retriever
        
        cfg = ConfigDict({
            "models": [{
                "type": "test_model",
                "generation_kwargs": {
                    "num_return_sequences": 3
                }
            }],
            "datasets": [{"infer_cfg": {"inferencer": {}, "retriever": {}}}],
            "work_dir": self.temp_dir,
            "cli_args": {}
        })
        
        task = OpenICLInferTask(cfg)
        # 修复：将task的logger设置为mock，这样才能验证调用
        task.logger = mock_logger
        # 修复：model_cfg需要abbr字段，否则model_abbr_from_cfg会需要path字段
        if "abbr" not in task.model_cfg:
            task.model_cfg["abbr"] = "test_model"
        
        # 修复：dataset_cfgs类型问题
        if not isinstance(task.dataset_cfgs, list):
            task.dataset_cfgs = [task.dataset_cfgs]
        
        task_state_manager = MagicMock()
        task.run(task_state_manager)
        
        # 验证记录了信息日志
        mock_logger.info.assert_called()
        # 检查是否包含num_return_sequences相关的日志
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        has_num_return_log = any("num_return_sequences is greater than 1" in str(call) for call in mock_logger.info.call_args_list)
        self.assertTrue(has_num_return_log, f"Expected log about num_return_sequences, but got: {info_calls}")

    def test_parse_args(self):
        """测试parse_args函数"""
        import sys
        from unittest.mock import patch
        
        test_args = ['test_script', 'config.py']
        with patch.object(sys, 'argv', test_args):
            from ais_bench.benchmark.tasks.openicl_infer import parse_args
            args = parse_args()
            self.assertEqual(args.config, 'config.py')


if __name__ == '__main__':
    unittest.main()

