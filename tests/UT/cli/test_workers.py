import sys
import os
import pytest
from unittest.mock import patch, MagicMock, call
from collections import defaultdict

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from mmengine.config import ConfigDict

from ais_bench.benchmark.cli.workers import (
    BaseWorker,
    Infer,
    Eval,
    AccViz,
    PerfViz,
    WorkFlowExecutor,
    WORK_FLOW
)

# 创建一个模拟ConfigDict类，支持点访问和merge_from_dict方法
class MockConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 将嵌套字典转换为MockConfigDict
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = MockConfigDict(value)
            elif isinstance(value, list):
                self[key] = [MockConfigDict(item) if isinstance(item, dict) else item for item in value]
    
    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"'MockConfigDict' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        if isinstance(value, dict):
            self[name] = MockConfigDict(value)
        else:
            self[name] = value
    
    def merge_from_dict(self, data):
        for key, value in data.items():
            if isinstance(value, dict) and key in self and isinstance(self[key], dict):
                if not isinstance(self[key], MockConfigDict):
                    self[key] = MockConfigDict(self[key])
                if not isinstance(value, MockConfigDict):
                    value = MockConfigDict(value)
                self[key].merge_from_dict(value)
            else:
                if isinstance(value, dict):
                    self[key] = MockConfigDict(value)
                else:
                    self[key] = value
    
    def get(self, key, default=None):
        return super().get(key, default)


class TestBaseWorker:
    def test_init(self):
        """测试BaseWorker初始化（使用具体子类测试）"""
        mock_args = MagicMock()
        # 创建一个临时子类来测试抽象基类
        class ConcreteWorker(BaseWorker):
            def update_cfg(self, cfg):
                pass
            def do_work(self, cfg):
                pass
        
        worker = ConcreteWorker(mock_args)
        assert worker.args == mock_args


class TestInfer:
    def setup_method(self):
        """设置测试环境"""
        self.mock_args = MagicMock()
        self.mock_args.max_num_workers = 4
        self.mock_args.max_workers_per_gpu = 2
        self.mock_args.debug = False
        self.infer_worker = Infer(self.mock_args)

    @patch('ais_bench.benchmark.cli.workers.get_config_type')
    @patch('ais_bench.benchmark.cli.workers.fill_model_path_if_datasets_need')
    def test_update_cfg_service_model(self, mock_fill_model_path, mock_get_config_type):
        """测试update_cfg方法，使用service模型"""
        # 设置mock返回值
        mock_get_config_type.side_effect = ['MockNaivePartitioner', 'MockOpenICLApiInferTask', 'MockLocalRunner']
        
        # 创建测试配置 - 使用MockConfigDict
        cfg = MockConfigDict({
            'models': [{'attr': 'service', 'abbr': 'test_model'}],
            'datasets': [{
                'abbr': 'test_dataset',
                'infer_cfg': {
                    'retriever': {},
                    'prompt_template': 'test_prompt',
                    'ice_template': 'test_ice'
                }
            }],
            'work_dir': '/test/workdir',
            'cli_args': MagicMock(debug=False)
        })
        
        # 执行测试
        with patch('os.path.join', return_value='/test/workdir/predictions/'):
            result = self.infer_worker.update_cfg(cfg)
        
        # 验证结果
        assert result == cfg
        assert cfg['infer']['partitioner']['type'] == 'MockNaivePartitioner'
        assert cfg['infer']['runner']['type'] == 'MockLocalRunner'
        assert cfg['infer']['runner']['task']['type'] == 'MockOpenICLApiInferTask'
        assert cfg['infer']['runner']['max_num_workers'] == 4
        assert cfg['infer']['runner']['max_workers_per_gpu'] == 2
        assert cfg['infer']['runner']['debug'] == False
        assert cfg['infer']['partitioner']['out_dir'] == '/test/workdir/predictions/'
        # 注意：在Infer.update_cfg中，prompt_template和ice_template不会被设置到retriever中
        # 它们是在_fill_dataset_configs中设置的，而_fill_dataset_configs是在ConfigManager.load_config中调用的
        # 所以这里不应该验证这些字段
        
        # 注意：fill_model_path_if_datasets_need是在_fill_dataset_configs中调用的，不是在update_cfg中
        # 所以这里不应该验证它被调用

    @patch('ais_bench.benchmark.cli.workers.get_config_type')
    @patch('ais_bench.benchmark.cli.workers.fill_model_path_if_datasets_need')
    def test_update_cfg_local_model(self, mock_fill_model_path, mock_get_config_type):
        """测试update_cfg方法，使用local模型"""
        # 设置mock返回值
        mock_get_config_type.side_effect = ['MockNaivePartitioner', 'MockOpenICLInferTask', 'MockLocalRunner']
        
        # 创建测试配置 - 使用MockConfigDict
        cfg = MockConfigDict({
            'models': [{'attr': 'local', 'abbr': 'test_model'}],
            'datasets': [{
                'abbr': 'test_dataset',
                'infer_cfg': {
                    'retriever': {},
                }
            }],
            'work_dir': '/test/workdir',
            'cli_args': MagicMock(debug=True)
        })
        
        # 执行测试
        with patch('os.path.join', return_value='/test/workdir/predictions/'):
            self.infer_worker.update_cfg(cfg)
        
        # 验证结果
        assert cfg['infer']['runner']['task']['type'] == 'MockOpenICLInferTask'
        assert cfg['infer']['runner']['debug'] == True  # 应该从cli_args获取debug值

    @patch('ais_bench.benchmark.cli.workers.PARTITIONERS')
    @patch('ais_bench.benchmark.cli.workers.RUNNERS')
    @patch('ais_bench.benchmark.cli.workers.logger')
    def test_do_work_no_merge(self, mock_logger, mock_runners, mock_partitioners):
        """测试do_work方法，不合并数据集的情况"""
        # 设置mock对象
        mock_partitioner = MagicMock()
        mock_partitioners.build.return_value = mock_partitioner
        mock_tasks = [MagicMock()]
        mock_partitioner.return_value = mock_tasks
        
        mock_runner = MagicMock()
        mock_runners.build.return_value = mock_runner
        
        # 创建测试配置 - 使用MockConfigDict
        cfg = MockConfigDict({
            'infer': {
                'partitioner': {},
                'runner': {}
            },
            'cli_args': MagicMock(merge_ds=False, mode='infer')
        })
        
        # 模拟_update_tasks_cfg方法
        with patch.object(self.infer_worker, '_update_tasks_cfg') as mock_update_tasks_cfg:
            # 执行测试
            self.infer_worker.do_work(cfg)
            
            # 验证结果
            mock_partitioners.build.assert_called_once_with(cfg['infer']['partitioner'])
            mock_partitioner.assert_called_once_with(cfg)
            mock_runners.build.assert_called_once_with(cfg['infer']['runner'])
            mock_runner.assert_called_once_with(mock_tasks)
            mock_update_tasks_cfg.assert_called_once_with(mock_tasks, cfg)
            
            # 验证正确的日志调用
            logs_called = [call for call in mock_logger.info.call_args_list]
            assert call("Starting inference tasks...") in logs_called
            assert call("Inference tasks completed.") in logs_called

    @patch('ais_bench.benchmark.cli.workers.PARTITIONERS')
    @patch('ais_bench.benchmark.cli.workers.RUNNERS')
    @patch('ais_bench.benchmark.cli.workers.logger')
    def test_do_work_merge_datasets(self, mock_logger, mock_runners, mock_partitioners):
        """测试do_work方法，合并数据集的情况"""
        # 设置mock对象
        mock_partitioner = MagicMock()
        mock_partitioners.build.return_value = mock_partitioner
        
        # 创建模拟任务
        task1 = {
            'models': [{'abbr': 'model1'}],
            'datasets': [[{'type': 'dataset_type', 'infer_cfg': {'inferencer': 'inferencer_type'}}]]
        }
        task2 = {
            'models': [{'abbr': 'model1'}],
            'datasets': [[{'type': 'dataset_type', 'infer_cfg': {'inferencer': 'inferencer_type'}}]]
        }
        mock_tasks = [task1, task2]
        mock_partitioner.return_value = mock_tasks
        
        mock_runner = MagicMock()
        mock_runners.build.return_value = mock_runner
        
        # 创建测试配置 - 使用MockConfigDict
        cfg = MockConfigDict({
            'infer': {
                'partitioner': {},
                'runner': {}
            },
            'cli_args': MagicMock(merge_ds=True, mode='infer')
        })
        
        # 模拟_update_tasks_cfg方法
        with patch.object(self.infer_worker, '_update_tasks_cfg'):
            # 执行测试
            self.infer_worker.do_work(cfg)
            
            # 验证结果
            logs_called = [call for call in mock_logger.info.call_args_list]
            assert call("Merging datasets with the same model and inferencer...") in logs_called
            # 验证runner被调用了一次，但参数应该是合并后的任务
            mock_runner.assert_called_once()

    @patch('ais_bench.benchmark.cli.workers.PARTITIONERS')
    @patch('ais_bench.benchmark.cli.workers.RUNNERS')
    @patch('ais_bench.benchmark.cli.workers.logger')
    def test_do_work_perf_mode(self, mock_logger, mock_runners, mock_partitioners):
        """测试do_work方法，性能模式的情况（应自动合并数据集）"""
        # 设置mock对象
        mock_partitioner = MagicMock()
        mock_partitioners.build.return_value = mock_partitioner
        mock_tasks = [MagicMock()]
        mock_partitioner.return_value = mock_tasks
        
        mock_runner = MagicMock()
        mock_runners.build.return_value = mock_runner
        
        # 创建测试配置 - 使用MockConfigDict
        cfg = MockConfigDict({
            'infer': {
                'partitioner': {},
                'runner': {}
            },
            'cli_args': MagicMock(merge_ds=False, mode='perf')
        })
        
        # 模拟_update_tasks_cfg方法
        with patch.object(self.infer_worker, '_update_tasks_cfg'):
            # 执行测试
            with patch.object(self.infer_worker, '_merge_datasets') as mock_merge:
                mock_merge.return_value = mock_tasks
                self.infer_worker.do_work(cfg)
                
                # 验证_merge_datasets被调用
                mock_merge.assert_called_once_with(mock_tasks)

    def test_merge_datasets(self):
        """测试_merge_datasets方法"""
        # 创建测试数据
        task1 = {
            'models': [{'abbr': 'model1'}],
            'datasets': [[{'type': 'dataset_type', 'infer_cfg': {'inferencer': 'inferencer_type'}}]]
        }
        task2 = {
            'models': [{'abbr': 'model1'}],
            'datasets': [[{'type': 'dataset_type', 'infer_cfg': {'inferencer': 'inferencer_type'}}]]
        }
        task3 = {
            'models': [{'abbr': 'model2'}],
            'datasets': [[{'type': 'dataset_type', 'infer_cfg': {'inferencer': 'inferencer_type'}}]]
        }
        
        # 执行测试
        result = self.infer_worker._merge_datasets([task1, task2, task3])
        
        # 验证结果
        assert len(result) == 2  # 应该合并为2个任务
        # 第一个任务应该包含合并后的数据集
        assert len(result[0]['datasets'][0]) == 2
        # 第二个任务应该保持不变
        assert len(result[1]['datasets'][0]) == 1

    def test_update_tasks_cfg_with_attack(self):
        """测试_update_tasks_cfg方法，有attack属性的情况"""
        # 创建测试数据
        task = MagicMock()
        task.datasets = [[MagicMock(abbr='test_dataset')]]
        tasks = [task]
        
        cfg = MagicMock()
        cfg.attack = MagicMock()
        
        # 执行测试
        self.infer_worker._update_tasks_cfg(tasks, cfg)
        
        # 验证结果
        assert cfg.attack.dataset == 'test_dataset'
        assert task.attack == cfg.attack

    def test_update_tasks_cfg_without_attack(self):
        """测试_update_tasks_cfg方法，没有attack属性的情况"""
        # 创建测试数据
        task = MagicMock()
        tasks = [task]
        
        cfg = MagicMock()
        # 删除attack属性
        if hasattr(cfg, 'attack'):
            delattr(cfg, 'attack')
        
        # 执行测试 - 不应抛出异常
        self.infer_worker._update_tasks_cfg(tasks, cfg)


class TestEval:
    def setup_method(self):
        """设置测试环境"""
        self.mock_args = MagicMock()
        self.mock_args.max_num_workers = 4
        self.mock_args.max_workers_per_gpu = 2
        self.mock_args.debug = False
        self.eval_worker = Eval(self.mock_args)

    @patch('ais_bench.benchmark.cli.workers.get_config_type')
    def test_update_cfg(self, mock_get_config_type):
        """测试update_cfg方法"""
        # 设置mock返回值
        mock_get_config_type.side_effect = ['MockNaivePartitioner', 'MockOpenICLEvalTask', 'MockLocalRunner']
        
        # 创建测试配置 - 使用MockConfigDict
        cli_args = MagicMock()
        cli_args.dump_eval_details = True
        cli_args.dump_extract_rate = True
        cli_args.debug = True
        
        cfg = MockConfigDict({
            'models': [{'abbr': 'test_model'}],
            'datasets': [{'abbr': 'test_dataset'}],
            'work_dir': '/test/workdir',
            'cli_args': cli_args
        })
        
        # 执行测试
        with patch('os.path.join', return_value='/test/workdir/results/'):
            result = self.eval_worker.update_cfg(cfg)
        
        # 验证结果
        assert result == cfg
        assert cfg['eval']['partitioner']['type'] == 'MockNaivePartitioner'
        assert cfg['eval']['runner']['type'] == 'MockLocalRunner'
        assert cfg['eval']['runner']['task']['type'] == 'MockOpenICLEvalTask'
        assert cfg['eval']['runner']['max_num_workers'] == 4
        assert cfg['eval']['runner']['max_workers_per_gpu'] == 2
        assert cfg['eval']['runner']['debug'] == True
        assert cfg['eval']['runner']['task']['dump_details'] == True
        assert cfg['eval']['runner']['task']['cal_extract_rate'] == True
        assert cfg['eval']['partitioner']['out_dir'] == '/test/workdir/results/'
        
        # 注意：fill_model_path_if_datasets_need是在_fill_dataset_configs中调用的，不是在Eval.update_cfg中
        # 所以这里不应该验证它被调用

    @patch('ais_bench.benchmark.cli.workers.PARTITIONERS')
    @patch('ais_bench.benchmark.cli.workers.RUNNERS')
    @patch('ais_bench.benchmark.cli.workers.logger')
    def test_do_work_normal_tasks(self, mock_logger, mock_runners, mock_partitioners):
        """测试do_work方法，普通任务列表的情况"""
        # 设置mock对象
        mock_partitioner = MagicMock()
        mock_partitioners.build.return_value = mock_partitioner
        mock_tasks = [MagicMock()]
        mock_partitioner.return_value = mock_tasks
        
        mock_runner = MagicMock()
        mock_runners.build.return_value = mock_runner
        
        # 创建测试配置 - 使用MockConfigDict
        cfg = MockConfigDict({
            'eval': {
                'partitioner': {},
                'runner': {}
            }
        })
        
        # 模拟_update_tasks_cfg方法
        with patch.object(self.eval_worker, '_update_tasks_cfg'):
            # 执行测试
            self.eval_worker.do_work(cfg)
            
            # 验证结果
            mock_partitioners.build.assert_called_once_with(cfg['eval']['partitioner'])
            mock_partitioner.assert_called_once_with(cfg)
            mock_runners.build.assert_called_once_with(cfg['eval']['runner'])
            mock_runner.assert_called_once_with(mock_tasks)

    @patch('ais_bench.benchmark.cli.workers.PARTITIONERS')
    @patch('ais_bench.benchmark.cli.workers.RUNNERS')
    @patch('ais_bench.benchmark.cli.workers.logger')
    def test_do_work_nested_tasks(self, mock_logger, mock_runners, mock_partitioners):
        """测试do_work方法，嵌套任务列表的情况（用于元评审）"""
        # 设置mock对象
        mock_partitioner = MagicMock()
        mock_partitioners.build.return_value = mock_partitioner
        # 创建嵌套任务列表
        mock_task_part1 = [MagicMock()]
        mock_task_part2 = [MagicMock()]
        mock_tasks = [mock_task_part1, mock_task_part2]
        mock_partitioner.return_value = mock_tasks
        
        mock_runner = MagicMock()
        mock_runners.build.return_value = mock_runner
        
        # 创建测试配置 - 使用MockConfigDict
        cfg = MockConfigDict({
            'eval': {
                'partitioner': {},
                'runner': {}
            }
        })
        
        # 模拟_update_tasks_cfg方法
        with patch.object(self.eval_worker, '_update_tasks_cfg'):
            # 执行测试
            self.eval_worker.do_work(cfg)
            
            # 验证结果 - runner应该被调用两次，分别处理每个任务部分
            assert mock_runner.call_count == 2
            mock_runner.assert_any_call(mock_task_part1)
            mock_runner.assert_any_call(mock_task_part2)

    def test_update_tasks_cfg(self):
        """测试_update_tasks_cfg方法（Eval中的实现为空）"""
        # 执行测试 - 不应抛出异常
        self.eval_worker._update_tasks_cfg([], MagicMock())


class TestAccViz:
    def setup_method(self):
        """设置测试环境"""
        self.mock_args = MagicMock()
        self.mock_args.cfg_time_str = '20240101_120000'
        self.acc_viz_worker = AccViz(self.mock_args)

    @patch('ais_bench.benchmark.cli.workers.get_config_type')
    def test_update_cfg_no_summarizer(self, mock_get_config_type):
        """测试update_cfg方法，没有summarizer配置的情况"""
        # 设置mock返回值
        mock_get_config_type.return_value = 'MockDefaultSummarizer'
        
        # 创建测试配置 - 使用MockConfigDict
        cfg = MockConfigDict({})
        
        # 执行测试
        result = self.acc_viz_worker.update_cfg(cfg)
        
        # 验证结果
        assert result == cfg
        assert cfg['summarizer']['type'] == 'MockDefaultSummarizer'
        assert 'attr' not in cfg['summarizer']

    @patch('ais_bench.benchmark.cli.workers.get_config_type')
    def test_update_cfg_with_attr(self, mock_get_config_type):
        """测试update_cfg方法，summarizer有attr属性的情况"""
        # 设置mock返回值
        mock_get_config_type.return_value = 'MockDefaultSummarizer'
        
        # 创建测试配置 - 使用MockConfigDict
        cfg = MockConfigDict({
            'summarizer': {
                'attr': 'accuracy'
            }
        })
        
        # 执行测试
        self.acc_viz_worker.update_cfg(cfg)
        
        # 验证结果
        assert 'attr' not in cfg['summarizer']

    @patch('ais_bench.benchmark.cli.workers.logger')
    @patch('ais_bench.benchmark.cli.workers.build_from_cfg')
    def test_do_work_normal(self, mock_build_from_cfg, mock_logger):
        """测试do_work方法，普通摘要器的情况"""
        # 设置mock对象
        mock_summarizer = MagicMock()
        mock_build_from_cfg.return_value = mock_summarizer
        
        # 创建测试配置 - 使用MockConfigDict
        cfg = MockConfigDict({
            'summarizer': {}
        })
        
        # 执行测试
        self.acc_viz_worker.do_work(cfg)
        
        # 验证结果
        mock_build_from_cfg.assert_called_once_with({'config': cfg})
        mock_summarizer.summarize.assert_called_once_with(time_str='20240101_120000')

    @patch('ais_bench.benchmark.cli.workers.logger')
    @patch('ais_bench.benchmark.cli.workers.build_from_cfg')
    def test_do_work_subjective(self, mock_build_from_cfg, mock_logger):
        """测试do_work方法，主观摘要器的情况"""
        # 设置mock对象
        mock_summarizer1 = MagicMock()
        mock_summarizer1.summarize.return_value = {'score': 0.9}
        mock_summarizer2 = MagicMock()
        mock_summarizer3 = MagicMock()
        # 使用列表而不是生成器，避免StopIteration错误
        mock_build_from_cfg.side_effect = [mock_summarizer1, mock_summarizer1, mock_summarizer3]
        
        # 创建测试配置 - 使用MockConfigDict
        cfg = MockConfigDict({
            'summarizer': {
                'function': 'subjective_summary'
            },
            'datasets': [
                {'abbr': 'dataset1_1', 'summarizer': {'type': 'summarizer_type1'}},
                {'abbr': 'dataset1_2', 'summarizer': {'type': 'summarizer_type1'}},
                {'abbr': 'dataset2_1', 'summarizer': {'type': 'summarizer_type2'}}
            ]
        })
        
        # 执行测试
        self.acc_viz_worker.do_work(cfg)
        
        # 验证结果 - 应该构建多个摘要器
        assert mock_build_from_cfg.call_count == 3
        # 验证主摘要器被调用时传入了主观分数
        mock_summarizer3.summarize.assert_called_once()
        call_args = mock_summarizer3.summarize.call_args
        assert call_args[1]['time_str'] == '20240101_120000'
        assert len(call_args[1]['subjective_scores']) == 2


class TestPerfViz:
    def setup_method(self):
        """设置测试环境"""
        self.perf_viz_worker = PerfViz(MagicMock())

    @patch('ais_bench.benchmark.cli.workers.get_config_type')
    def test_update_cfg_complete(self, mock_get_config_type):
        """测试update_cfg方法，完整配置的情况"""
        # 设置mock返回值
        mock_get_config_type.side_effect = ['MockDefaultPerfSummarizer', 'MockDefaultPerfMetricCalculator']
        
        # 创建测试配置 - 使用MockConfigDict
        cfg = MockConfigDict({
            'summarizer': {
                'attr': 'performance',
                'dataset_abbrs': ['dataset1'],
                'summary_groups': ['group1'],
                'prompt_db': 'db_path'
            }
        })
        
        # 执行测试
        result = self.perf_viz_worker.update_cfg(cfg)
        
        # 验证结果
        assert result == cfg
        assert cfg['summarizer']['type'] == 'MockDefaultPerfSummarizer'
        assert cfg['summarizer']['calculator']['type'] == 'MockDefaultPerfMetricCalculator'
        assert 'attr' not in cfg['summarizer']
        assert 'dataset_abbrs' not in cfg['summarizer']
        assert 'summary_groups' not in cfg['summarizer']
        assert 'prompt_db' not in cfg['summarizer']

    @patch('ais_bench.benchmark.cli.workers.get_config_type')
    def test_update_cfg_minimal(self, mock_get_config_type):
        """测试update_cfg方法，最小配置的情况"""
        # 设置mock返回值
        mock_get_config_type.side_effect = ['MockDefaultPerfSummarizer', 'MockDefaultPerfMetricCalculator']
        
        # 创建测试配置 - 使用MockConfigDict
        cfg = MockConfigDict({})
        
        # 执行测试
        self.perf_viz_worker.update_cfg(cfg)
        
        # 验证结果
        assert cfg['summarizer']['type'] == 'MockDefaultPerfSummarizer'
        assert cfg['summarizer']['calculator']['type'] == 'MockDefaultPerfMetricCalculator'

    @patch('ais_bench.benchmark.cli.workers.logger')
    @patch('ais_bench.benchmark.cli.workers.build_from_cfg')
    def test_do_work(self, mock_build_from_cfg, mock_logger):
        """测试do_work方法"""
        # 设置mock对象
        mock_summarizer = MagicMock()
        mock_build_from_cfg.return_value = mock_summarizer
        
        # 创建测试配置 - 使用MockConfigDict
        cfg = MockConfigDict({
            'summarizer': {}
        })
        
        # 执行测试
        self.perf_viz_worker.do_work(cfg)
        
        # 验证结果
        mock_build_from_cfg.assert_called_once_with({'config': cfg})
        mock_summarizer.summarize.assert_called_once()
        mock_logger.info.assert_called_once_with("Summarizing performance results...")


class TestWorkFlowExecutor:
    def test_init(self):
        """测试WorkFlowExecutor初始化"""
        mock_cfg = MagicMock()
        mock_workflow = [MagicMock(), MagicMock()]
        executor = WorkFlowExecutor(mock_cfg, mock_workflow)
        
        assert executor.cfg == mock_cfg
        assert executor.workflow == mock_workflow

    def test_execute(self):
        """测试execute方法"""
        mock_cfg = MagicMock()
        
        # 创建两个mock worker
        mock_worker1 = MagicMock()
        mock_worker2 = MagicMock()
        mock_workflow = [mock_worker1, mock_worker2]
        
        # 创建执行器
        executor = WorkFlowExecutor(mock_cfg, mock_workflow)
        
        # 执行测试
        executor.execute()
        
        # 验证结果 - 每个worker的do_work方法都应该被调用
        mock_worker1.do_work.assert_called_once_with(mock_cfg)
        mock_worker2.do_work.assert_called_once_with(mock_cfg)


def test_work_flow_dict():
    """测试WORK_FLOW字典的内容"""
    # 验证WORK_FLOW包含所有必要的工作流
    assert 'all' in WORK_FLOW
    assert 'infer' in WORK_FLOW
    assert 'eval' in WORK_FLOW
    assert 'viz' in WORK_FLOW
    assert 'perf' in WORK_FLOW
    assert 'perf_viz' in WORK_FLOW
    
    # 验证工作流内容正确
    assert Infer in WORK_FLOW['all']
    assert Eval in WORK_FLOW['all']
    assert AccViz in WORK_FLOW['all']
    
    assert Infer in WORK_FLOW['infer']
    
    assert Eval in WORK_FLOW['eval']
    assert AccViz in WORK_FLOW['eval']
    
    assert AccViz in WORK_FLOW['viz']
    
    assert Infer in WORK_FLOW['perf']
    assert PerfViz in WORK_FLOW['perf']
    
    assert PerfViz in WORK_FLOW['perf_viz']