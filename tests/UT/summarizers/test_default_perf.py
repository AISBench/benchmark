import unittest
import json
import os
import tempfile
import time
from unittest.mock import patch, MagicMock, mock_open, call
import multiprocessing
import numpy as np

from ais_bench.benchmark.summarizers.default_perf import (
    DefaultPerfSummarizer,
    model_abbr_from_cfg_used_in_summarizer
)
from ais_bench.benchmark.utils.logging.exceptions import AISBenchDataContentError, FileMatchError
from mmengine import ConfigDict


class TestModelAbbrFromCfgUsedInSummarizer(unittest.TestCase):
    def test_with_summarizer_abbr(self):
        """测试当模型配置中包含summarizer_abbr时的情况"""
        model = {"summarizer_abbr": "custom_model"}
        result = model_abbr_from_cfg_used_in_summarizer(model)
        self.assertEqual(result, "custom_model")

    @patch('ais_bench.benchmark.summarizers.default_perf.model_abbr_from_cfg')
    def test_without_summarizer_abbr(self, mock_model_abbr_from_cfg):
        """测试当模型配置中不包含summarizer_abbr时的情况"""
        mock_model_abbr_from_cfg.return_value = "default_model"
        model = {"name": "test_model"}
        result = model_abbr_from_cfg_used_in_summarizer(model)
        self.assertEqual(result, "default_model")
        mock_model_abbr_from_cfg.assert_called_once_with(model)


class TestDefaultPerfSummarizer(unittest.TestCase):
    def setUp(self):
        """设置测试环境"""
        self.model_cfg = {
            "name": "test_model",
            "type": "TestModelType",  # 添加type字段
            "path": "test/path/model",  # 添加path字段
            "attr": "service",
            "batch_size": 4
        }
        self.dataset_cfg = {
            "type": "TestDataset",
            "abbr": "test_ds",
            "infer_cfg": {"inferencer": "test_inferencer"}
        }
        self.config = ConfigDict({
            "models": [self.model_cfg],
            "datasets": [self.dataset_cfg],
            "work_dir": "/tmp/test_work_dir",
            "cli_args": {"merge_ds": False}
        })
        self.calculator_cfg = ConfigDict({"type": "TestCalculator"})

    @patch('ais_bench.benchmark.summarizers.default_perf.AISLogger')
    def test_init(self, mock_ais_logger):
        """测试DefaultPerfSummarizer的初始化"""
        summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)

        self.assertEqual(summarizer.cfg, self.config)
        self.assertEqual(summarizer.model_cfgs, [self.model_cfg])
        self.assertEqual(summarizer.dataset_cfgs, [self.dataset_cfg])
        self.assertEqual(summarizer.calculator_conf, self.calculator_cfg)
        self.assertEqual(summarizer.work_dir, "/tmp/test_work_dir")
        mock_ais_logger.assert_called_once()

    def test_get_dataset_abbr_single(self):
        """测试获取单个数据集的缩写"""
        summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
        result = summarizer._get_dataset_abbr([self.dataset_cfg])
        self.assertEqual(result, "test_ds")

    def test_get_dataset_abbr_multiple(self):
        """测试获取多个数据集的缩写"""
        dataset_cfg2 = {
            "type": "TestDataset",
            "abbr": "test_ds2",
            "infer_cfg": {"inferencer": "test_inferencer"}
        }
        summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
        result = summarizer._get_dataset_abbr([self.dataset_cfg, dataset_cfg2])
        self.assertEqual(result, "testdataset")

    @patch('ais_bench.benchmark.summarizers.default_perf.build_model_from_cfg')
    @patch('ais_bench.benchmark.summarizers.default_perf.init_db')
    @patch('ais_bench.benchmark.summarizers.default_perf.load_all_numpy_from_db')
    @patch('ais_bench.benchmark.summarizers.default_perf.is_mm_prompt')
    @patch('ais_bench.benchmark.summarizers.default_perf.AISTokenizer')
    @patch('ais_bench.benchmark.summarizers.default_perf.load_tokenizer')
    def test_calc_perf_data_success(self, mock_load_tokenizer, mock_aistokenizer, mock_is_mm_prompt, mock_load_all_numpy_from_db,
                                 mock_init_db, mock_build_model_from_cfg):
        """测试成功计算性能数据的情况"""
        # 设置mock
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda x: list(range(len(x)))
        mock_build_model_from_cfg.return_value = mock_model
        
        # Mock AISTokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda x: list(range(len(x)))
        mock_aistokenizer.return_value = mock_tokenizer
        
        # Mock load_tokenizer
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_conn = MagicMock()
        mock_init_db.return_value = mock_conn
        mock_load_all_numpy_from_db.return_value = {}
        mock_is_mm_prompt.return_value = False

        # 准备测试数据
        manager_list = []
        perf_data = {
            "success": True,
            "input": "test input",
            "prediction": "test output",
            "output_tokens": 0,
            "time_points": [0, 1, 2, 3],
            "db_name": "test_db"
        }

        summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
        summarizer._calc_perf_data(manager_list, self.model_cfg, "/tmp/db", [perf_data])

        # 验证结果
        self.assertEqual(len(manager_list), 1)
        result = manager_list[0]
        self.assertEqual(result["start_time"], 0)
        self.assertEqual(result["end_time"], 3)
        self.assertEqual(result["latency"], 3)
        self.assertEqual(result["ttft"], 1)
        mock_conn.close.assert_called_once()

    @patch('ais_bench.benchmark.summarizers.default_perf.build_model_from_cfg')
    @patch('ais_bench.benchmark.summarizers.default_perf.AISTokenizer')
    @patch('ais_bench.benchmark.summarizers.default_perf.load_tokenizer')
    def test_calc_perf_data_failure(self, mock_load_tokenizer, mock_aistokenizer, mock_build_model_from_cfg):
        """测试计算性能数据失败的情况"""
        manager_list = []
        perf_data = {"success": False}

        summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
        summarizer._calc_perf_data(manager_list, self.model_cfg, "/tmp/db", [perf_data])

        self.assertEqual(len(manager_list), 1)
        self.assertEqual(manager_list[0], {"success": False})

    @patch('ais_bench.benchmark.summarizers.default_perf.build_model_from_cfg')
    @patch('ais_bench.benchmark.summarizers.default_perf.init_db')
    @patch('ais_bench.benchmark.summarizers.default_perf.load_all_numpy_from_db')
    @patch('ais_bench.benchmark.summarizers.default_perf.AISTokenizer')
    @patch('ais_bench.benchmark.summarizers.default_perf.load_tokenizer')
    def test_calc_perf_data_time_points_none(self, mock_load_tokenizer, mock_aistokenizer, mock_load_all_numpy_from_db, mock_init_db, mock_build_model_from_cfg):
        """测试time_points为None的情况"""
        # 设置mock
        mock_model = MagicMock()
        mock_build_model_from_cfg.return_value = mock_model
        mock_conn = MagicMock()
        mock_init_db.return_value = mock_conn
        
        # Mock AISTokenizer和load_tokenizer
        mock_load_tokenizer.return_value = MagicMock()
        mock_aistokenizer.return_value = MagicMock()

        manager_list = []
        perf_data = {
            "success": True,
            "input": "test",
            "prediction": "test",
            "time_points": None,
            "db_name": "test_db"
        }

        summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
        summarizer._calc_perf_data(manager_list, self.model_cfg, "/tmp/db", [perf_data])

        self.assertEqual(len(manager_list), 1)
        self.assertEqual(manager_list[0], {"success": False})

    @patch('os.path.exists')
    def test_load_tmp_result_no_dir(self, mock_exists):
        """测试临时目录不存在的情况"""
        mock_exists.return_value = False

        summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
        result = summarizer._load_tmp_result("test_model", ["test_ds"])

        self.assertEqual(result, {})
        mock_exists.assert_called_once_with("/tmp/test_work_dir/performances/test_model/tmp")

    @patch('os.path.exists')
    @patch('os.listdir')
    @patch('builtins.open', new_callable=mock_open, read_data='{"data_abbr": "test_ds", "db_name": "test_db"}\n')
    def test_load_tmp_result_with_files(self, mock_file, mock_listdir, mock_exists):
        """测试加载临时结果文件的情况"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["test.jsonl"]

        summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
        result = summarizer._load_tmp_result("test_model", ["test_ds"])

        self.assertIn("test_db", result)
        self.assertEqual(len(result["test_db"]), 1)

    @patch('os.path.exists')
    def test_load_details_perf_data_no_files(self, mock_exists):
        """测试没有找到性能数据文件的情况"""
        mock_exists.return_value = False

        summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)

        with self.assertRaises(FileMatchError):
            summarizer._load_details_perf_data(self.model_cfg, [self.dataset_cfg])

    def test_load_csv_to_table(self):
        """测试加载CSV文件到表格"""
        csv_content = "col1,col2\nvalue1,value2\nvalue3,value4\n"

        with patch('builtins.open', new_callable=mock_open, read_data=csv_content):
            summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
            result = summarizer._load_csv_to_table("dummy.csv")

            self.assertEqual(len(result), 3)
            self.assertEqual(result[0], ["col1", "col2"])
            self.assertEqual(result[1], ["value1", "value2"])

    def test_load_json_to_table(self):
        """测试加载JSON文件到表格"""
        json_data = {
            "metric1": {"stage1": 0.5, "stage2": 0.6},
            "metric2": {"stage1": 0.7}
        }

        with patch('builtins.open', new_callable=mock_open, read_data=json.dumps(json_data)):
            summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
            result = summarizer._load_json_to_table("dummy.json")

            self.assertEqual(len(result), 4)  # 1 header + 3 data rows
            self.assertEqual(result[0], ["Common Metric", "Stage", "Value"])
            self.assertEqual(result[1], ["metric1", "stage1", 0.5])

    @patch('ais_bench.benchmark.summarizers.default_perf.DefaultPerfSummarizer._get_dataset_abbr')
    @patch('os.path.exists')
    def test_pick_up_results(self, mock_exists, mock_get_dataset_abbr):
        """测试获取结果"""
        mock_get_dataset_abbr.return_value = "test_ds"
        mock_exists.side_effect = lambda x: "csv" in x  # 只有CSV文件存在

        # 模拟CSV文件内容
        with patch('builtins.open', new_callable=mock_open, read_data="col1,col2\nval1,val2\n") as mock_file:
            summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
            with patch.object(summarizer, 'model_abbrs', ["test_model"]), \
                 patch.object(summarizer, 'dataset_groups', {"key": [self.dataset_cfg]}):
                result = summarizer._pick_up_results()

                self.assertIn("test_model/test_ds", result)
                self.assertEqual(len(result["test_model/test_ds"]), 1)

    @patch('tabulate.tabulate')
    def test_output_to_screen(self, mock_tabulate):
        """测试输出到屏幕"""
        mock_tabulate.return_value = "formatted table"

        summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
        summarizer.logger = MagicMock()

        tables_dict = {
            "model1/dataset1": [[["header1", "header2"], ["val1", "val2"]]]
        }

        with patch('builtins.print') as mock_print:
            summarizer._output_to_screen(tables_dict)

            mock_print.assert_called_with("formatted table")
            summarizer.logger.info.assert_called()

    @patch('ais_bench.benchmark.summarizers.default_perf.DefaultPerfSummarizer._load_details_perf_data')
    @patch('ais_bench.benchmark.summarizers.default_perf.DefaultPerfSummarizer._get_dataset_abbr')
    @patch('ais_bench.benchmark.summarizers.default_perf.plot_sorted_request_timelines')
    @patch('ais_bench.benchmark.summarizers.default_perf.build_perf_metric_calculator_from_cfg')
    @patch('ais_bench.benchmark.summarizers.default_perf.DefaultPerfSummarizer._dump_calculated_perf_data')
    @patch('ais_bench.benchmark.summarizers.default_perf.DefaultPerfSummarizer._pick_up_results')
    @patch('ais_bench.benchmark.summarizers.default_perf.DefaultPerfSummarizer._output_to_screen')
    def test_summarize(self, mock_output_to_screen, mock_pick_up_results,
                      mock_dump_calculated_perf_data, mock_build_perf_metric_calculator_from_cfg,
                      mock_plot_sorted_request_timelines, mock_get_dataset_abbr, mock_load_details_perf_data):
        """测试summarize方法"""
        # 设置mock
        mock_get_dataset_abbr.return_value = "test_ds"
        mock_load_details_perf_data.return_value = {
            "start_time": [0, 1],
            "end_time": [2, 3],
            "ttft": [0.5, 0.6]
        }
        mock_plot_sorted_request_timelines.return_value = True

        mock_calculator = MagicMock()
        mock_build_perf_metric_calculator_from_cfg.return_value = mock_calculator
        mock_pick_up_results.return_value = {"test_model/test_ds": []}

        summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
        summarizer.logger = MagicMock()

        # 执行测试
        summarizer.summarize()

        # 验证调用
        mock_load_details_perf_data.assert_called_once()
        mock_plot_sorted_request_timelines.assert_called_once()
        mock_dump_calculated_perf_data.assert_called_once()
        mock_pick_up_results.assert_called_once()
        mock_output_to_screen.assert_called_once()


    @patch('ais_bench.benchmark.summarizers.default_perf.dump_results_dict')
    def test_dump_calculated_perf_data(self, mock_dump_results_dict):
        """测试_dump_calculated_perf_data方法"""
        # 创建模拟计算器对象
        mock_calculator1 = MagicMock()
        mock_calculator1.get_common_res.return_value = {"metric": {"stage": 0.5}}
        mock_calculator2 = MagicMock()
        mock_calculator2.get_common_res.return_value = {"metric": {"stage": 0.6}}

        # 设置summarizers的calculators属性
        summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
        summarizer.calculators = {
            "model1": {
                "dataset1": mock_calculator1
            },
            "model2": {
                "dataset2": mock_calculator2
            }
        }

        # 执行测试
        summarizer._dump_calculated_perf_data()

        # 验证调用
        mock_calculator1.calculate.assert_called_once()
        mock_calculator2.calculate.assert_called_once()
        mock_calculator1.save_performance.assert_called_with("/tmp/test_work_dir/performances/model1/dataset1.csv")
        mock_calculator2.save_performance.assert_called_with("/tmp/test_work_dir/performances/model2/dataset2.csv")
        mock_dump_results_dict.assert_any_call({"metric": {"stage": 0.5}}, "/tmp/test_work_dir/performances/model1/dataset1.json")
        mock_dump_results_dict.assert_any_call({"metric": {"stage": 0.6}}, "/tmp/test_work_dir/performances/model2/dataset2.json")

    def test_tqdm_monitor(self):
        """测试tqdm_monitor方法"""
        manager_list = [1, 2, 3]  # 模拟已经有3个元素的列表
        event = multiprocessing.Event()

        # 使用patch替代tqdm以避免实际显示进度条
        with patch('ais_bench.benchmark.summarizers.default_perf.tqdm') as mock_tqdm:
            mock_pbar = MagicMock()
            mock_tqdm.return_value.__enter__.return_value = mock_pbar

            # 创建一个线程来运行监控函数，这样我们可以在适当的时候设置event
            import threading
            monitor_thread = threading.Thread(
                target=DefaultPerfSummarizer(self.config, self.calculator_cfg).tqdm_monitor,
                args=(5, manager_list, event)
            )
            monitor_thread.daemon = True
            monitor_thread.start()

            # 等待一小段时间让监控线程运行
            time.sleep(0.2)

            # 设置event来结束监控
            event.set()
            monitor_thread.join(timeout=1.0)

            # 验证tqdm被正确调用并更新了进度
            mock_tqdm.assert_called_once_with(total=5, desc="Calculating performance details")
            self.assertTrue(mock_pbar.n >= 3)  # 进度条至少更新到了3
            mock_pbar.refresh.assert_called()

    @patch('ais_bench.benchmark.summarizers.default_perf.is_mm_prompt')
    @patch('ais_bench.benchmark.summarizers.default_perf.build_model_from_cfg')
    @patch('ais_bench.benchmark.summarizers.default_perf.init_db')
    @patch('ais_bench.benchmark.summarizers.default_perf.load_all_numpy_from_db')
    @patch('ais_bench.benchmark.summarizers.default_perf.AISTokenizer')
    @patch('ais_bench.benchmark.summarizers.default_perf.load_tokenizer')
    def test_calc_perf_data_multimodal(self, mock_load_tokenizer, mock_aistokenizer, mock_load_all_numpy_from_db, mock_init_db,
                                     mock_build_model_from_cfg, mock_is_mm_prompt):
        """测试计算多模态提示的性能数据"""
        # 设置mock
        mock_model = MagicMock()
        mock_build_model_from_cfg.return_value = mock_model
        mock_conn = MagicMock()
        mock_init_db.return_value = mock_conn
        mock_load_all_numpy_from_db.return_value = {}
        
        # Mock AISTokenizer和load_tokenizer
        mock_load_tokenizer.return_value = MagicMock()
        mock_aistokenizer.return_value = MagicMock()
        mock_is_mm_prompt.return_value = True  # 模拟多模态提示

        manager_list = []
        perf_data = {
            "success": True,
            "input": "<image>test</image>",  # 模拟多模态输入
            "prediction": "test output",
            "output_tokens": 5,
            "time_points": [0, 1, 2, 3],
            "db_name": "test_db"
        }

        summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
        summarizer._calc_perf_data(manager_list, self.model_cfg, "/tmp/db", [perf_data])

        # 验证结果
        self.assertEqual(len(manager_list), 1)
        result = manager_list[0]
        self.assertEqual(result["input_tokens"], 0)  # 多模态输入应该设置为0

    @patch('ais_bench.benchmark.summarizers.default_perf.build_model_from_cfg')
    @patch('ais_bench.benchmark.summarizers.default_perf.init_db')
    @patch('ais_bench.benchmark.summarizers.default_perf.load_all_numpy_from_db')
    @patch('ais_bench.benchmark.summarizers.default_perf.AISTokenizer')
    @patch('ais_bench.benchmark.summarizers.default_perf.load_tokenizer')
    def test_calc_perf_data_recursive_update(self, mock_load_tokenizer, mock_aistokenizer, mock_load_all_numpy_from_db, mock_init_db,
                                          mock_build_model_from_cfg):
        """测试_calc_perf_data方法中的递归更新功能"""
        # 设置mock
        mock_model = MagicMock()
        mock_model.encode.return_value = [1, 2, 3]
        mock_build_model_from_cfg.return_value = mock_model
        mock_conn = MagicMock()
        mock_init_db.return_value = mock_conn
        
        # Mock AISTokenizer和load_tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_aistokenizer.return_value = mock_tokenizer

        # 模拟numpy数据
        mock_numpy_data = np.array([1.0, 2.0, 3.0])
        mock_load_all_numpy_from_db.return_value = {"numpy_1": mock_numpy_data}

        manager_list = []
        perf_data = {
            "success": True,
            "input": "test",
            "prediction": "test",
            "output_tokens": 0,
            "time_points": [0, 1],
            "db_name": "test_db",
            "nested_data": {
                "__db_ref__": "numpy_1"
            }
        }

        summarizer = DefaultPerfSummarizer(self.config, self.calculator_cfg)
        with patch('ais_bench.benchmark.summarizers.default_perf.is_mm_prompt', return_value=False):
            summarizer._calc_perf_data(manager_list, self.model_cfg, "/tmp/db", [perf_data])

        # 验证结果
        self.assertEqual(len(manager_list), 1)
        result = manager_list[0]
        # 验证嵌套数据被正确替换为numpy数组
        np.testing.assert_array_equal(result["nested_data"], mock_numpy_data)

    @patch('ais_bench.benchmark.summarizers.default_perf.DefaultPerfSummarizer._load_details_perf_data')
    def test_summarize_non_service_model(self, mock_load_details_perf_data):
        """测试summarize方法对非service模型的处理"""
        # 修改model_cfg，使其不是service类型
        non_service_model_cfg = self.model_cfg.copy()
        non_service_model_cfg["attr"] = "local"

        # 创建一个新的配置
        non_service_config = self.config.copy()
        non_service_config["models"] = [non_service_model_cfg]

        summarizer = DefaultPerfSummarizer(non_service_config, self.calculator_cfg)
        summarizer.summarize()

        # 验证_load_details_perf_data没有被调用
        mock_load_details_perf_data.assert_not_called()

if __name__ == '__main__':
    unittest.main()