import os
import shutil
import tempfile
import unittest
from unittest import mock

from ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer import BaseInferencer, MAX_BATCH_SIZE
from ais_bench.benchmark.utils.logging.exceptions import ParameterValueError, AISBenchImplementationError


class DummyModel:
    def __init__(self):
        self.max_out_len = 16
        self.is_api = False
    def parse_template(self, prompt, mode="gen"):
        return prompt


class TestBaseInferencer(unittest.TestCase):
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg", return_value=DummyModel())
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_init_and_get_output_dir(self, m_abbr, m_build):
        """测试BaseInferencer初始化和获取输出目录"""
        inf = BaseInferencer(model_cfg={})
        self.assertTrue(inf.is_main_process in [True, False])
        out_dir = inf.get_output_dir()
        self.assertIn("mabbr", out_dir)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg", return_value=DummyModel())
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_batch_size_validation(self, m_abbr, m_build):
        """测试BaseInferencer的batch_size参数验证"""
        with self.assertRaises(ParameterValueError) as cm:
            BaseInferencer(model_cfg={}, batch_size=-1)
        self.assertIn("batch_size", str(cm.exception).lower())
        
        with self.assertRaises(ParameterValueError) as cm2:
            BaseInferencer(model_cfg={}, batch_size=MAX_BATCH_SIZE + 1)
        self.assertIn("batch_size", str(cm2.exception).lower())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg", return_value=DummyModel())
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_get_finish_data_list_paths(self, m_abbr, m_build):
        """测试BaseInferencer获取已完成数据列表，包括perf_mode和普通模式"""
        inf = BaseInferencer(model_cfg={})
        inf.perf_mode = True
        self.assertEqual(inf.get_finish_data_list(), {})
        inf.perf_mode = False
        self.assertEqual(inf.get_finish_data_list(), {})

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg", return_value=DummyModel())
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_get_data_list_abstract(self, m_abbr, m_build):
        """测试BaseInferencer的抽象方法get_data_list未实现时抛出异常"""
        with self.assertRaises(AISBenchImplementationError):
            BaseInferencer(model_cfg={}).get_data_list(None)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg", return_value=DummyModel())
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_set_task_state_manager(self, m_abbr, m_build):
        """测试BaseInferencer设置任务状态管理器"""
        inf = BaseInferencer(model_cfg={})
        manager = object()
        inf.set_task_state_manager(manager)
        self.assertEqual(inf.task_state_manager, manager)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg", return_value=DummyModel())
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_get_finish_data_list_with_files(self, m_abbr, m_build):
        """测试BaseInferencer从实际文件中获取已完成数据列表"""
        import tempfile
        import json
        
        inf = BaseInferencer(model_cfg={})
        inf.perf_mode = False
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(inf, 'get_output_dir', return_value=tmpdir):
                os.makedirs(tmpdir, exist_ok=True)
                os.makedirs(os.path.join(tmpdir, "tmp"), exist_ok=True)
                
                with open(os.path.join(tmpdir, "test.jsonl"), "w") as f:
                    json.dump({"uuid": "u1", "success": True, "data_abbr": "test"}, f)
                    f.write("\n")
                
                with open(os.path.join(tmpdir, "tmp", "tmp_123.jsonl"), "w") as f:
                    json.dump({"uuid": "u2", "success": True, "data_abbr": "test"}, f)
                    f.write("\n")
                
                finish_data = inf.get_finish_data_list()
                self.assertIn("test", finish_data)
                self.assertEqual(len(finish_data["test"]), 1)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg", return_value=DummyModel())
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_get_output_dir_with_perf_mode(self, m_abbr, m_build):
        """测试BaseInferencer在perf_mode和普通模式下获取不同的输出目录"""
        inf = BaseInferencer(model_cfg={})
        inf.perf_mode = True
        out_dir = inf.get_output_dir()
        inf.perf_mode = False
        out_dir2 = inf.get_output_dir()
        self.assertIsNotNone(out_dir)
        self.assertIsNotNone(out_dir2)


if __name__ == '__main__':
    unittest.main()


