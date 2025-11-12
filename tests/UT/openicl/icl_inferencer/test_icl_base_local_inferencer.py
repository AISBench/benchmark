import unittest
from unittest import mock

from ais_bench.benchmark.openicl.icl_inferencer.icl_base_local_inferencer import BaseLocalInferencer
from ais_bench.benchmark.openicl.icl_retriever.icl_base_retriever import BaseRetriever
from ais_bench.benchmark.utils.logging.exceptions import AISBenchImplementationError, ParameterValueError


class DummyInf(BaseLocalInferencer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_every = 1  # BaseLocalInferencer.inference uses self.save_every
    
    def get_data_list(self, retriever):
        return [
            {"index": 0, "prompt": "p0", "data_abbr": "d", "max_out_len": 8},
            {"index": 1, "prompt": "p1", "data_abbr": "d", "max_out_len": 8},
        ]


class DummyDataset:
    def __init__(self):
        self.reader = type("R", (), {"output_column": "label"})()
        self.train = None
        self.test = None
        self.abbr = "dummy"


class DummyRetriever(BaseRetriever):
    def __init__(self, dataset):
        super().__init__(dataset)
    def retrieve(self):
        return [[0], [1]]
    def generate_ice(self, idx_list):
        return "ICE"
    def generate_prompt_for_generate_task(self, idx, ice, gen_field_replace_token=""):
        return f"P{idx}|{ice}"
    def get_gold_ans(self):
        return ["g0", "g1"]


class TestBaseLocalInferencer(unittest.TestCase):
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_get_dataloader_and_batch_inference_abstract(self, m_abbr, m_build):
        """测试BaseLocalInferencer的get_dataloader方法和batch_inference抽象方法"""
        m_build.return_value = object()
        inf = DummyInf(model_cfg={}, batch_size=2)
        dl = inf.get_dataloader([
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ], batch_size=2)
        batch = next(iter(dl))
        self.assertEqual(batch["a"], [1, 3])
        with self.assertRaises(AISBenchImplementationError):
            BaseLocalInferencer(model_cfg={}).batch_inference({})

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_inference_invalid_retriever_type(self, m_abbr, m_build):
        """测试BaseLocalInferencer在retriever类型无效时抛出ParameterValueError"""
        m_build.return_value = object()
        inf = DummyInf(model_cfg={}, batch_size=1)
        inf.output_handler.run_cache_consumer = mock.Mock()
        inf.output_handler.stop_cache_consumer = mock.Mock()
        
        with self.assertRaises(ParameterValueError):
            inf.inference(retriever=object(), output_json_filepath="/tmp")

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_inference_with_list_retrievers(self, m_abbr, m_build):
        """测试BaseLocalInferencer使用retriever列表进行推理"""
        from ais_bench.benchmark.openicl.icl_retriever.icl_base_retriever import BaseRetriever
        
        m_build.return_value = object()
        inf = DummyInf(model_cfg={}, batch_size=1)
        inf.output_handler.run_cache_consumer = mock.Mock()
        inf.output_handler.stop_cache_consumer = mock.Mock()
        
        ret1 = mock.Mock(spec=BaseRetriever)
        ret2 = mock.Mock(spec=BaseRetriever)
        ret1.dataset = DummyDataset()
        ret2.dataset = DummyDataset()
        
        inf.get_data_list = mock.Mock(side_effect=[
            [{"index": 0, "prompt": "p0", "data_abbr": "d", "max_out_len": 8}],
            [{"index": 1, "prompt": "p1", "data_abbr": "d", "max_out_len": 8}],
        ])
        
        inf.batch_inference = mock.Mock()
        inf.inference(retriever=[ret1, ret2], output_json_filepath="/tmp")
        self.assertEqual(inf.get_data_list.call_count, 2)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_inference_with_single_retriever(self, m_abbr, m_build):
        """测试BaseLocalInferencer使用单个retriever进行推理"""
        from ais_bench.benchmark.openicl.icl_retriever.icl_base_retriever import BaseRetriever
        
        m_build.return_value = object()
        inf = DummyInf(model_cfg={}, batch_size=1)
        inf.output_handler.run_cache_consumer = mock.Mock()
        inf.output_handler.stop_cache_consumer = mock.Mock()
        
        ret = mock.Mock(spec=BaseRetriever)
        ret.dataset = DummyDataset()
        
        inf.get_data_list = mock.Mock(return_value=[
            {"index": 0, "prompt": "p0", "data_abbr": "d", "max_out_len": 8}
        ])
        
        inf.batch_inference = mock.Mock()
        inf.inference(retriever=ret, output_json_filepath="/tmp")
        inf.get_data_list.assert_called_once_with(ret)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_inference_with_task_state_manager(self, m_abbr, m_build):
        """测试BaseLocalInferencer在推理时使用task_state_manager更新任务状态"""
        m_build.return_value = object()
        inf = DummyInf(model_cfg={}, batch_size=1)
        inf.output_handler.run_cache_consumer = mock.Mock()
        inf.output_handler.stop_cache_consumer = mock.Mock()
        inf.is_main_process = True
        
        task_mgr = mock.Mock()
        inf.set_task_state_manager(task_mgr)
        
        inf.batch_inference = mock.Mock()
        inf.inference(retriever=DummyRetriever(DummyDataset()), output_json_filepath="/tmp")
        
        self.assertTrue(task_mgr.update_task_state.called)


if __name__ == '__main__':
    unittest.main()
