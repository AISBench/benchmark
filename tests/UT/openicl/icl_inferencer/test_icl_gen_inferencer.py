import unittest
from unittest import mock
import asyncio

from datasets import Dataset

from ais_bench.benchmark.openicl.icl_inferencer.icl_gen_inferencer import GenInferencer
from ais_bench.benchmark.models.output import RequestOutput


class DummyDataset:
    def __init__(self):
        self.reader = type("R", (), {"output_column": "label", "get_max_out_len": lambda self=None: [5, 6]})()
        self.train = Dataset.from_dict({"text": ["t0", "t1"], "label": [0, 1]})
        self.test = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
        self.abbr = "abbrd"


class DummyRetriever:
    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset_reader = dataset.reader
    def retrieve(self):
        return [[0], [1]]
    def generate_ice(self, idx_list):
        return "ICE"
    def generate_prompt_for_generate_task(self, idx, ice, gen_field_replace_token=""):
        return f"P{idx}|{ice}"
    def get_gold_ans(self):
        return ["g0", "g1"]


class DummyModel:
    def __init__(self, is_api=False):
        self.max_out_len = 4
        self.is_api = is_api
    def parse_template(self, p, mode="gen"):
        return p
    def generate(self, inputs, max_out_len, output=None, session=None, **kwargs):
        # batch_inference uses synchronous generate
        if output is None:
            # Synchronous batch inference
            return ["output1", "output2"]
        else:
            # Async do_request
            output.success = True
            output.content = "test_output"


class DummyStatusCounter:
    async def post(self):
        pass
    async def rev(self):
        pass
    async def failed(self):
        pass
    async def finish(self):
        pass


class TestGenInferencer(unittest.TestCase):
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_get_data_list(self, m_abbr, m_build):
        """测试GenInferencer从retriever获取数据列表，包括gold和max_out_len"""
        m_build.return_value = DummyModel()
        inf = GenInferencer(model_cfg={}, batch_size=1)
        r = DummyRetriever(DummyDataset())
        data_list = inf.get_data_list(r)
        self.assertEqual(len(data_list), 2)
        self.assertEqual(data_list[0]["gold"], "g0")
        self.assertEqual(data_list[0]["max_out_len"], 5)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_batch_inference_async(self, m_abbr, m_build):
        """测试GenInferencer的batch_inference方法调用model.generate和report_cache_info_sync"""
        m_build.return_value = DummyModel()
        inf = GenInferencer(model_cfg={}, batch_size=1)
        
        datum = {
            "index": [[0], [1]],
            "prompt": [["p0"], ["p1"]],
            "data_abbr": [["d"], ["d"]],
            "max_out_len": [8, 8],
            "gold": [["g0"], ["g1"]],
        }
        
        try:
            inf.batch_inference(datum)
        except Exception as e:
            self.fail(f"batch_inference raised {type(e).__name__}: {e}")
        
        self.assertTrue(hasattr(DummyModel, 'generate'))

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_batch_inference_with_is_api(self, m_abbr, m_build):
        """测试GenInferencer在is_api=True时使用列表形式的max_out_len调用generate"""
        m_build.return_value = DummyModel(is_api=True)
        inf = GenInferencer(model_cfg={}, batch_size=1)
        
        datum = {
            "index": [[0], [1]],
            "prompt": [["p0"], ["p1"]],
            "data_abbr": [["d"], ["d"]],
            "max_out_len": [8, 8],
            "gold": [["g0"], ["g1"]],
        }
        
        inf.model.generate = mock.Mock(return_value=["out1", "out2"])
        inf.output_handler.report_cache_info_sync = mock.Mock(return_value=True)
        
        inf.batch_inference(datum)
        inf.model.generate.assert_called_once()

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_do_request(self, m_abbr, m_build):
        """测试GenInferencer的do_request异步方法成功处理请求并报告缓存信息"""
        m_build.return_value = DummyModel()
        inf = GenInferencer(model_cfg={}, batch_size=1)
        inf.status_counter = DummyStatusCounter()
        inf.output_handler.report_cache_info = mock.AsyncMock(return_value=True)
        
        async def async_generate(inputs, max_out_len, output=None, session=None, **kwargs):
            if output:
                output.success = True
                output.content = "test_output"
        
        inf.model.generate = async_generate
        
        data = {
            "index": 0,
            "prompt": "test_input",
            "data_abbr": "test",
            "max_out_len": 10,
            "gold": "test_gold",
        }
        
        async def run_test():
            token_bucket = mock.Mock()
            session = mock.Mock()
            await inf.do_request(data, token_bucket, session)
            inf.output_handler.report_cache_info.assert_called_once()
            
        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_do_request_failure(self, m_abbr, m_build):
        """测试GenInferencer的do_request在输出失败时仍报告缓存信息"""
        m_build.return_value = DummyModel()
        
        async def failed_generate(inputs, max_out_len, output=None, session=None, **kwargs):
            if output:
                output.success = False
                output.content = None
        
        inf = GenInferencer(model_cfg={}, batch_size=1)
        inf.model.generate = failed_generate
        inf.status_counter = DummyStatusCounter()
        inf.output_handler.report_cache_info = mock.AsyncMock(return_value=True)
        
        data = {
            "index": 0,
            "prompt": "test_input",
            "data_abbr": "test",
            "max_out_len": 10,
            "gold": "test_gold",
        }
        
        async def run_test():
            token_bucket = mock.Mock()
            session = mock.Mock()
            await inf.do_request(data, token_bucket, session)
            inf.output_handler.report_cache_info.assert_called_once()
            
        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_do_request_without_gold(self, m_abbr, m_build):
        """测试GenInferencer的do_request在没有gold参数时也能正常工作"""
        m_build.return_value = DummyModel()
        inf = GenInferencer(model_cfg={}, batch_size=1)
        inf.status_counter = DummyStatusCounter()
        inf.output_handler.report_cache_info = mock.AsyncMock(return_value=True)
        
        async def async_generate(inputs, max_out_len, output=None, session=None, **kwargs):
            if output:
                output.success = True
                output.content = "test_output"
        
        inf.model.generate = async_generate
        
        data = {
            "index": 0,
            "prompt": "test_input",
            "data_abbr": "test",
            "max_out_len": 10,
        }
        
        async def run_test():
            token_bucket = mock.Mock()
            session = mock.Mock()
            await inf.do_request(data, token_bucket, session)
            inf.output_handler.report_cache_info.assert_called_once()
            
        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()


