import unittest
from unittest import mock
import asyncio

from datasets import Dataset

from ais_bench.benchmark.openicl.icl_inferencer.icl_multiturn_inferencer import MultiTurnGenInferencer
from ais_bench.benchmark.models.output import RequestOutput
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.utils.logging.exceptions import ParameterValueError


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
        # Return multiturn prompt list
        return PromptList([
            {"role": "SYS", "prompt": "start"},
            {"role": "USER", "prompt": "q1"},
            {"role": "BOT", "prompt": ""},
            {"role": "USER", "prompt": "q2"},
            {"role": "BOT", "prompt": ""},
            {"role": "SYS", "prompt": "end"},
        ])
    def get_gold_ans(self):
        return ["g0", "g1"]


class DummyModel:
    def __init__(self):
        self.max_out_len = 4
        self.is_api = False
    def parse_template(self, p, mode="gen"):
        return p
    async def generate(self, inputs, max_out_len, output=None, session=None, **kwargs):
        if output:
            output.success = True
            output.content = "test_output"
        return "output"


class DummyStatusCounter:
    async def post(self):
        pass
    async def rev(self):
        pass
    async def failed(self):
        pass
    async def finish(self):
        pass


class TestMultiTurnGenInferencer(unittest.TestCase):
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_init(self, m_abbr, m_build):
        """测试MultiTurnGenInferencer初始化，设置infer_mode、gen_field_replace_token和stopping_criteria"""
        m_build.return_value = DummyModel()
        inf = MultiTurnGenInferencer(model_cfg={}, infer_mode="every")
        self.assertEqual(inf.infer_mode, "every")
        self.assertEqual(inf.gen_field_replace_token, "")
        self.assertEqual(inf.stopping_criteria, [])

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_do_request_invalid_mode(self, m_abbr, m_build):
        """测试MultiTurnGenInferencer在infer_mode无效时抛出ParameterValueError"""
        m_build.return_value = DummyModel()
        inf = MultiTurnGenInferencer(model_cfg={}, infer_mode="invalid")
        inf.status_counter = DummyStatusCounter()
        
        data = {
            "index": 0,
            "prompt": PromptList([{"role": "USER", "prompt": "test"}]),
            "data_abbr": "test",
            "max_out_len": 10,
        }
        
        async def run_test():
            token_bucket = mock.Mock()
            session = mock.Mock()
            with self.assertRaises(ParameterValueError):
                await inf.do_request(data, token_bucket, session)
                
        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_multiturn_inferencer.PromptList")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_infer_last(self, m_abbr, m_build, m_promptlist):
        """测试MultiTurnGenInferencer的infer_last方法对最后一个BOT轮次进行推理"""
        m_build.return_value = DummyModel()
        m_promptlist.side_effect = lambda x: list(x) if not isinstance(x, list) else x
        
        inf = MultiTurnGenInferencer(model_cfg={}, infer_mode="last")
        inf.status_counter = DummyStatusCounter()
        inf.output_handler.report_cache_info = mock.AsyncMock(return_value=True)
        inf.model.parse_template = mock.Mock(return_value="parsed_history")
        
        chat = [
            {"role": "SYS", "prompt": "start"},
            {"role": "USER", "prompt": "q1"},
            {"role": "BOT", "prompt": ""},
            {"role": "SYS", "prompt": "end"},
        ]
        
        data = {
            "index": 0,
            "prompt": chat,
            "data_abbr": "test",
            "max_out_len": 10,
            "gold": "test_gold",
        }
        
        async def run_test():
            session = mock.Mock()
            await inf.infer_last(data, session)
            inf.output_handler.report_cache_info.assert_called_once()
            
        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_multiturn_inferencer.PromptList")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_infer_last_with_list_max_out_len(self, m_abbr, m_build, m_promptlist):
        """测试infer_last方法处理列表形式的max_out_len"""
        m_build.return_value = DummyModel()
        m_promptlist.side_effect = lambda x: list(x) if not isinstance(x, list) else x
        
        inf = MultiTurnGenInferencer(model_cfg={}, infer_mode="last")
        inf.status_counter = DummyStatusCounter()
        inf.output_handler.report_cache_info = mock.AsyncMock(return_value=True)
        inf.model.parse_template = mock.Mock(return_value="parsed_history")
        
        chat = [
            {"role": "SYS", "prompt": "start"},
            {"role": "USER", "prompt": "q1"},
            {"role": "BOT", "prompt": ""},
            {"role": "SYS", "prompt": "end"},
        ]
        
        data = {
            "index": 0,
            "prompt": chat,
            "data_abbr": "test",
            "max_out_len": [10, 20],
            "gold": "test_gold",
        }
        
        async def run_test():
            session = mock.Mock()
            await inf.infer_last(data, session)
            inf.output_handler.report_cache_info.assert_called_once()
            
        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_infer_every(self, m_abbr, m_build):
        """测试infer_every方法对每个BOT轮次进行推理"""
        m_build.return_value = DummyModel()
        inf = MultiTurnGenInferencer(model_cfg={}, infer_mode="every")
        inf.status_counter = DummyStatusCounter()
        inf.output_handler.report_cache_info = mock.AsyncMock(return_value=True)
        inf.model.parse_template = mock.Mock(return_value="parsed_history")
        
        chat = [
            {"role": "SYS", "prompt": "start"},
            {"role": "USER", "prompt": "q1"},
            {"role": "BOT", "prompt": ""},
            {"role": "USER", "prompt": "q2"},
            {"role": "BOT", "prompt": ""},
            {"role": "SYS", "prompt": "end"},
        ]
        
        data = {
            "index": 0,
            "prompt": chat,
            "data_abbr": "test",
            "max_out_len": [10, 20],
            "gold": "test_gold",
        }
        
        async def run_test():
            session = mock.Mock()
            await inf.infer_every(data, session)
            self.assertGreaterEqual(inf.output_handler.report_cache_info.call_count, 1)
            
        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_infer_every_with_failure(self, m_abbr, m_build):
        """测试infer_every方法在输出失败时中断推理"""
        m_build.return_value = DummyModel()
        
        class FailedModel(DummyModel):
            async def generate(self, inputs, max_out_len, output=None, session=None, **kwargs):
                if output:
                    output.success = False
                    output.content = None
        
        inf = MultiTurnGenInferencer(model_cfg={}, infer_mode="every")
        inf.model = FailedModel()
        inf.status_counter = DummyStatusCounter()
        inf.output_handler.report_cache_info = mock.AsyncMock(return_value=True)
        inf.model.parse_template = mock.Mock(return_value="parsed_history")
        
        chat = [
            {"role": "SYS", "prompt": "start"},
            {"role": "USER", "prompt": "q1"},
            {"role": "BOT", "prompt": ""},
            {"role": "SYS", "prompt": "end"},
        ]
        
        data = {
            "index": 0,
            "prompt": chat,
            "data_abbr": "test",
            "max_out_len": 10,
            "gold": "test_gold",
        }
        
        async def run_test():
            session = mock.Mock()
            await inf.infer_every(data, session)
            inf.output_handler.report_cache_info.assert_called_once()
            
        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_infer_every_with_gt(self, m_abbr, m_build):
        """测试infer_every_with_gt方法对每个BOT轮次使用ground truth进行推理"""
        m_build.return_value = DummyModel()
        inf = MultiTurnGenInferencer(model_cfg={}, infer_mode="every_with_gt")
        inf.status_counter = DummyStatusCounter()
        inf.output_handler.report_cache_info = mock.AsyncMock(return_value=True)
        inf.model.parse_template = mock.Mock(return_value="parsed_history")
        
        chat = [
            {"role": "SYS", "prompt": "start"},
            {"role": "USER", "prompt": "q1"},
            {"role": "BOT", "prompt": ""},
            {"role": "SYS", "prompt": "end"},
        ]
        
        data = {
            "index": 0,
            "prompt": chat,
            "data_abbr": "test",
            "max_out_len": [10, 20],
            "gold": "test_gold",
        }
        
        async def run_test():
            session = mock.Mock()
            await inf.infer_every_with_gt(data, session)
            self.assertGreaterEqual(inf.output_handler.report_cache_info.call_count, 1)
            
        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_get_data_list(self, m_abbr, m_build):
        """测试MultiTurnGenInferencer从retriever获取数据列表，prompt为PromptList格式"""
        m_build.return_value = DummyModel()
        inf = MultiTurnGenInferencer(model_cfg={}, infer_mode="every")
        inf.is_main_process = True
        r = DummyRetriever(DummyDataset())
        data_list = inf.get_data_list(r)
        self.assertEqual(len(data_list), 2)
        self.assertIsInstance(data_list[0]["prompt"], PromptList)
        self.assertEqual(data_list[0]["gold"], "g0")

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_get_data_list_with_max_out_len(self, m_abbr, m_build):
        """测试get_data_list使用数据集中的max_out_len值"""
        m_build.return_value = DummyModel()
        inf = MultiTurnGenInferencer(model_cfg={}, infer_mode="every")
        inf.is_main_process = True
        
        ds = DummyDataset()
        ds.reader.get_max_out_len = lambda self=None: [10, 20]
        r = DummyRetriever(ds)
        data_list = inf.get_data_list(r)
        self.assertEqual(data_list[0]["max_out_len"], 10)
        self.assertEqual(data_list[1]["max_out_len"], 20)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_get_data_list_with_empty_max_out_len(self, m_abbr, m_build):
        """测试get_data_list在max_out_len为None或0时使用模型的max_out_len"""
        m_build.return_value = DummyModel()
        inf = MultiTurnGenInferencer(model_cfg={}, infer_mode="every")
        inf.is_main_process = True
        
        ds = DummyDataset()
        ds.reader.get_max_out_len = lambda self=None: [None, 0]
        r = DummyRetriever(ds)
        data_list = inf.get_data_list(r)
        self.assertEqual(data_list[0]["max_out_len"], 4)
        self.assertEqual(data_list[1]["max_out_len"], 4)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_do_request_every_with_gt(self, m_abbr, m_build):
        """测试do_request在infer_mode='every_with_gt'时调用infer_every_with_gt方法"""
        m_build.return_value = DummyModel()
        inf = MultiTurnGenInferencer(model_cfg={}, infer_mode="every_with_gt")
        inf.status_counter = DummyStatusCounter()
        inf.infer_every_with_gt = mock.AsyncMock()
        
        data = {
            "index": 0,
            "prompt": PromptList([{"role": "USER", "prompt": "test"}]),
            "data_abbr": "test",
            "max_out_len": 10,
        }
        
        async def run_test():
            token_bucket = mock.Mock()
            session = mock.Mock()
            await inf.do_request(data, token_bucket, session)
            inf.infer_every_with_gt.assert_called_once()
                
        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_infer_every_with_gt_failure(self, m_abbr, m_build):
        """测试infer_every_with_gt方法在输出失败时中断推理"""
        m_build.return_value = DummyModel()
        
        class FailedModel(DummyModel):
            async def generate(self, inputs, max_out_len, output=None, session=None, **kwargs):
                if output:
                    output.success = False
                    output.content = None
        
        inf = MultiTurnGenInferencer(model_cfg={}, infer_mode="every_with_gt")
        inf.model = FailedModel()
        inf.status_counter = DummyStatusCounter()
        inf.output_handler.report_cache_info = mock.AsyncMock(return_value=True)
        inf.model.parse_template = mock.Mock(return_value="parsed_history")
        
        chat = [
            {"role": "SYS", "prompt": "start"},
            {"role": "USER", "prompt": "q1"},
            {"role": "BOT", "prompt": ""},
            {"role": "SYS", "prompt": "end"},
        ]
        
        data = {
            "index": 0,
            "prompt": chat,
            "data_abbr": "test",
            "max_out_len": 10,
            "gold": "test_gold",
        }
        
        async def run_test():
            session = mock.Mock()
            await inf.infer_every_with_gt(data, session)
            inf.output_handler.report_cache_info.assert_called_once()
            
        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_infer_every_with_list_max_out_len(self, m_abbr, m_build):
        """测试infer_every方法处理列表形式的max_out_len"""
        m_build.return_value = DummyModel()
        inf = MultiTurnGenInferencer(model_cfg={}, infer_mode="every")
        inf.status_counter = DummyStatusCounter()
        inf.output_handler.report_cache_info = mock.AsyncMock(return_value=True)
        inf.model.parse_template = mock.Mock(return_value="parsed_history")
        
        chat = [
            {"role": "SYS", "prompt": "start"},
            {"role": "USER", "prompt": "q1"},
            {"role": "BOT", "prompt": ""},
            {"role": "SYS", "prompt": "end"},
        ]
        
        data = {
            "index": 0,
            "prompt": chat,
            "data_abbr": "test",
            "max_out_len": [10, 20],
            "gold": "test_gold",
        }
        
        async def run_test():
            session = mock.Mock()
            await inf.infer_every(data, session)
            inf.output_handler.report_cache_info.assert_called_once()
            
        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_get_data_list_without_gold(self, m_abbr, m_build):
        """测试get_data_list在retriever没有gold_ans时不添加gold字段"""
        m_build.return_value = DummyModel()
        inf = MultiTurnGenInferencer(model_cfg={}, infer_mode="every")
        inf.is_main_process = True
        
        class NoGoldRetriever(DummyRetriever):
            def get_gold_ans(self):
                return None
        
        r = NoGoldRetriever(DummyDataset())
        data_list = inf.get_data_list(r)
        self.assertEqual(len(data_list), 2)
        self.assertNotIn("gold", data_list[0])


if __name__ == '__main__':
    unittest.main()

