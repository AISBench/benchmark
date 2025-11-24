import unittest
from unittest import mock
import asyncio
from datasets import Dataset

from ais_bench.benchmark.openicl.icl_inferencer.ppl_inferencer import PPLInferencer
from ais_bench.benchmark.openicl.icl_inferencer.output_handler.ppl_inferencer_output_handler import (
    PPLRequestOutput,
    PPLResponseOutput
)
from ais_bench.benchmark.utils.logging.exceptions import AISBenchValueError
from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES, ICLE_CODES


class DummyDataset:
    def __init__(self):
        self.reader = type("R", (), {
            "output_column": "label",
            "get_max_out_len": lambda self=None: None
        })()
        self.train = Dataset.from_dict({"text": ["t0", "t1"], "label": [0, 1]})
        self.test = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
        self.abbr = "test_abbr"


class DummyRetriever:
    def __init__(self, dataset, labels=None):
        self.dataset = dataset
        self.dataset_reader = dataset.reader
        self._labels = labels or ["A", "B", "C"]
    
    def retrieve(self):
        return [[0], [1]]
    
    def generate_ice(self, idx_list):
        return "ICE"
    
    def generate_label_prompt(self, idx, ice, label):
        return f"P{idx}|{ice}|{label}"
    
    def get_gold_ans(self):
        return ["A", "B"]
    
    def get_labels(self):
        return self._labels


class DummyModel:
    def __init__(self, is_api=True):
        self.max_out_len = 4
        self.is_api = is_api
    
    def get_token_len_from_template(self, prompt, mode="ppl"):
        return len(prompt)
    
    async def get_ppl(self, prompt, max_out_len, output, session=None, **kwargs):
        output.success = True
        output.ppl = 2.5
        output.input = prompt
        output.origin_prompt_logprobs = {"token1": {"logprob": -0.5}}
    
    def parse_template(self, template, mode="ppl"):
        """
        Minimal implementation for tests: return the template unchanged.
        PPLInferencer expects a parsed prompt; returning the original string
        is sufficient for unit tests that use DummyModel.
        """
        return template


class DummyStatusCounter:
    async def post(self):
        pass
    
    async def rev(self):
        pass
    
    async def failed(self):
        pass
    
    async def finish(self):
        pass
    
    async def case_finish(self):
        pass


class TestPPLInferencer(unittest.TestCase):
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_init_normal(self, m_abbr, m_build):
        """测试PPLInferencer正常初始化"""
        m_build.return_value = DummyModel()
        inf = PPLInferencer(model_cfg={}, batch_size=1, mode="infer")
        self.assertEqual(inf.batch_size, 1)
        self.assertFalse(inf.perf_mode)
        self.assertIsNotNone(inf.output_handler)
        self.assertIsNone(inf.labels)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_init_with_labels(self, m_abbr, m_build):
        """测试PPLInferencer使用自定义labels初始化"""
        m_build.return_value = DummyModel()
        labels = ["X", "Y", "Z"]
        inf = PPLInferencer(model_cfg={}, batch_size=1, labels=labels)
        self.assertEqual(inf.labels, labels)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_init_perf_mode_error(self, m_abbr, m_build):
        """测试PPLInferencer在perf模式下应该抛出错误"""
        m_build.return_value = DummyModel()
        with self.assertRaises(AISBenchValueError) as ctx:
            PPLInferencer(model_cfg={}, batch_size=1, mode="perf")
        self.assertEqual(ctx.exception.error_code_str, ICLI_CODES.PERF_MODE_NOT_SUPPORTED_FOR_PPL_INFERENCE.full_code)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_init_stream_mode_error(self, m_abbr, m_build):
        """测试PPLInferencer在stream模式下应该抛出错误"""
        m_build.return_value = DummyModel()
        with self.assertRaises(AISBenchValueError) as ctx:
            PPLInferencer(model_cfg={"stream": True}, batch_size=1)
        self.assertEqual(ctx.exception.error_code_str, ICLI_CODES.STREAM_MODE_NOT_SUPPORTED_FOR_PPL_INFERENCE.full_code)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_get_data_list_normal(self, m_abbr, m_build):
        """测试get_data_list正常情况"""
        m_build.return_value = DummyModel()
        inf = PPLInferencer(model_cfg={}, batch_size=1)
        r = DummyRetriever(DummyDataset(), labels=["A", "B"])
        data_list = inf.get_data_list(r)
        
        self.assertEqual(len(data_list), 2)
        self.assertEqual(data_list[0]["gold"], "A")
        self.assertEqual(data_list[1]["gold"], "B")
        self.assertEqual(data_list[0]["data_abbr"], "test_abbr")
        self.assertIn("qa", data_list[0])
        self.assertIn("A", data_list[0]["qa"])
        self.assertIn("B", data_list[0]["qa"])
        self.assertEqual(data_list[0]["max_out_len"], 4)  # model.max_out_len

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_get_data_list_with_max_out_len(self, m_abbr, m_build):
        """测试get_data_list使用dataset指定的max_out_len"""
        dataset = DummyDataset()
        dataset.reader = type("R", (), {
            "output_column": "label",
            "get_max_out_len": lambda self=None: [5, 6]
        })()
        m_build.return_value = DummyModel()
        inf = PPLInferencer(model_cfg={}, batch_size=1)
        r = DummyRetriever(dataset, labels=["A", "B"])
        data_list = inf.get_data_list(r)
        
        self.assertEqual(data_list[0]["max_out_len"], 5)
        self.assertEqual(data_list[1]["max_out_len"], 6)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_get_data_list_length_mismatch(self, m_abbr, m_build):
        """测试get_data_list在data_list和gold_answer长度不匹配时抛出错误"""
        m_build.return_value = DummyModel()
        inf = PPLInferencer(model_cfg={}, batch_size=1)
        
        class MismatchRetriever(DummyRetriever):
            def get_gold_ans(self):
                return ["A"]  # 只有1个，但retrieve返回2个
        
        r = MismatchRetriever(DummyDataset(), labels=["A", "B"])
        with self.assertRaises(AISBenchValueError) as ctx:
            inf.get_data_list(r)
        self.assertEqual(ctx.exception.error_code_str, ICLE_CODES.PREDICTION_LENGTH_MISMATCH.full_code)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_calc_prediction_empty_list(self, m_abbr, m_build):
        """测试_calc_prediction处理空列表"""
        m_build.return_value = DummyModel()
        inf = PPLInferencer(model_cfg={}, batch_size=1)
        result = inf._calc_prediction([])
        self.assertIsNone(result)

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_calc_prediction_single_item(self, m_abbr, m_build):
        """测试_calc_prediction处理单个选项"""
        m_build.return_value = DummyModel()
        inf = PPLInferencer(model_cfg={}, batch_size=1)
        ppl_list = [{"label": "A", "ppl": 2.5}]
        result = inf._calc_prediction(ppl_list)
        self.assertEqual(result, "A")

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_calc_prediction_multiple_items(self, m_abbr, m_build):
        """测试_calc_prediction处理多个选项，选择最小PPL"""
        m_build.return_value = DummyModel()
        inf = PPLInferencer(model_cfg={}, batch_size=1)
        ppl_list = [
            {"label": "A", "ppl": 3.5},
            {"label": "B", "ppl": 1.5},  # 最小
            {"label": "C", "ppl": 2.5}
        ]
        result = inf._calc_prediction(ppl_list)
        self.assertEqual(result, "B")

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_calc_prediction_equal_ppl(self, m_abbr, m_build):
        """测试_calc_prediction处理相同PPL值（选择第一个）"""
        m_build.return_value = DummyModel()
        inf = PPLInferencer(model_cfg={}, batch_size=1)
        ppl_list = [
            {"label": "A", "ppl": 2.5},  # 第一个
            {"label": "B", "ppl": 2.5}
        ]
        result = inf._calc_prediction(ppl_list)
        self.assertEqual(result, "A")

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_do_request_success(self, m_abbr, m_build):
        """测试do_request成功情况"""
        m_build.return_value = DummyModel()
        inf = PPLInferencer(model_cfg={}, batch_size=1)
        inf.status_counter = DummyStatusCounter()
        
        # Mock output_handler
        mock_handler = mock.AsyncMock()
        inf.output_handler = mock_handler
        
        data = {
            "index": 0,
            "qa": {
                "A": {"prompt": "prompt_A"},
                "B": {"prompt": "prompt_B"}
            },
            "gold": "A",
            "max_out_len": 4,
            "data_abbr": "test"
        }
        
        async def run_test():
            token_bucket = mock.Mock()
            session = mock.Mock()
            await inf.do_request(data, token_bucket, session)
            
            # 验证output_handler被调用
            mock_handler.report_cache_info.assert_called_once()
            call_args = mock_handler.report_cache_info.call_args
            self.assertEqual(call_args[0][0], 0)  # index
            self.assertEqual(call_args[0][3], "test")  # data_abbr
            self.assertEqual(call_args[0][4], "A")  # gold
            resp_output = call_args[0][2]
            self.assertIsInstance(resp_output, PPLResponseOutput)
            self.assertTrue(resp_output.success)
            self.assertIsNotNone(resp_output.content)
            self.assertEqual(len(resp_output.label_ppl_list), 2)
        
        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_do_request_partial_failure(self, m_abbr, m_build):
        """测试do_request部分失败情况（某个选项失败）"""
        m_build.return_value = DummyModel()
        inf = PPLInferencer(model_cfg={}, batch_size=1)
        inf.status_counter = DummyStatusCounter()
        
        # Mock model to fail on second call
        call_count = [0]
        async def mock_get_ppl(prompt, max_out_len, output, session=None, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                output.success = False
                output.error_info = "API error"
            else:
                output.success = True
                output.ppl = 2.5
                output.input = prompt
                output.origin_prompt_logprobs = {}
        
        inf.model.get_ppl = mock_get_ppl
        
        # Mock output_handler
        mock_handler = mock.AsyncMock()
        inf.output_handler = mock_handler
        
        data = {
            "index": 0,
            "qa": {
                "A": {"prompt": "prompt_A"},
                "B": {"prompt": "prompt_B"}
            },
            "gold": "A",
            "max_out_len": 4,
            "data_abbr": "test"
        }
        
        async def run_test():
            token_bucket = mock.Mock()
            session = mock.Mock()
            await inf.do_request(data, token_bucket, session)
            
            # 验证output_handler被调用
            mock_handler.report_cache_info.assert_called_once()
            call_args = mock_handler.report_cache_info.call_args
            resp_output = call_args[0][2]
            self.assertIsInstance(resp_output, PPLResponseOutput)
            self.assertFalse(resp_output.success)
            self.assertEqual(resp_output.error_info, "API error")
            self.assertIsNone(resp_output.content)
            # 只有第一个成功的被添加到列表
            self.assertEqual(len(resp_output.label_ppl_list), 1)
        
        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_do_request_all_failure(self, m_abbr, m_build):
        """测试do_request全部失败情况"""
        m_build.return_value = DummyModel()
        inf = PPLInferencer(model_cfg={}, batch_size=1)
        inf.status_counter = DummyStatusCounter()
        
        # Mock model to always fail
        async def mock_get_ppl(prompt, max_out_len, output, session=None, **kwargs):
            output.success = False
            output.error_info = "API error"
        
        inf.model.get_ppl = mock_get_ppl
        
        # Mock output_handler
        mock_handler = mock.AsyncMock()
        inf.output_handler = mock_handler
        
        data = {
            "index": 0,
            "qa": {
                "A": {"prompt": "prompt_A"}
            },
            "gold": "A",
            "max_out_len": 4,
            "data_abbr": "test"
        }
        
        async def run_test():
            token_bucket = mock.Mock()
            session = mock.Mock()
            await inf.do_request(data, token_bucket, session)
            
            # 验证output_handler被调用
            mock_handler.report_cache_info.assert_called_once()
            call_args = mock_handler.report_cache_info.call_args
            resp_output = call_args[0][2]
            self.assertIsInstance(resp_output, PPLResponseOutput)
            self.assertFalse(resp_output.success)
            self.assertEqual(resp_output.error_info, "API error")
            self.assertIsNone(resp_output.content)
            self.assertEqual(len(resp_output.label_ppl_list), 0)
        
        asyncio.run(run_test())

    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.build_model_from_cfg")
    @mock.patch("ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer.model_abbr_from_cfg", return_value="mabbr")
    def test_do_request_with_failed_ppl(self, m_abbr, m_build):
        """测试do_request中output.success为False时使用0作为ppl"""
        m_build.return_value = DummyModel()
        inf = PPLInferencer(model_cfg={}, batch_size=1)
        inf.status_counter = DummyStatusCounter()
        
        # Mock model to return success=False but still add to list
        async def mock_get_ppl(prompt, max_out_len, output, session=None, **kwargs):
            output.success = False  # 但不会break，因为这是第一个
            output.ppl = None
            output.input = prompt
        
        inf.model.get_ppl = mock_get_ppl
        
        # Mock output_handler
        mock_handler = mock.AsyncMock()
        inf.output_handler = mock_handler
        
        data = {
            "index": 0,
            "qa": {
                "A": {"prompt": "prompt_A"}
            },
            "gold": "A",
            "max_out_len": 4,
            "data_abbr": "test"
        }
        
        async def run_test():
            token_bucket = mock.Mock()
            session = mock.Mock()
            await inf.do_request(data, token_bucket, session)
            
            # 验证output_handler被调用
            mock_handler.report_cache_info.assert_called_once()
            call_args = mock_handler.report_cache_info.call_args
            resp_output = call_args[0][2]
            # 当success=False时，会break，所以不会添加到ppl_list
            self.assertFalse(resp_output.success)
            self.assertEqual(len(resp_output.label_ppl_list), 0)
        
        asyncio.run(run_test())


if __name__ == '__main__':
    unittest.main()

