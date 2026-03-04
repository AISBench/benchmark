import sys
import os
import pytest
from unittest.mock import patch, MagicMock, call
import uuid

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from ais_bench.benchmark.models.output import LMMOutput


class TestLMMGenInferencer:
    def setup_method(self):
        """设置测试环境"""
        self.mock_model_cfg = {
            'type': 'MockModel',
            'abbr': 'test_model'
        }

    def test_init(self):
        """测试LMMGenInferencer初始化"""
        mock_model = MagicMock()

        with patch('ais_bench.benchmark.utils.config.build.MODELS.build', return_value=mock_model):
            from ais_bench.benchmark.openicl.icl_inferencer.icl_lmm_gen_inferencer import LMMGenInferencer

            with patch('ais_bench.benchmark.openicl.icl_inferencer.icl_lmm_gen_inferencer.LMMGenInferencerOutputHandler') as mock_handler:
                inferencer = LMMGenInferencer(
                    model_cfg=self.mock_model_cfg,
                    stopping_criteria=[],
                    batch_size=1,
                    mode='infer'
                )

                assert inferencer.model_cfg == self.mock_model_cfg
                assert inferencer.batch_size == 1
                mock_handler.assert_called_once()

    def test_init_with_custom_params(self):
        """测试使用自定义参数初始化"""
        mock_model = MagicMock()

        with patch('ais_bench.benchmark.utils.config.build.MODELS.build', return_value=mock_model):
            from ais_bench.benchmark.openicl.icl_inferencer.icl_lmm_gen_inferencer import LMMGenInferencer

            with patch('ais_bench.benchmark.openicl.icl_inferencer.icl_lmm_gen_inferencer.LMMGenInferencerOutputHandler') as mock_handler:
                inferencer = LMMGenInferencer(
                    model_cfg=self.mock_model_cfg,
                    stopping_criteria=['stop1', 'stop2'],
                    batch_size=4,
                    mode='perf',
                    gen_field_replace_token='<GEN>',
                    output_json_filepath='/test/output',
                    save_every=50
                )

                assert inferencer.stopping_criteria == ['stop1', 'stop2']
                assert inferencer.batch_size == 4
                assert inferencer.perf_mode is True

    def test_inference(self):
        """测试inference方法"""
        mock_model = MagicMock()

        with patch('ais_bench.benchmark.utils.config.build.MODELS.build', return_value=mock_model):
            from ais_bench.benchmark.openicl.icl_inferencer.icl_lmm_gen_inferencer import LMMGenInferencer

            with patch('ais_bench.benchmark.openicl.icl_inferencer.icl_lmm_gen_inferencer.LMMGenInferencerOutputHandler') as mock_handler:
                mock_handler_instance = MagicMock()
                mock_handler.return_value = mock_handler_instance

                inferencer = LMMGenInferencer(
                    model_cfg=self.mock_model_cfg,
                    batch_size=1
                )

                mock_retriever = MagicMock()

                with patch('ais_bench.benchmark.openicl.icl_inferencer.icl_gen_inferencer.GenInferencer.inference') as mock_super_inference:
                    mock_super_inference.return_value = []

                    result = inferencer.inference(mock_retriever, '/test/output.jsonl')

                    mock_handler_instance.set_output_path.assert_called_once_with('/test/output.jsonl')
                    mock_super_inference.assert_called_once()

    def test_batch_inference(self):
        """测试batch_inference方法"""
        mock_model = MagicMock()

        with patch('ais_bench.benchmark.utils.config.build.MODELS.build', return_value=mock_model):
            from ais_bench.benchmark.openicl.icl_inferencer.icl_lmm_gen_inferencer import LMMGenInferencer

            with patch('ais_bench.benchmark.openicl.icl_inferencer.icl_lmm_gen_inferencer.LMMGenInferencerOutputHandler') as mock_handler:
                mock_handler_instance = MagicMock()
                mock_handler.return_value = mock_handler_instance

                inferencer = LMMGenInferencer(
                    model_cfg=self.mock_model_cfg,
                    batch_size=1
                )

                inferencer.model = MagicMock()

                datum = {
                    'index': [0, 1],
                    'prompt': ['prompt1', 'prompt2'],
                    'data_abbr': ['dataset1', 'dataset1'],
                    'gold': ['gold1', 'gold2'],
                    'extra_param': 'value'
                }

                inferencer.batch_inference(datum)

                inferencer.model.generate.assert_called_once()
                assert mock_handler_instance.report_cache_info_sync.call_count == 2

    def test_batch_inference_without_gold(self):
        """测试batch_inference方法，没有gold字段"""
        mock_model = MagicMock()

        with patch('ais_bench.benchmark.utils.config.build.MODELS.build', return_value=mock_model):
            from ais_bench.benchmark.openicl.icl_inferencer.icl_lmm_gen_inferencer import LMMGenInferencer

            with patch('ais_bench.benchmark.openicl.icl_inferencer.icl_lmm_gen_inferencer.LMMGenInferencerOutputHandler') as mock_handler:
                mock_handler_instance = MagicMock()
                mock_handler.return_value = mock_handler_instance

                inferencer = LMMGenInferencer(
                    model_cfg=self.mock_model_cfg,
                    batch_size=1
                )

                inferencer.model = MagicMock()

                datum = {
                    'index': [0],
                    'prompt': ['prompt1'],
                    'data_abbr': ['dataset1']
                }

                inferencer.batch_inference(datum)

                inferencer.model.generate.assert_called_once()
                mock_handler_instance.report_cache_info_sync.assert_called_once()

    def test_batch_inference_uuid_generation(self):
        """测试batch_inference方法中UUID生成"""
        mock_model = MagicMock()

        with patch('ais_bench.benchmark.utils.config.build.MODELS.build', return_value=mock_model):
            from ais_bench.benchmark.openicl.icl_inferencer.icl_lmm_gen_inferencer import LMMGenInferencer

            with patch('ais_bench.benchmark.openicl.icl_inferencer.icl_lmm_gen_inferencer.LMMGenInferencerOutputHandler') as mock_handler:
                mock_handler_instance = MagicMock()
                mock_handler.return_value = mock_handler_instance

                inferencer = LMMGenInferencer(
                    model_cfg=self.mock_model_cfg,
                    batch_size=1
                )

                inferencer.model = MagicMock()

                datum = {
                    'index': [0],
                    'prompt': ['prompt1'],
                    'data_abbr': ['dataset1']
                }

                inferencer.batch_inference(datum)

                call_args = mock_handler_instance.report_cache_info_sync.call_args
                output = call_args[0][2]
                assert isinstance(output, LMMOutput)
                assert len(output.uuid) == 32
