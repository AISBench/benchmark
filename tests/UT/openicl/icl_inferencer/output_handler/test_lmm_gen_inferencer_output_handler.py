import sys
import os
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import uuid
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ais_bench.benchmark.openicl.icl_inferencer.output_handler.lmm_gen_inferencer_output_handler import (
    LMMGenInferencerOutputHandler
)
from ais_bench.benchmark.models.output import LMMOutput
from ais_bench.benchmark.utils.logging.exceptions import AISBenchRuntimeError


class TestLMMGenInferencerOutputHandler:
    def setup_method(self):
        """设置测试环境"""
        self.handler = LMMGenInferencerOutputHandler()
        self.handler.output_path = None

    def test_init(self):
        """测试初始化"""
        handler = LMMGenInferencerOutputHandler(perf_mode=True, save_every=50)
        handler.output_path = None
        assert handler.perf_mode is True
        assert handler.save_every == 50

    def test_set_output_path(self):
        """测试set_output_path方法"""
        self.handler.set_output_path('/test/output/path')
        assert self.handler.output_path == '/test/output/path'

    def test_get_prediction_result_with_string_output(self):
        """测试get_prediction_result方法，字符串输出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.handler.set_output_path(tmpdir)
            result = self.handler.get_prediction_result(
                output="test prediction",
                gold="test gold",
                input="test input",
                data_abbr="test_dataset"
            )

            assert result["success"] is True
            assert result["prediction"] == "test prediction"
            assert result["gold"] == "test gold"
            assert result["origin_prompt"] == "test input"
            assert len(result["uuid"]) == 32

    def test_get_prediction_result_with_lmm_output(self):
        """测试get_prediction_result方法，LMMOutput输出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.handler.set_output_path(tmpdir)

            output = LMMOutput()
            output.uuid = "test_uuid_123"
            output.success = True
            output.content = ["text content"]

            result = self.handler.get_prediction_result(
                output=output,
                gold="test gold",
                input="test input",
                data_abbr="test_dataset"
            )

            assert result["success"] is True
            assert result["uuid"] == "test_uuid_123"
            assert result["prediction"] == "text content"
            assert result["gold"] == "test gold"

    def test_get_prediction_result_with_image_output(self):
        """测试get_prediction_result方法，图片输出"""
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmpdir:
            self.handler.set_output_path(tmpdir)

            output = LMMOutput()
            output.uuid = "test_uuid_456"
            output.success = True
            test_image = Image.new('RGB', (100, 100), color='red')
            output.content = [test_image]

            result = self.handler.get_prediction_result(
                output=output,
                data_abbr="test_dataset"
            )

            assert result["success"] is True
            assert "image_test_uuid_456_0.png" in result["prediction"]

    def test_get_prediction_result_creates_output_dir(self):
        """测试get_prediction_result方法创建输出目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'subdir')
            self.handler.set_output_path(output_path)

            output = LMMOutput()
            output.uuid = "test_uuid"
            output.success = True
            output.content = ["text"]

            result = self.handler.get_prediction_result(
                output=output,
                data_abbr="test_dataset"
            )

            assert os.path.exists(os.path.join(output_path, "test_dataset_out_file"))

    def test_get_prediction_result_with_long_base64_input(self):
        """测试get_prediction_result方法，处理长base64输入"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.handler.set_output_path(tmpdir)

            long_base64 = "a" * 300
            input_data = [{
                "prompt": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": long_base64
                        }
                    }
                ]
            }]

            output = LMMOutput()
            output.uuid = "test_uuid"
            output.success = True
            output.content = ["text"]

            result = self.handler.get_prediction_result(
                output=output,
                input=input_data,
                data_abbr="test_dataset"
            )

            assert result["success"] is True
            assert "..." in input_data[0]["prompt"][0]["image_url"]["url"]

    def test_get_prediction_result_with_dict_input(self):
        """测试get_prediction_result方法，字典类型输入"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.handler.set_output_path(tmpdir)

            input_data = [{
                "prompt": [
                    {
                        "type": "text",
                        "text": "test text"
                    }
                ]
            }]

            output = LMMOutput()
            output.uuid = "test_uuid"
            output.success = True
            output.content = ["text"]

            result = self.handler.get_prediction_result(
                output=output,
                input=input_data,
                data_abbr="test_dataset"
            )

            assert result["success"] is True

    def test_get_prediction_result_with_empty_input(self):
        """测试get_prediction_result方法，空输入"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.handler.set_output_path(tmpdir)

            output = LMMOutput()
            output.uuid = "test_uuid"
            output.success = True
            output.content = ["text"]

            result = self.handler.get_prediction_result(
                output=output,
                input=None,
                data_abbr="test_dataset"
            )

            assert result["success"] is True
            assert result["origin_prompt"] == ""

    def test_get_prediction_result_without_gold(self):
        """测试get_prediction_result方法，没有gold"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.handler.set_output_path(tmpdir)
            result = self.handler.get_prediction_result(
                output="test prediction",
                input="test input",
                data_abbr="test_dataset"
            )

            assert "gold" not in result

    def test_get_prediction_result_with_failed_output(self):
        """测试get_prediction_result方法，失败的输出"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.handler.set_output_path(tmpdir)

            output = LMMOutput()
            output.uuid = "test_uuid"
            output.success = False
            output.error_info = "test error"
            output.content = [""]

            result = self.handler.get_prediction_result(
                output=output,
                data_abbr="test_dataset"
            )

            assert result["success"] is False

    def test_get_prediction_result_with_non_dict_prompt_items(self):
        """测试get_prediction_result方法，非字典类型的prompt项"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.handler.set_output_path(tmpdir)

            input_data = [{
                "prompt": [
                    "string_item",
                    123,
                    None
                ]
            }]

            output = LMMOutput()
            output.uuid = "test_uuid"
            output.success = True
            output.content = ["text"]

            result = self.handler.get_prediction_result(
                output=output,
                input=input_data,
                data_abbr="test_dataset"
            )

            assert result["success"] is True

    def test_get_prediction_result_with_non_dict_image_url(self):
        """测试get_prediction_result方法，非字典类型的image_url"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.handler.set_output_path(tmpdir)

            input_data = [{
                "prompt": [
                    {
                        "type": "image_url",
                        "image_url": "not_a_dict"
                    }
                ]
            }]

            output = LMMOutput()
            output.uuid = "test_uuid"
            output.success = True
            output.content = ["text"]

            result = self.handler.get_prediction_result(
                output=output,
                input=input_data,
                data_abbr="test_dataset"
            )

            assert result["success"] is True

    def test_get_prediction_result_with_non_string_url(self):
        """测试get_prediction_result方法，非字符串类型的URL"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.handler.set_output_path(tmpdir)

            input_data = [{
                "prompt": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": 123
                        }
                    }
                ]
            }]

            output = LMMOutput()
            output.uuid = "test_uuid"
            output.success = True
            output.content = ["text"]

            result = self.handler.get_prediction_result(
                output=output,
                input=input_data,
                data_abbr="test_dataset"
            )

            assert result["success"] is True
