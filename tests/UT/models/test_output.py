import pytest
import numpy as np
import asyncio
import tempfile
import os
from PIL import Image
from ais_bench.benchmark.models.output import Output, RequestOutput, FunctionCallOutput, LMMOutput
import time


# 创建Output的具体实现类用于测试
class ConcreteOutput(Output):
    def get_metrics(self) -> dict:
        return {"test_metric": "value"}


def test_output_initialization():
    """测试Output类的初始化功能"""
    # 测试默认参数初始化
    output = ConcreteOutput()
    assert output.perf_mode is False
    assert output.success is False
    assert output.error_info == ""
    assert isinstance(output.time_points, list)
    assert output.content == ""
    assert output.reasoning_content == ""
    assert output.input_tokens == 0
    assert output.output_tokens == 0
    assert isinstance(output.extra_perf_data, dict)
    assert isinstance(output.extra_details_data, dict)
    assert output.input is None
    assert output.uuid == ""
    assert output.turn_id == 0

    # 测试perf_mode=True初始化
    output_perf = ConcreteOutput(perf_mode=True)
    assert output_perf.perf_mode is True


def test_concate_reasoning_content():
    """测试_concate_reasoning_content方法的不同分支"""
    output = ConcreteOutput()

    # 测试reasoning_content和content都不为空的情况
    result1 = output._concate_reasoning_content("content", "reasoning")
    assert result1 == "reasoning</think>content"

    # 测试reasoning_content不为空但content为空的情况
    result2 = output._concate_reasoning_content("", "reasoning")
    assert result2 == "reasoning"

    # 测试reasoning_content为空但content不为空的情况
    result3 = output._concate_reasoning_content("content", "")
    assert result3 == "content"

    # 测试两者都为空的情况
    result4 = output._concate_reasoning_content("", "")
    assert result4 == ""


def test_get_prediction():
    """测试get_prediction方法的不同分支"""
    output = ConcreteOutput()

    # 测试reasoning_content为空的情况
    output.content = "test content"
    output.reasoning_content = ""
    assert output.get_prediction() == "test content"

    # 测试content和reasoning_content都是列表的情况
    output.content = ["content1", "content2"]
    output.reasoning_content = ["reasoning1", "reasoning2"]
    assert output.get_prediction() == ["reasoning1</think>content1", "reasoning2</think>content2"]

    # 测试reasoning_content是字符串的情况
    output.content = "content string"
    output.reasoning_content = "reasoning string"
    assert output.get_prediction() == "reasoning string</think>content string"

    # 测试其他类型的情况（应该返回原始content）
    output.content = "test content"
    output.reasoning_content = None  # 非字符串非列表类型
    assert output.get_prediction() == "test content"


def test_to_dict():
    """测试to_dict方法"""
    output = ConcreteOutput()
    output.content = "test"
    output.uuid = "test_uuid"
    output.turn_id = 1

    result = output.to_dict()
    assert isinstance(result, dict)
    assert result["content"] == "test"
    assert result["uuid"] == "test_uuid"
    assert result["turn_id"] == 1
    # 确保所有属性都被包含
    assert "perf_mode" in result
    assert "success" in result
    assert "error_info" in result
    assert "time_points" in result
    assert "reasoning_content" in result
    assert "input_tokens" in result
    assert "output_tokens" in result
    assert "extra_perf_data" in result
    assert "extra_details_data" in result
    assert "input" in result


def test_record_time_point():
    """测试record_time_point异步方法"""
    # 测试perf_mode=False时不记录时间点
    output = ConcreteOutput(perf_mode=False)
    asyncio.run(output.record_time_point())
    assert len(output.time_points) == 0

    # 测试perf_mode=True时记录时间点
    output_perf = ConcreteOutput(perf_mode=True)
    asyncio.run(output_perf.record_time_point())
    assert len(output_perf.time_points) == 1
    assert isinstance(output_perf.time_points[0], float)

    # 测试多次记录时间点
    asyncio.run(output_perf.record_time_point())
    assert len(output_perf.time_points) == 2


def test_clear_time_points():
    """测试clear_time_points异步方法"""
    output = ConcreteOutput(perf_mode=True)
    asyncio.run(output.record_time_point())
    asyncio.run(output.record_time_point())
    assert len(output.time_points) == 2

    asyncio.run(output.clear_time_points())
    assert len(output.time_points) == 0


def test_request_output_get_metrics():
    """测试RequestOutput类的get_metrics方法的不同分支"""
    # 测试success=False的情况
    output = RequestOutput()
    output.success = False
    output.error_info = "test error"
    output.content = "test content"
    output.reasoning_content = "test reasoning"
    output.perf_mode = True

    metrics = output.get_metrics()
    assert isinstance(metrics, dict)
    assert metrics["success"] is False
    assert metrics["error_info"] == "test error"
    # 确保clean_result函数被正确应用
    assert "content" not in metrics
    assert "reasoning_content" not in metrics
    assert "perf_mode" not in metrics
    # 确保prediction字段被设置
    assert "prediction" in metrics

    # 测试success=True但time_points.size <= 1的情况
    output = RequestOutput()
    output.success = True
    output.time_points = [time.perf_counter()]

    metrics = output.get_metrics()
    assert metrics["success"] is False  # 应该被设置为False
    assert metrics["error_info"] == "chunk size is less than 2"
    # 确保time_points被转换为numpy数组
    assert isinstance(metrics["time_points"], np.ndarray)

    # 测试success=True且time_points.size > 1的情况
    output = RequestOutput()
    output.success = True
    output.time_points = [time.perf_counter() - 1, time.perf_counter()]
    output.input_tokens = 10
    output.output_tokens = 20

    metrics = output.get_metrics()
    assert metrics["success"] is True
    assert isinstance(metrics["time_points"], np.ndarray)
    assert metrics["time_points"].size == 2
    assert metrics["input_tokens"] == 10
    assert metrics["output_tokens"] == 20


def test_request_output_edge_cases():
    """测试RequestOutput类的边缘情况"""
    # 测试空的time_points列表
    output = RequestOutput()
    output.success = True
    output.time_points = []

    metrics = output.get_metrics()
    assert metrics["success"] is False
    assert metrics["error_info"] == "chunk size is less than 2"

    # 测试包含其他属性的情况
    output = RequestOutput()
    output.success = False
    output.uuid = "test_uuid_123"
    output.turn_id = 5
    output.extra_perf_data = {"test_key": "test_value"}

    metrics = output.get_metrics()
    assert metrics["uuid"] == "test_uuid_123"
    assert metrics["turn_id"] == 5
    assert metrics["extra_perf_data"] == {"test_key": "test_value"}


class TestFunctionCallOutput:
    def test_init(self):
        """测试FunctionCallOutput初始化"""
        output = FunctionCallOutput()
        assert output.perf_mode is False
        assert isinstance(output.inference_log, list)
        assert isinstance(output.tool_calls, list)

    def test_init_perf_mode(self):
        """测试FunctionCallOutput性能模式初始化"""
        output = FunctionCallOutput(perf_mode=True)
        assert output.perf_mode is True

    def test_update_extra_details_data_from_text_response(self):
        """测试update_extra_details_data_from_text_response方法"""
        output = FunctionCallOutput()
        text_response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "test content"
                    }
                }
            ]
        }

        output.update_extra_details_data_from_text_response(text_response)

        assert "message" in output.extra_details_data
        assert output.extra_details_data["message"]["role"] == "assistant"

    def test_update_extra_details_data_empty_choices(self):
        """测试空choices的情况"""
        output = FunctionCallOutput()
        text_response = {"choices": []}

        output.update_extra_details_data_from_text_response(text_response)

        assert output.extra_details_data == {}

    def test_update_extra_details_data_no_choices(self):
        """测试没有choices的情况"""
        output = FunctionCallOutput()
        text_response = {}

        output.update_extra_details_data_from_text_response(text_response)

        assert output.extra_details_data == {}

    def test_get_metrics(self):
        """测试get_metrics方法"""
        output = FunctionCallOutput()
        output.success = True
        output.uuid = "test_uuid"
        output.tool_calls = [{"function": "test_func"}]
        output.time_points = [time.perf_counter() - 1, time.perf_counter()]

        metrics = output.get_metrics()

        assert metrics is not None
        assert "tool_calls" in metrics
        assert metrics["tool_calls"] == [{"function": "test_func"}]


class TestLMMOutput:
    def test_init(self):
        """测试LMMOutput初始化"""
        output = LMMOutput()
        assert output.perf_mode is False
        assert isinstance(output.content, list)
        assert len(output.content) == 1
        assert output.content[0] == ""

    def test_init_perf_mode(self):
        """测试LMMOutput性能模式初始化"""
        output = LMMOutput(perf_mode=True)
        assert output.perf_mode is True

    def test_handle_text(self):
        """测试_handle_text方法"""
        output = LMMOutput()
        output.content = ["text content"]

        result = output._handle_text("/test/dir", 0)
        assert result == "text content"

    def test_handle_image(self):
        """测试_handle_image方法"""
        output = LMMOutput()
        output.uuid = "test_uuid"
        test_image = Image.new('RGB', (100, 100), color='red')
        output.content = [test_image]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = output._handle_image(tmpdir, 0)

            assert "image_test_uuid_0.png" in result
            expected_path = os.path.join(tmpdir, "image_test_uuid_0.png")
            assert os.path.exists(expected_path)

    def test_handle_image_overwrite(self):
        """测试_handle_image方法覆盖已存在的文件"""
        output = LMMOutput()
        output.uuid = "test_uuid"
        test_image = Image.new('RGB', (100, 100), color='red')
        output.content = [test_image]

        with tempfile.TemporaryDirectory() as tmpdir:
            result1 = output._handle_image(tmpdir, 0)
            result2 = output._handle_image(tmpdir, 0)

            assert result1 == result2

    def test_get_prediction_single_text(self):
        """测试get_prediction方法，单个文本"""
        output = LMMOutput()
        output.uuid = "test_uuid"
        output.content = ["text content"]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = output.get_prediction(tmpdir)
            assert result == "text content"

    def test_get_prediction_single_image(self):
        """测试get_prediction方法，单个图片"""
        output = LMMOutput()
        output.uuid = "test_uuid"
        test_image = Image.new('RGB', (100, 100), color='red')
        output.content = [test_image]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = output.get_prediction(tmpdir)
            assert "image_test_uuid_0.png" in result

    def test_get_prediction_multiple_items(self):
        """测试get_prediction方法，多个项目"""
        output = LMMOutput()
        output.uuid = "test_uuid"
        test_image = Image.new('RGB', (100, 100), color='red')
        output.content = [test_image, "text content"]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = output.get_prediction(tmpdir)
            assert isinstance(result, list)
            assert len(result) == 2

    def test_get_metrics(self):
        """测试get_metrics方法"""
        output = LMMOutput()
        output.success = True
        output.uuid = "test_uuid"
        output.content = ["test"]
        output.time_points = [time.perf_counter() - 1, time.perf_counter()]

        metrics = output.get_metrics()

        assert metrics is not None
        assert "content" not in metrics
        assert "perf_mode" not in metrics
        assert metrics["uuid"] == "test_uuid"


def test_output_update_extra_perf_data_from_stream_response():
    """测试update_extra_perf_data_from_stream_response方法（默认实现）"""
    output = ConcreteOutput()
    output.update_extra_perf_data_from_stream_response({"test": "data"})
    assert output.extra_perf_data == {}


def test_output_update_extra_perf_data_from_text_response():
    """测试update_extra_perf_data_from_text_response方法（默认实现）"""
    output = ConcreteOutput()
    output.update_extra_perf_data_from_text_response({"test": "data"})
    assert output.extra_perf_data == {}


def test_output_update_extra_details_data_from_stream_response():
    """测试update_extra_details_data_from_stream_response方法（默认实现）"""
    output = ConcreteOutput()
    output.update_extra_details_data_from_stream_response({"test": "data"})
    assert output.extra_details_data == {}


def test_output_update_extra_details_data_from_text_response():
    """测试update_extra_details_data_from_text_response方法（默认实现）"""
    output = ConcreteOutput()
    output.update_extra_details_data_from_text_response({"test": "data"})
    assert output.extra_details_data == {}