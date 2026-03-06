import sys
import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import json
import base64
from io import BytesIO
from PIL import Image

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from ais_bench.benchmark.datasets.utils import lmm_judge
from ais_bench.benchmark.datasets.utils.lmm_judge import (
    get_lmm_point_list,
    LMMImgJDGDataset,
    ImgSCJDGDataset,
    ImgPQJDGDataset,
    LMMJudgeImageEditEvaluator
)


class TestGetLmmPointList:
    def test_get_lmm_point_list_valid(self):
        """测试提取有效的列表"""
        result = get_lmm_point_list("The answer is [1, 2, 3]")
        assert result == "[1, 2, 3]"

    def test_get_lmm_point_list_single(self):
        """测试提取单个数字的列表"""
        result = get_lmm_point_list("Result: [5]")
        assert result == "[5]"

    def test_get_lmm_point_list_with_spaces(self):
        """测试提取带空格的列表"""
        result = get_lmm_point_list("Scores: [ 1 , 2 , 3 ]")
        assert result == "[ 1 , 2 , 3 ]"

    def test_get_lmm_point_list_no_match(self):
        """测试没有匹配时返回空列表"""
        result = get_lmm_point_list("No list here")
        assert result == "[]"

    def test_get_lmm_point_list_empty(self):
        """测试空字符串"""
        result = get_lmm_point_list("")
        assert result == "[]"

    def test_get_lmm_point_list_multiple(self):
        """测试多个数字的列表"""
        result = get_lmm_point_list("Points: [10, 20, 30, 40]")
        assert result == "[10, 20, 30, 40]"


class TestLMMImgJDGDataset:
    def test_load_from_predictions_file_not_exists(self):
        """测试文件不存在的情况"""
        ds = LMMImgJDGDataset.__new__(LMMImgJDGDataset)
        ds.task_state_manager = None
        
        with patch('os.path.exists', return_value=False):
            result = ds._load_from_predictions('/test/nonexistent.jsonl')
            assert result == []

    def test_load_from_predictions_success(self):
        """测试成功加载predictions"""
        # 创建测试图片并转换为base64
        test_image = Image.new('RGB', (100, 100), color='red')
        buffered = BytesIO()
        test_image.save(buffered, format="PNG")
        expected_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        mock_preds = [
            {"id": 1, "prediction": "image1.png"},
            {"id": 0, "prediction": "image2.png"}
        ]
        
        ds = LMMImgJDGDataset.__new__(LMMImgJDGDataset)
        ds.task_state_manager = MagicMock()
        ds.update_task_state = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试图片文件
            img_path1 = os.path.join(tmpdir, "image1.png")
            img_path2 = os.path.join(tmpdir, "image2.png")
            test_image.save(img_path1)
            test_image.save(img_path2)
            
            # 创建predictions文件
            pred_file = os.path.join(tmpdir, "predictions.jsonl")
            
            with patch('os.path.exists', return_value=True):
                with patch.object(lmm_judge, 'load_jsonl', return_value=mock_preds):
                    result = ds._load_from_predictions(pred_file)
                    
                    assert len(result) == 2
                    assert result[0]["id"] == 0
                    assert result[1]["id"] == 1
                    # 验证图片被转换为base64
                    assert result[0]["prediction"] == expected_base64

    def test_load_from_predictions_with_nonexistent_image(self):
        """测试图片文件不存在的情况"""
        mock_preds = [
            {"id": 0, "prediction": "nonexistent.png"}
        ]
        
        ds = LMMImgJDGDataset.__new__(LMMImgJDGDataset)
        ds.task_state_manager = MagicMock()
        ds.update_task_state = MagicMock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_file = os.path.join(tmpdir, "predictions.jsonl")
            
            with patch('os.path.exists') as mock_exists:
                # 文件存在但图片不存在
                mock_exists.side_effect = lambda path: path.endswith('.jsonl') or path == pred_file
                
                with patch.object(lmm_judge, 'load_jsonl', return_value=mock_preds):
                    result = ds._load_from_predictions(pred_file)
                    
                    assert len(result) == 1
                    assert result[0]["prediction"] == "nonexistent.png"


class TestImgSCJDGDataset:
    def test_modify_dataset_item(self):
        """测试_modify_dataset_item方法"""
        from ais_bench.benchmark.utils.prompt import AIS_CONTENT_TAG, AIS_TEXT_START, AIS_IMAGE_START
        
        ds = ImgSCJDGDataset.__new__(ImgSCJDGDataset)
        ds.logger = MagicMock()
        
        question = "What is in the image?"
        org_image_url = "original_base64_string"
        pred_image_url = "prediction_base64_string"
        
        dataset_item = {
            "content": AIS_TEXT_START + question + AIS_CONTENT_TAG + AIS_IMAGE_START + org_image_url + AIS_CONTENT_TAG
        }
        pred_item = {"prediction": pred_image_url}
        
        ds._modify_dataset_item(dataset_item, pred_item)
        
        # 验证content被正确修改
        assert AIS_TEXT_START + question + AIS_CONTENT_TAG in dataset_item["content"]
        assert AIS_IMAGE_START + org_image_url + AIS_CONTENT_TAG in dataset_item["content"]
        assert AIS_IMAGE_START + pred_image_url + AIS_CONTENT_TAG in dataset_item["content"]


class TestImgPQJDGDataset:
    def test_modify_dataset_item(self):
        """测试_modify_dataset_item方法"""
        from ais_bench.benchmark.utils.prompt import AIS_CONTENT_TAG, AIS_TEXT_START, AIS_IMAGE_START
        
        ds = ImgPQJDGDataset.__new__(ImgPQJDGDataset)
        ds.logger = MagicMock()
        
        question = "Describe the image?"
        org_image_url = "original_base64_string"
        pred_image_url = "prediction_base64_string"
        
        dataset_item = {
            "content": AIS_TEXT_START + question + AIS_CONTENT_TAG + AIS_IMAGE_START + org_image_url + AIS_CONTENT_TAG
        }
        pred_item = {"prediction": pred_image_url}
        
        ds._modify_dataset_item(dataset_item, pred_item)
        
        # 验证content被正确修改（PQ版本不包含原始图片）
        assert AIS_TEXT_START + question + AIS_CONTENT_TAG in dataset_item["content"]
        assert AIS_IMAGE_START + pred_image_url + AIS_CONTENT_TAG in dataset_item["content"]
        # PQ版本不应该包含原始图片URL
        assert org_image_url not in dataset_item["content"]


class TestLMMJudgeImageEditEvaluator:
    def test_init_default_metric(self):
        """测试默认metric初始化"""
        evaluator = LMMJudgeImageEditEvaluator()
        assert evaluator.metric == "SC"
        assert evaluator.point_key_list == ["editing success", "over editing"]

    def test_init_pq_metric(self):
        """测试PQ metric初始化"""
        evaluator = LMMJudgeImageEditEvaluator(metric="PQ")
        assert evaluator.metric == "PQ"
        assert evaluator.point_key_list == ["naturalness", "artifacts"]

    def test_score_success(self):
        """测试score方法成功情况"""
        evaluator = LMMJudgeImageEditEvaluator(metric="SC")
        predictions = ["[5, 4]", "[3, 2]", "[4, 5]"]
        references = ["ref1", "ref2", "ref3"]
        
        result = evaluator.score(predictions, references)
        
        assert "SC" in result
        assert "details" in result
        assert len(result["details"]) == 3
        # min(5,4)=4, min(3,2)=2, min(4,5)=4, average = (4+2+4)/3 = 3.33
        assert result["SC"] == pytest.approx(10/3, rel=1e-2)

    def test_score_pq_metric(self):
        """测试PQ metric的score方法"""
        evaluator = LMMJudgeImageEditEvaluator(metric="PQ")
        predictions = ["[4, 5]", "[3, 3]"]
        references = ["ref1", "ref2"]
        
        result = evaluator.score(predictions, references)
        
        assert "PQ" in result
        assert len(result["details"]) == 2
        # min(4,5)=4, min(3,3)=3, average = (4+3)/2 = 3.5
        assert result["PQ"] == pytest.approx(3.5, rel=1e-2)

    def test_score_length_mismatch(self):
        """测试长度不匹配的情况"""
        evaluator = LMMJudgeImageEditEvaluator()
        predictions = ["[1, 2]"]
        references = ["ref1", "ref2"]
        
        result = evaluator.score(predictions, references)
        
        assert "error" in result

    def test_score_non_string_predictions(self):
        """测试predictions不是字符串的情况"""
        evaluator = LMMJudgeImageEditEvaluator()
        predictions = [[1, 2], [3, 4]]
        references = ["ref1", "ref2"]
        
        result = evaluator.score(predictions, references)
        
        assert "error" in result

    def test_score_invalid_prediction_format(self):
        """测试prediction格式错误的情况"""
        evaluator = LMMJudgeImageEditEvaluator(metric="SC")
        # SC需要2个分数，但这里给出3个
        predictions = ["[1, 2, 3]"]
        references = ["ref1"]
        
        result = evaluator.score(predictions, references)
        
        assert "details" in result
        assert result["details"][0]["eval_success"] is False
        assert "failed reason" in result["details"][0]

    def test_score_empty_predictions(self):
        """测试空predictions"""
        evaluator = LMMJudgeImageEditEvaluator()
        predictions = []
        references = []
        
        result = evaluator.score(predictions, references)
        
        assert "SC" in result
        assert result["SC"] == 0.0
        assert "details" in result

    def test_score_detail_structure(self):
        """测试details结构正确"""
        evaluator = LMMJudgeImageEditEvaluator(metric="SC")
        predictions = ["[5, 4]"]
        references = ["ref1"]
        
        result = evaluator.score(predictions, references)
        
        detail = result["details"][0]
        assert detail["eval_success"] is True
        assert "pred" in detail
        assert detail["pred"]["editing success"] == 5
        assert detail["pred"]["over editing"] == 4
        assert detail["org_uuid"] == "ref1"
