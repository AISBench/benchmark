"""OneIGTextEvaluator 单元测试 - 聚焦对外API"""
import unittest
from unittest.mock import MagicMock, patch

from ais_bench.benchmark.tasks.oneig.oneig_text_eval import (
    OneIGTextEvaluator,
)


TEXT_MODULE = 'ais_bench.benchmark.tasks.oneig.oneig_text_eval'


class TestOneIGTextEvaluator(unittest.TestCase):
    """OneIGTextEvaluator 对外API测试"""

    def _make_evaluator(self, judge_model_path="", inferencer=None, mode='EN'):
        with patch(TEXT_MODULE + '.BaseEvaluator.__init__', return_value=None):
            evaluator = OneIGTextEvaluator.__new__(OneIGTextEvaluator)
            evaluator.judge_model_path = judge_model_path
            evaluator.judge_device = "cuda"
            evaluator.judge_dtype = "bfloat16"
            evaluator.judge_batch_size = 8
            evaluator.judge_use_flash_attention = True
            evaluator.judge_seed = 42
            evaluator.mode = mode
            evaluator._inferencer = inferencer
            evaluator.logger = MagicMock()
        return evaluator

    def _setup_score_mocks(self, split_images=None, exists=True):
        """Set up common mocks for score() tests."""
        if split_images is None:
            split_images = ['/tmp/img1.jpg']
        patches = [
            patch(TEXT_MODULE + '.split_image_grid', return_value=split_images),
            patch(TEXT_MODULE + '.rm_error'),
            patch('os.path.exists', return_value=exists),
            patch('tempfile.mkdtemp', return_value='/tmp/fake_cache'),
            patch('shutil.rmtree'),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])

    def test_text_init_defaults(self):
        """测试默认参数初始化"""
        with patch(TEXT_MODULE + '.BaseEvaluator.__init__'):
            evaluator = OneIGTextEvaluator()
        self.assertEqual(evaluator.judge_model_path, "")
        self.assertEqual(evaluator.judge_device, "cuda")
        self.assertEqual(evaluator.judge_dtype, "bfloat16")
        self.assertEqual(evaluator.mode, 'EN')
        self.assertIsNone(evaluator._inferencer)

    def test_text_score_none_test_set(self):
        """测试test_set为None时返回默认值"""
        evaluator = self._make_evaluator(inferencer=MagicMock())
        result = evaluator.score([], [], test_set=None)
        self.assertEqual(result, {'accuracy': 0.0, 'details': []})

    def test_text_score_no_inferencer(self):
        """测试无inferencer时返回默认值"""
        evaluator = self._make_evaluator(inferencer=None)
        result = evaluator.score([], [], test_set=[])
        self.assertEqual(result, {'accuracy': 0.0, 'details': []})

    def test_text_score_perfect_match(self):
        """测试完美匹配时accuracy=100.0"""
        inferencer = MagicMock()
        inferencer.infer_ocr.return_value = ["hello world"]
        evaluator = self._make_evaluator(inferencer=inferencer, mode='EN')
        self._setup_score_mocks()
        test_set = [{
            'id': 's1', 'image_path': '/tmp/img.jpg',
            'expected_text': 'hello world',
        }]
        result = evaluator.score([], [], test_set=test_set)
        self.assertEqual(result['accuracy'], 100.0)
        self.assertIn('ED', result)
        self.assertIn('CR', result)
        self.assertIn('WAC', result)

    def test_text_score_mode_zh_max_edit_distance(self):
        """测试ZH模式下最大编辑距离限制"""
        inferencer = MagicMock()
        inferencer.infer_ocr.return_value = ["b" * 60]
        evaluator = self._make_evaluator(inferencer=inferencer, mode='ZH')
        self._setup_score_mocks()
        test_set = [{
            'id': 's1', 'image_path': '/tmp/img.jpg',
            'expected_text': 'a' * 60,
        }]
        result = evaluator.score([], [], test_set=test_set)
        self.assertEqual(result['accuracy'], 0.0)

    def test_text_score_mode_en_max_edit_distance(self):
        """测试EN模式下最大编辑距离限制"""
        inferencer = MagicMock()
        inferencer.infer_ocr.return_value = ["b" * 60]
        evaluator = self._make_evaluator(inferencer=inferencer, mode='EN')
        self._setup_score_mocks()
        test_set = [{
            'id': 's1', 'image_path': '/tmp/img.jpg',
            'expected_text': 'a' * 60,
        }]
        result = evaluator.score([], [], test_set=test_set)
        self.assertEqual(result['accuracy'], 40.0)

    def test_text_score_image_not_found(self):
        """测试图片不存在时跳过"""
        inferencer = MagicMock()
        evaluator = self._make_evaluator(inferencer=inferencer)
        self._setup_score_mocks(exists=False)
        test_set = [{
            'id': 's1', 'image_path': '/tmp/notexist.jpg',
            'expected_text': 'hello',
        }]
        result = evaluator.score([], [], test_set=test_set)
        inferencer.infer_ocr.assert_not_called()
        self.assertEqual(result['accuracy'], 0.0)
        self.assertEqual(result['details'], [])

    def test_text_score_dynamic_max_new_tokens(self):
        """测试动态max_new_tokens计算"""
        inferencer = MagicMock()
        inferencer.infer_ocr.return_value = ["hello"]
        evaluator = self._make_evaluator(inferencer=inferencer)
        self._setup_score_mocks()
        expected_text = ' '.join(['word'] * 61)
        test_set = [{
            'id': 's1', 'image_path': '/tmp/img.jpg',
            'expected_text': expected_text,
        }]
        evaluator.score([], [], test_set=test_set)
        inferencer.infer_ocr.assert_called_once_with(['/tmp/img1.jpg'], 256)


if __name__ == '__main__':
    unittest.main()
