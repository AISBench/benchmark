"""OneIGReasoningEvaluator 单元测试 - 聚焦对外API"""
import unittest
from unittest.mock import MagicMock, patch

from ais_bench.benchmark.tasks.oneig.oneig_reasoning_eval import (
    OneIGReasoningEvaluator,
)


REASON_MODULE = 'ais_bench.benchmark.tasks.oneig.oneig_reasoning_eval'


class TestOneIGReasoningEvaluator(unittest.TestCase):
    """OneIGReasoningEvaluator 对外API测试"""

    def setUp(self):
        with patch(REASON_MODULE + '.BaseEvaluator.__init__', return_value=None):
            self.evaluator = OneIGReasoningEvaluator.__new__(OneIGReasoningEvaluator)
            self.evaluator.logger = MagicMock()
            self.evaluator.oneig_root = ''
            self.evaluator.llm2clip_cfg = {}
            self.evaluator.device = 'cuda'
            self.evaluator.model = MagicMock()

    def _setup_score_patches(self, split_images=None, exists=True):
        """Set up common patches for score() tests."""
        if split_images is None:
            split_images = ['/split.jpg']
        patches = [
            patch(REASON_MODULE + '.tempfile.mkdtemp', return_value='/cache'),
            patch(REASON_MODULE + '.shutil.rmtree'),
            patch(REASON_MODULE + '.split_image_grid', return_value=split_images),
            patch(REASON_MODULE + '.os.path.exists', return_value=exists),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])

    def test_reasoning_init_defaults(self):
        """测试默认参数初始化"""
        with patch(REASON_MODULE + '.BaseEvaluator.__init__', return_value=None):
            evaluator = OneIGReasoningEvaluator()
        self.assertEqual(evaluator.oneig_root, '')
        self.assertEqual(evaluator.llm2clip_cfg, {})
        self.assertEqual(evaluator.device, 'cuda')
        self.assertIsNone(evaluator.model)

    def test_reasoning_init_custom_params(self):
        """测试自定义参数初始化"""
        cfg = {'processor_model': '/p', 'clip_model': '/c', 'llm_model': '/l'}
        with patch(REASON_MODULE + '.BaseEvaluator.__init__', return_value=None):
            evaluator = OneIGReasoningEvaluator(
                oneig_root='/oneig/root', llm2clip_cfg=cfg, device='cpu')
        self.assertEqual(evaluator.oneig_root, '/oneig/root')
        self.assertEqual(evaluator.llm2clip_cfg, cfg)
        self.assertEqual(evaluator.device, 'cpu')

    def test_reasoning_score_none_test_set(self):
        """测试test_set为None时返回默认值"""
        result = self.evaluator.score(predictions=[], references=[], test_set=None)
        self.assertEqual(result, {'accuracy': 0.0, 'details': []})
        self.evaluator.logger.error.assert_called()

    def test_reasoning_score_empty_gt_answer(self):
        """测试gt_answer为空时score=None"""
        test_set = [{'id': 's1', 'image_path': '/img.png', 'gt_answer': ''}]
        self._setup_score_patches()
        result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        self.assertIsNone(result['details'][0]['score'])
        self.evaluator.model.text_img_similarity_score.assert_not_called()

    def test_reasoning_score_model_returns_valid_scores(self):
        """测试模型返回有效分数时正确计算"""
        self.evaluator.model.text_img_similarity_score.return_value = [0.8, 0.6]
        test_set = [{'id': 's1', 'image_path': '/img.png', 'gt_answer': 'answer'}]
        self._setup_score_patches(split_images=['/a.jpg', '/b.jpg'])
        result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        # avg = (0.8 + 0.6) / 2 = 0.7, overall = 70.0
        self.assertAlmostEqual(result['details'][0]['score'], 0.7)
        self.assertAlmostEqual(result['accuracy'], 70.0)

    def test_reasoning_score_model_returns_none(self):
        """测试模型返回None时score=None"""
        self.evaluator.model.text_img_similarity_score.return_value = None
        test_set = [{'id': 's1', 'image_path': '/img.png', 'gt_answer': 'answer'}]
        self._setup_score_patches()
        result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        self.assertIsNone(result['details'][0]['score'])
        self.assertEqual(result['accuracy'], 0.0)

    def test_reasoning_score_model_exception(self):
        """测试模型推理异常时score=None"""
        self.evaluator.model.text_img_similarity_score.side_effect = RuntimeError("infer failed")
        test_set = [{'id': 's1', 'image_path': '/img.png', 'gt_answer': 'answer'}]
        self._setup_score_patches()
        result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        self.assertIsNone(result['details'][0]['score'])
        self.assertEqual(result['accuracy'], 0.0)
        self.evaluator.logger.error.assert_called()

    def test_reasoning_score_partial_valid(self):
        """测试部分样本有效时正确计算均值"""
        self.evaluator.model.text_img_similarity_score.return_value = [0.6]
        test_set = [
            {'id': 's1', 'image_path': '/img1.png', 'gt_answer': 'answer'},
            {'id': 's2', 'image_path': '/img2.png', 'gt_answer': ''},
        ]
        self._setup_score_patches()
        result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        # 仅 s1 有效 (0.6), overall = 60.0
        self.assertAlmostEqual(result['accuracy'], 60.0)
        self.assertAlmostEqual(result['details'][0]['score'], 0.6)
        self.assertIsNone(result['details'][1]['score'])


if __name__ == '__main__':
    unittest.main()
