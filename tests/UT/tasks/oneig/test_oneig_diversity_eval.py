"""OneIGDiversityEvaluator 单元测试 - 聚焦对外API"""
import unittest
from unittest.mock import MagicMock, patch

from ais_bench.benchmark.tasks.oneig.oneig_diversity_eval import (
    OneIGDiversityEvaluator,
)


DIVERSITY_MODULE = 'ais_bench.benchmark.tasks.oneig.oneig_diversity_eval'
DIVERSITY_CLASS = DIVERSITY_MODULE + '.OneIGDiversityEvaluator'


class TestOneIGDiversityEvaluator(unittest.TestCase):
    """OneIGDiversityEvaluator 对外API测试"""

    def setUp(self):
        with patch(DIVERSITY_MODULE + '.BaseEvaluator.__init__', return_value=None):
            self.evaluator = OneIGDiversityEvaluator.__new__(OneIGDiversityEvaluator)
            self.evaluator.logger = MagicMock()
            self.evaluator.device = 'cuda'
            self.evaluator.oneig_root = ''
            self.evaluator.dreamsim_path = None
            self.evaluator.dreamsim = MagicMock()
            self.evaluator.preprocess = MagicMock()
            processed = MagicMock()
            self.evaluator.preprocess.return_value = processed
            processed.to.return_value = processed
            dist = MagicMock()
            dist.item.return_value = 0.5
            self.evaluator.dreamsim.return_value = dist

    def _setup_score_patches(self, split_images=None, exists=True):
        """Set up common patches for score() tests."""
        if split_images is None:
            split_images = ['/a.jpg', '/b.jpg']
        patches = [
            patch(DIVERSITY_CLASS + '._ensure_model'),
            patch(DIVERSITY_MODULE + '.tempfile.mkdtemp', return_value='/cache'),
            patch(DIVERSITY_MODULE + '.shutil.rmtree'),
            patch(DIVERSITY_MODULE + '.split_image_grid', return_value=split_images),
            patch(DIVERSITY_MODULE + '.os.path.exists', return_value=exists),
            patch('PIL.Image.open'),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])

    def test_diversity_init_defaults(self):
        """测试默认参数初始化"""
        with patch(DIVERSITY_MODULE + '.BaseEvaluator.__init__', return_value=None):
            evaluator = OneIGDiversityEvaluator()
        self.assertEqual(evaluator.device, 'cuda')
        self.assertEqual(evaluator.oneig_root, '')
        self.assertIsNone(evaluator.dreamsim_path)
        self.assertIsNone(evaluator.dreamsim)

    def test_diversity_init_custom_params(self):
        """测试自定义参数初始化"""
        with patch(DIVERSITY_MODULE + '.BaseEvaluator.__init__', return_value=None):
            evaluator = OneIGDiversityEvaluator(
                device='cpu', oneig_root='/oneig/root', dreamsim_path='/ds/cache')
        self.assertEqual(evaluator.device, 'cpu')
        self.assertEqual(evaluator.oneig_root, '/oneig/root')
        self.assertEqual(evaluator.dreamsim_path, '/ds/cache')

    def test_diversity_score_none_test_set(self):
        """测试test_set为None时返回默认值"""
        result = self.evaluator.score(predictions=[], references=[], test_set=None)
        self.assertEqual(result, {'accuracy': 0.0, 'details': []})
        self.evaluator.logger.error.assert_called()

    def test_diversity_score_valid_sample(self):
        """测试有效样本正确计算两两距离"""
        test_set = [{'id': 's1', 'class_item': 'cat', 'image_path': '/img.png'}]
        # 3 张切分图 -> 3 对 (0,1),(0,2),(1,2), 距离 0.1/0.2/0.3 -> avg 0.2
        dist = MagicMock()
        dist.item.side_effect = [0.1, 0.2, 0.3]
        self.evaluator.dreamsim.return_value = dist
        self._setup_score_patches(split_images=['/a.jpg', '/b.jpg', '/c.jpg'])
        result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        # avg = (0.1+0.2+0.3)/3 = 0.2, overall = 20.0
        self.assertAlmostEqual(result['accuracy'], 20.0)
        self.assertAlmostEqual(result['details'][0]['score'], 0.2)
        self.assertEqual(self.evaluator.dreamsim.call_count, 3)

    def test_diversity_score_image_not_found(self):
        """测试图片不存在时跳过"""
        test_set = [{'id': 's1', 'class_item': 'cat', 'image_path': '/missing.png'}]
        self._setup_score_patches(exists=False)
        result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        self.assertEqual(result['accuracy'], 0.0)
        self.assertEqual(result['details'], [])
        self.evaluator.dreamsim.assert_not_called()

    def test_diversity_score_exception_handling(self):
        """测试单样本异常时score=None"""
        test_set = [{'id': 's1', 'class_item': 'cat', 'image_path': '/img.png'}]
        patches = [
            patch(DIVERSITY_CLASS + '._ensure_model'),
            patch(DIVERSITY_MODULE + '.tempfile.mkdtemp', return_value='/cache'),
            patch(DIVERSITY_MODULE + '.shutil.rmtree'),
            patch(DIVERSITY_MODULE + '.split_image_grid',
                  side_effect=RuntimeError("split failed")),
            patch(DIVERSITY_MODULE + '.os.path.exists', return_value=True),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        self.assertEqual(result['accuracy'], 0.0)
        self.evaluator.logger.error.assert_called()

    def test_diversity_score_class_scores_aggregation(self):
        """测试class_scores正确聚合同类样本"""
        test_set = [
            {'id': 's1', 'class_item': 'cat', 'image_path': '/img1.png'},
            {'id': 's2', 'class_item': 'cat', 'image_path': '/img2.png'},
        ]
        dist = MagicMock()
        dist.item.side_effect = [0.4, 0.6]
        self.evaluator.dreamsim.return_value = dist
        self._setup_score_patches()
        result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        class_scores = result['class_scores']
        self.assertIn('cat', class_scores)
        self.assertEqual(len(class_scores['cat']), 2)
        # overall = (0.4 + 0.6) / 2 * 100 = 50.0
        self.assertAlmostEqual(result['accuracy'], 50.0)


if __name__ == '__main__':
    unittest.main()
