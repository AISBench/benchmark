"""OneIGStyleEvaluator 单元测试 - 聚焦对外API"""
import unittest
from unittest.mock import MagicMock, patch

from ais_bench.benchmark.tasks.oneig.oneig_style_eval import OneIGStyleEvaluator


STYLE_MODULE = 'ais_bench.benchmark.tasks.oneig.oneig_style_eval'
STYLE_CLASS = STYLE_MODULE + '.OneIGStyleEvaluator'


class TestOneIGStyleEvaluator(unittest.TestCase):
    """OneIGStyleEvaluator 对外API测试"""

    def setUp(self):
        with patch(STYLE_MODULE + '.BaseEvaluator.__init__', return_value=None):
            self.evaluator = OneIGStyleEvaluator.__new__(OneIGStyleEvaluator)
            self.evaluator.logger = MagicMock()
            self.evaluator.oneig_root = ''
            self.evaluator.device = 'cuda'
            self.evaluator.csd_embed_path = None
            self.evaluator.se_embed_path = None
            self.evaluator.encoder_cfg = {}
            self.evaluator.csd_encoder = MagicMock()
            self.evaluator.se_encoder = MagicMock()
            self.evaluator.csd_ref_embeds = None
            self.evaluator.se_ref_embeds = None

    def _setup_score_patches(self, split_images=None, exists=True):
        """Set up common patches for score() tests."""
        if split_images is None:
            split_images = ['/split.jpg']
        patches = [
            patch(STYLE_CLASS + '._ensure_models'),
            patch(STYLE_MODULE + '.tempfile.mkdtemp', return_value='/cache'),
            patch(STYLE_MODULE + '.shutil.rmtree'),
            patch(STYLE_MODULE + '.split_image_grid', return_value=split_images),
            patch(STYLE_MODULE + '.os.path.exists', return_value=exists),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])

    def test_style_init_defaults(self):
        """测试默认参数初始化"""
        with patch(STYLE_MODULE + '.BaseEvaluator.__init__', return_value=None):
            evaluator = OneIGStyleEvaluator()
        self.assertEqual(evaluator.oneig_root, '')
        self.assertEqual(evaluator.device, 'cuda')
        self.assertIsNone(evaluator.csd_embed_path)
        self.assertIsNone(evaluator.se_embed_path)
        self.assertEqual(evaluator.encoder_cfg, {})
        self.assertIsNone(evaluator.csd_encoder)
        self.assertIsNone(evaluator.se_encoder)

    def test_style_init_custom_params(self):
        """测试自定义参数初始化"""
        cfg = {'csd_model_path': '/csd', 'clip_model_path': '/clip', 'se_model_path': '/se'}
        with patch(STYLE_MODULE + '.BaseEvaluator.__init__', return_value=None):
            evaluator = OneIGStyleEvaluator(
                oneig_root='/oneig/root', device='cpu',
                csd_embed_path='/csd_emb.pt', se_embed_path='/se_emb.pt',
                encoder_cfg=cfg,
            )
        self.assertEqual(evaluator.oneig_root, '/oneig/root')
        self.assertEqual(evaluator.device, 'cpu')
        self.assertEqual(evaluator.csd_embed_path, '/csd_emb.pt')
        self.assertEqual(evaluator.se_embed_path, '/se_emb.pt')
        self.assertEqual(evaluator.encoder_cfg, cfg)

    def test_style_score_none_test_set(self):
        """测试test_set为None时返回默认值"""
        result = self.evaluator.score(predictions=[], references=[], test_set=None)
        self.assertEqual(result, {'accuracy': 0.0, 'details': []})
        self.evaluator.logger.error.assert_called()

    def test_style_score_valid_sample(self):
        """测试有效样本正确计算"""
        self.evaluator.csd_encoder = MagicMock()
        self.evaluator.se_encoder = MagicMock()
        test_set = [{'id': 's1', 'image_path': '/img.png', 'style_label': 'impressionism'}]
        self._setup_score_patches()
        with patch.object(self.evaluator, '_compute_similarity', side_effect=[0.8, 0.6]):
            result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        # max_style_score = (0.8 + 0.6) / 2 = 0.7, overall = 70.0
        self.assertAlmostEqual(result['accuracy'], 70.0)
        self.assertAlmostEqual(result['details'][0]['score'], 0.7)

    def test_style_score_empty_style_label(self):
        """测试style_label为空时跳过"""
        test_set = [{'id': 's1', 'image_path': '/img.png', 'style_label': ''}]
        self._setup_score_patches()
        result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        self.assertEqual(result['accuracy'], 0.0)
        self.assertEqual(result['details'], [])
        self.evaluator.csd_encoder.get_style_embedding.assert_not_called()

    def test_style_score_image_not_found(self):
        """测试image_path不存在时跳过"""
        test_set = [{'id': 's1', 'image_path': '/missing.png', 'style_label': 'impressionism'}]
        self._setup_score_patches(exists=False)
        result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        self.assertEqual(result['accuracy'], 0.0)

    def test_style_score_exception_handling(self):
        """测试单样本异常时score=None"""
        test_set = [{'id': 's1', 'image_path': '/img.png', 'style_label': 'impressionism'}]
        patches = [
            patch(STYLE_CLASS + '._ensure_models'),
            patch(STYLE_MODULE + '.tempfile.mkdtemp', return_value='/cache'),
            patch(STYLE_MODULE + '.shutil.rmtree'),
            patch(STYLE_MODULE + '.split_image_grid',
                  side_effect=RuntimeError("split failed")),
            patch(STYLE_MODULE + '.os.path.exists', return_value=True),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        self.assertEqual(result['accuracy'], 0.0)
        self.evaluator.logger.error.assert_called()

    def test_style_score_style_dict_aggregation(self):
        """测试style_dict正确聚合多个相同样本的分数"""
        self.evaluator.csd_encoder = MagicMock()
        self.evaluator.se_encoder = MagicMock()
        test_set = [
            {'id': 's1', 'image_path': '/img1.png', 'style_label': 'impressionism'},
            {'id': 's2', 'image_path': '/img2.png', 'style_label': 'impressionism'},
        ]
        self._setup_score_patches()
        with patch.object(self.evaluator, '_compute_similarity',
                          side_effect=[0.8, 0.6, 0.4, 0.6]):
            result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        # s1: (0.8+0.6)/2 = 0.7; s2: (0.4+0.6)/2 = 0.5
        style_scores = result['style_scores']
        self.assertIn('impressionism', style_scores)
        self.assertEqual(len(style_scores['impressionism']), 2)
        # overall = (0.7 + 0.5) / 2 * 100 = 60.0
        self.assertAlmostEqual(result['accuracy'], 60.0)

    def test_style_score_style_label_normalization(self):
        """测试style_label做lower().replace(' ','_')归一化"""
        self.evaluator.csd_encoder = MagicMock()
        self.evaluator.se_encoder = MagicMock()
        test_set = [{'id': 's1', 'image_path': '/img.png', 'style_label': 'Pop Art'}]
        self._setup_score_patches()
        with patch.object(self.evaluator, '_compute_similarity', side_effect=[0.8, 0.8]):
            result = self.evaluator.score(predictions=[], references=[], test_set=test_set)
        self.assertIn('pop_art', result['style_scores'])

    def test_style_ensure_models_already_loaded(self):
        """测试编码器已加载时直接返回"""
        self.evaluator.csd_encoder = MagicMock()
        self.evaluator.se_encoder = MagicMock()
        self.evaluator._ensure_models()
        self.evaluator.logger.info.assert_not_called()

    def test_style_ensure_models_import_error(self):
        """测试导入失败时抛出ImportError"""
        self.evaluator.csd_encoder = None
        self.evaluator.se_encoder = None
        with patch(STYLE_MODULE + '.ensure_oneig_path'), \
             patch('builtins.__import__', side_effect=ImportError("no module")):
            with self.assertRaises(ImportError):
                self.evaluator._ensure_models()
        self.evaluator.logger.error.assert_called()

    def test_style_compute_similarity_valid_tensor(self):
        """测试_compute_similarity使用有效tensor计算"""
        import torch
        embed = MagicMock(spec=torch.Tensor)
        embed.dim.return_value = 1
        embed.unsqueeze.return_value = embed
        ref_embed = MagicMock(spec=torch.Tensor)
        self.evaluator.csd_ref_embeds = {'impressionism': ref_embed}
        with patch('torch.max',
                   return_value=MagicMock(item=lambda: 0.85)):
            result = self.evaluator._compute_similarity(embed, 'impressionism', 'csd')
        self.assertEqual(result, 0.85)

    def test_style_compute_similarity_none_ref(self):
        """测试ref_embeds为None时返回0.0"""
        self.evaluator.csd_ref_embeds = None
        result = self.evaluator._compute_similarity(
            MagicMock(), 'impressionism', 'csd')
        self.assertEqual(result, 0.0)

    def test_style_compute_similarity_style_not_found(self):
        """测试style_label不在ref_embeds中时返回0.0"""
        self.evaluator.csd_ref_embeds = {'other_style': MagicMock()}
        result = self.evaluator._compute_similarity(
            MagicMock(), 'impressionism', 'csd')
        self.assertEqual(result, 0.0)
        self.evaluator.logger.warning.assert_called()


if __name__ == '__main__':
    unittest.main()
