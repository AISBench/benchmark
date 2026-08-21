"""OneIGAlignmentEvaluator 单元测试 - 聚焦对外API"""
import unittest
from unittest.mock import MagicMock, patch

from ais_bench.benchmark.tasks.oneig.oneig_alignment_eval import (
    OneIGAlignmentEvaluator,
)


ALIGN_MODULE = 'ais_bench.benchmark.tasks.oneig.oneig_alignment_eval'


class TestOneIGAlignmentEvaluator(unittest.TestCase):
    """OneIGAlignmentEvaluator 对外API测试"""

    def _make_evaluator(self, judge_model_path="", inferencer=None):
        with patch(ALIGN_MODULE + '.BaseEvaluator.__init__', return_value=None):
            evaluator = OneIGAlignmentEvaluator.__new__(OneIGAlignmentEvaluator)
            evaluator.judge_model_path = judge_model_path
            evaluator.judge_device = "cuda"
            evaluator.judge_dtype = "bfloat16"
            evaluator.judge_batch_size = 8
            evaluator.judge_use_flash_attention = True
            evaluator.judge_seed = 42
            evaluator._inferencer = inferencer
            evaluator.logger = MagicMock()
        return evaluator

    def _setup_score_mocks(self, split_images=None, exists=True):
        """Set up common mocks for score() tests."""
        if split_images is None:
            split_images = ['/tmp/img1.jpg']
        patches = [
            patch(ALIGN_MODULE + '.split_image_grid', return_value=split_images),
            patch(ALIGN_MODULE + '.rm_error'),
            patch('os.path.exists', return_value=exists),
            patch('tempfile.mkdtemp', return_value='/tmp/fake_cache'),
            patch('shutil.rmtree'),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])

    def test_alignment_init_defaults(self):
        """测试默认参数初始化"""
        with patch(ALIGN_MODULE + '.BaseEvaluator.__init__'):
            evaluator = OneIGAlignmentEvaluator()
        self.assertEqual(evaluator.judge_model_path, "")
        self.assertEqual(evaluator.judge_device, "cuda")
        self.assertEqual(evaluator.judge_dtype, "bfloat16")
        self.assertEqual(evaluator.judge_batch_size, 8)
        self.assertTrue(evaluator.judge_use_flash_attention)
        self.assertEqual(evaluator.judge_seed, 42)
        self.assertIsNone(evaluator._inferencer)

    def test_alignment_init_custom_params(self):
        """测试自定义参数初始化"""
        with patch(ALIGN_MODULE + '.BaseEvaluator.__init__'):
            evaluator = OneIGAlignmentEvaluator(
                judge_model_path="/path/to/model",
                judge_device="cpu",
                judge_dtype="float16",
                judge_batch_size=4,
                judge_use_flash_attention=False,
                judge_seed=123,
            )
        self.assertEqual(evaluator.judge_model_path, "/path/to/model")
        self.assertEqual(evaluator.judge_device, "cpu")
        self.assertEqual(evaluator.judge_dtype, "float16")
        self.assertEqual(evaluator.judge_batch_size, 4)
        self.assertFalse(evaluator.judge_use_flash_attention)
        self.assertEqual(evaluator.judge_seed, 123)

    def test_alignment_score_none_test_set(self):
        """测试test_set为None时返回默认值"""
        evaluator = self._make_evaluator(inferencer=MagicMock())
        result = evaluator.score([], [], test_set=None)
        self.assertEqual(result, {'accuracy': 0.0, 'details': []})

    def test_alignment_score_no_inferencer(self):
        """测试无inferencer时返回默认值"""
        evaluator = self._make_evaluator(inferencer=None)
        result = evaluator.score([], [], test_set=[])
        self.assertEqual(result, {'accuracy': 0.0, 'details': []})

    def test_alignment_score_all_yes(self):
        """测试全部Yes时accuracy=100.0"""
        inferencer = MagicMock()
        inferencer.infer_semantic.return_value = ["Yes"]
        evaluator = self._make_evaluator(inferencer=inferencer)
        self._setup_score_mocks()
        test_set = [{
            'id': 's1', 'class_item': 'cat', 'image_path': '/tmp/img.jpg',
            'question': '{"1": "q1"}', 'dependency': '{}',
        }]
        result = evaluator.score([], [], test_set=test_set)
        self.assertEqual(result['accuracy'], 100.0)
        self.assertEqual(len(result['details']), 1)
        self.assertEqual(result['details'][0]['class_item'], 'cat')

    def test_alignment_score_mixed_answers(self):
        """测试混合答案时正确计算accuracy"""
        inferencer = MagicMock()
        inferencer.infer_semantic.side_effect = [["Yes"], ["No"]]
        evaluator = self._make_evaluator(inferencer=inferencer)
        self._setup_score_mocks()
        test_set = [{
            'id': 's1', 'class_item': 'cat', 'image_path': '/tmp/img.jpg',
            'question': '{"1": "q1", "2": "q2"}', 'dependency': '{}',
        }]
        result = evaluator.score([], [], test_set=test_set)
        self.assertEqual(result['accuracy'], 50.0)

    def test_alignment_score_dependency_filter(self):
        """测试依赖过滤：父问题为No时跳过子问题"""
        inferencer = MagicMock()
        inferencer.infer_semantic.side_effect = [["No"], ["Yes"]]
        evaluator = self._make_evaluator(inferencer=inferencer)
        self._setup_score_mocks()
        test_set = [{
            'id': 's1', 'class_item': 'cat', 'image_path': '/tmp/img.jpg',
            'question': '{"1": "q1", "2": "q2"}', 'dependency': '{"2": [1]}',
        }]
        result = evaluator.score([], [], test_set=test_set)
        # q1=No -> q2 skipped, only q1 counted, accuracy=0.0
        self.assertEqual(result['accuracy'], 0.0)

    def test_alignment_score_image_not_found(self):
        """测试图片不存在时跳过"""
        inferencer = MagicMock()
        evaluator = self._make_evaluator(inferencer=inferencer)
        self._setup_score_mocks(exists=False)
        test_set = [{
            'id': 's1', 'class_item': 'cat', 'image_path': '/tmp/notexist.jpg',
            'question': '{"1": "q1"}', 'dependency': '{}',
        }]
        result = evaluator.score([], [], test_set=test_set)
        inferencer.infer_semantic.assert_not_called()
        self.assertEqual(result['accuracy'], 0.0)
        self.assertEqual(result['details'], [])

    def test_alignment_ensure_inferencer_loads(self):
        """测试_ensure_inferencer正确加载推理器"""
        evaluator = self._make_evaluator(judge_model_path='/path/to/model')
        with patch(ALIGN_MODULE + '.OneIGJudgeInferencer') as mock_cls:
            evaluator._ensure_inferencer()
            mock_cls.assert_called_once()
            self.assertIsNotNone(evaluator._inferencer)

    def test_alignment_score_dict_questions(self):
        """测试question为dict类型时正确解析"""
        inferencer = MagicMock()
        inferencer.infer_semantic.return_value = ["Yes"]
        evaluator = self._make_evaluator(inferencer=inferencer)
        self._setup_score_mocks()
        test_set = [{
            'id': 's1', 'class_item': 'cat', 'image_path': '/tmp/img.jpg',
            'question': {1: 'q1'}, 'dependency': {},
        }]
        result = evaluator.score([], [], test_set=test_set)
        self.assertEqual(result['accuracy'], 100.0)

    def test_alignment_score_invalid_json(self):
        """测试question为无效JSON时优雅处理"""
        inferencer = MagicMock()
        evaluator = self._make_evaluator(inferencer=inferencer)
        self._setup_score_mocks()
        test_set = [{
            'id': 's1', 'class_item': 'cat', 'image_path': '/tmp/img.jpg',
            'question': 'invalid json', 'dependency': 'invalid json',
        }]
        result = evaluator.score([], [], test_set=test_set)
        self.assertEqual(result['accuracy'], 0.0)


if __name__ == '__main__':
    unittest.main()
