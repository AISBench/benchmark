"""Unit tests for oneig_eval_utils - focus on OneIGJudgeInferencer public API."""
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import torch


@contextmanager
def _fake_oneig_imports():
    """Inject mock transformers and qwen_vl_utils into sys.modules."""
    mock_transformers = MagicMock()
    mock_qwen_vl = MagicMock()
    prev = {}
    for name in ('transformers', 'qwen_vl_utils'):
        prev[name] = sys.modules.pop(name, None)
    sys.modules['transformers'] = mock_transformers
    sys.modules['qwen_vl_utils'] = mock_qwen_vl
    try:
        yield mock_transformers, mock_qwen_vl
    finally:
        for name in ('transformers', 'qwen_vl_utils'):
            sys.modules.pop(name, None)
        for name, mod in prev.items():
            if mod is not None:
                sys.modules[name] = mod


class TestOneIGDTypeMap(unittest.TestCase):
    """Tests for ONEIG_DTYPE_MAP constant."""

    def test_dtype_map_contains_bfloat16(self):
        from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
            ONEIG_DTYPE_MAP,
        )
        self.assertIn('bfloat16', ONEIG_DTYPE_MAP)
        self.assertEqual(ONEIG_DTYPE_MAP['bfloat16'], torch.bfloat16)

    def test_dtype_map_contains_float16(self):
        from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
            ONEIG_DTYPE_MAP,
        )
        self.assertIn('float16', ONEIG_DTYPE_MAP)
        self.assertEqual(ONEIG_DTYPE_MAP['float16'], torch.float16)


class FakeInputs(dict):
    """A dict-like object that also supports attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class TestOneIGJudgeInferencer(unittest.TestCase):
    """Tests for OneIGJudgeInferencer public API."""

    @patch('torch.manual_seed')
    @patch('torch.cuda.manual_seed_all')
    def test_judge_init_loads_model(self, mock_cuda_seed, mock_seed):
        """__init__ should load model and processor."""
        from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
            OneIGJudgeInferencer,
        )
        with _fake_oneig_imports() as (mock_transformers, _):
            inferencer = OneIGJudgeInferencer(model_path='test/model', device='cpu')
            mock_transformers.Qwen3VLForConditionalGeneration.from_pretrained.assert_called_once()
            mock_transformers.AutoProcessor.from_pretrained.assert_called_once()
            self.assertIsNotNone(inferencer.model)
            self.assertIsNotNone(inferencer.processor)

    @patch('torch.manual_seed')
    @patch('torch.cuda.manual_seed_all')
    def test_judge_init_sets_seed(self, mock_cuda_seed, mock_seed):
        """__init__ should set random seeds."""
        from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
            OneIGJudgeInferencer,
        )
        with _fake_oneig_imports():
            OneIGJudgeInferencer(seed=42, device='cpu')
        mock_seed.assert_called_once_with(42)
        mock_cuda_seed.assert_called_once_with(42)

    @patch('torch.manual_seed')
    @patch('torch.cuda.manual_seed_all')
    def test_judge_batch_inference_single_batch(self, mock_cuda_seed, mock_seed):
        """batch_inference should process single batch correctly."""
        from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
            OneIGJudgeInferencer,
        )
        with _fake_oneig_imports() as (mock_transformers, mock_qwen_vl):
            inferencer = OneIGJudgeInferencer(batch_size=8, device='cpu')

            fake_inputs = FakeInputs(input_ids=[[1, 2, 3]])
            inferencer.processor.return_value.to.return_value = fake_inputs
            inferencer.processor.batch_decode.return_value = ['output1']
            inferencer.model.generate.return_value = [[1, 2, 3, 4]]
            mock_qwen_vl.process_vision_info.return_value = (None, None)

            messages = [[{'role': 'user', 'content': 'hi'}]]
            result = inferencer.batch_inference(messages)

            self.assertEqual(result, ['output1'])
            inferencer.model.generate.assert_called_once()

    @patch('torch.manual_seed')
    @patch('torch.cuda.manual_seed_all')
    def test_judge_batch_inference_multi_batch(self, mock_cuda_seed, mock_seed):
        """batch_inference should handle multiple batches."""
        from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
            OneIGJudgeInferencer,
        )
        with _fake_oneig_imports() as (mock_transformers, mock_qwen_vl):
            inferencer = OneIGJudgeInferencer(batch_size=2, device='cpu')

            fake_inputs_1 = FakeInputs(input_ids=[[1, 2, 3], [1, 2, 3]])
            fake_inputs_2 = FakeInputs(input_ids=[[1, 2, 3], [1, 2, 3]])
            fake_inputs_3 = FakeInputs(input_ids=[[1, 2, 3]])
            inferencer.processor.return_value.to.side_effect = [
                fake_inputs_1, fake_inputs_2, fake_inputs_3,
            ]
            inferencer.processor.batch_decode.side_effect = [
                ['out1', 'out2'], ['out3', 'out4'], ['out5'],
            ]
            inferencer.model.generate.side_effect = [
                [[1, 2, 3, 4], [1, 2, 3, 4]],
                [[1, 2, 3, 4], [1, 2, 3, 4]],
                [[1, 2, 3, 4]],
            ]
            mock_qwen_vl.process_vision_info.return_value = (None, None)

            messages = [[{'role': 'user', 'content': 'hi'}] for _ in range(5)]
            result = inferencer.batch_inference(messages)

            self.assertEqual(result, ['out1', 'out2', 'out3', 'out4', 'out5'])
            self.assertEqual(inferencer.model.generate.call_count, 3)

    @patch('torch.manual_seed')
    @patch('torch.cuda.manual_seed_all')
    def test_judge_infer_semantic_constructs_messages(self, mock_cuda_seed, mock_seed):
        """infer_semantic should construct correct messages."""
        from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
            OneIGJudgeInferencer,
        )
        with _fake_oneig_imports():
            inferencer = OneIGJudgeInferencer(device='cpu')
        with patch.object(inferencer, 'batch_inference', return_value=['Yes']) as mock_batch:
            result = inferencer.infer_semantic(['/img1.png', '/img2.png'], 'Is this aligned?')
            self.assertEqual(result, ['Yes'])
            called_messages = mock_batch.call_args[0][0]
            self.assertEqual(len(called_messages), 2)
            content = called_messages[0][0]['content']
            self.assertEqual(content[0]['type'], 'image')
            self.assertEqual(content[0]['image'], '/img1.png')
            self.assertIn('Is this aligned?', content[1]['text'])

    @patch('torch.manual_seed')
    @patch('torch.cuda.manual_seed_all')
    def test_judge_infer_ocr_constructs_messages(self, mock_cuda_seed, mock_seed):
        """infer_ocr should construct correct messages."""
        from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
            OneIGJudgeInferencer,
        )
        with _fake_oneig_imports():
            inferencer = OneIGJudgeInferencer(device='cpu')
        with patch.object(inferencer, 'batch_inference', return_value=['text']) as mock_batch:
            result = inferencer.infer_ocr(['/img1.png'])
            self.assertEqual(result, ['text'])
            called_messages = mock_batch.call_args[0][0]
            content = called_messages[0][0]['content']
            self.assertEqual(content[0]['type'], 'image')
            self.assertIn('Recognize the text', content[1]['text'])


class TestEnsureOneIGPath(unittest.TestCase):
    """Tests for ensure_oneig_path utility function."""

    def test_ensure_oneig_path_empty_raises(self):
        from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
            ensure_oneig_path,
        )
        with self.assertRaises(ImportError):
            ensure_oneig_path('')

    def test_ensure_oneig_path_not_exist_raises(self):
        from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
            ensure_oneig_path,
        )
        with self.assertRaises(ImportError):
            ensure_oneig_path('/nonexistent/path/12345')

    def test_ensure_oneig_path_valid_adds_to_sys_path(self):
        """有效路径应被加入sys.path"""
        from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
            ensure_oneig_path,
        )
        tmp = tempfile.mkdtemp()
        try:
            ensure_oneig_path(tmp)
            self.assertIn(tmp, sys.path)
        finally:
            if tmp in sys.path:
                sys.path.remove(tmp)
            shutil.rmtree(tmp, ignore_errors=True)


class TestSplitImageGrid(unittest.TestCase):
    """Tests for split_image_grid utility function."""

    def test_split_image_grid_non_black_image(self):
        """非黑色图片应被切分并保存"""
        from PIL import Image
        from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
            split_image_grid,
        )
        tmp = tempfile.mkdtemp()
        try:
            # 创建4x4像素的非黑色图片
            img = Image.new('RGB', (4, 4), color=(255, 0, 0))
            img_path = os.path.join(tmp, 'grid.png')
            img.save(img_path)
            cache = tempfile.mkdtemp()
            result = split_image_grid(img_path, (2, 2), cache)
            self.assertEqual(len(result), 4)
            for path in result:
                self.assertTrue(os.path.exists(path))
            shutil.rmtree(cache, ignore_errors=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_split_image_grid_black_image_filtered(self):
        """黑色子图应被过滤掉"""
        from PIL import Image
        from ais_bench.benchmark.tasks.oneig.oneig_eval_utils import (
            split_image_grid,
        )
        tmp = tempfile.mkdtemp()
        try:
            # 创建全黑图片
            img = Image.new('RGB', (4, 4), color=(0, 0, 0))
            img_path = os.path.join(tmp, 'black.png')
            img.save(img_path)
            cache = tempfile.mkdtemp()
            result = split_image_grid(img_path, (2, 2), cache)
            self.assertEqual(len(result), 0)
            shutil.rmtree(cache, ignore_errors=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
