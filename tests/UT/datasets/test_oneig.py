"""Unit tests for OneIGDataset - focus on public API."""
import atexit
import importlib
import importlib.util
import sys
import types
from unittest.mock import MagicMock


def _mock_mod(name, **attrs):
    """Create a mock module with proper __spec__ and optional attributes."""
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    mod.__path__ = []
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _setup_file_mocks():
    """Set up mock modules needed by this test file's import chain."""
    mocks = {}

    # datasets
    mocks['datasets'] = _mock_mod('datasets',
        Dataset=MagicMock(), DatasetDict=MagicMock(),
        load_dataset=MagicMock(), load_from_disk=MagicMock(),
        concatenate_datasets=MagicMock())
    mocks['datasets'].Dataset.from_list = MagicMock(return_value=MagicMock())
    mocks['datasets.utils'] = _mock_mod('datasets.utils')
    mocks['datasets.utils.logging'] = _mock_mod('datasets.utils.logging',
        disable_progress_bar=MagicMock())
    mocks['datasets'].utils = mocks['datasets.utils']
    mocks['datasets'].utils.logging = mocks['datasets.utils.logging']

    # torch and submodules
    torch = _mock_mod('torch',
        Tensor=MagicMock, nn=MagicMock(), optim=MagicMock(),
        utils=MagicMock(), cuda=MagicMock(), device=MagicMock(),
        no_grad=MagicMock(), manual_seed=MagicMock(), load=MagicMock(),
        float=MagicMock(), float16=MagicMock(), float32=MagicMock(),
        bfloat16=MagicMock(), tensor=MagicMock(), cat=MagicMock(),
        zeros=MagicMock(), ones=MagicMock(), max=MagicMock(),
        amp=MagicMock(), _utils=MagicMock(), _C=MagicMock(),
        jit=MagicMock(), onnx=MagicMock(), overrides=MagicMock(),
        library=MagicMock(), compile=MagicMock(return_value=lambda f: f),
        backends=MagicMock())
    torch.cuda.is_available = MagicMock(return_value=False)
    torch.cuda.manual_seed_all = MagicMock()
    torch.utils.data = MagicMock()
    torch.utils.data.DataLoader = MagicMock
    torch.utils.data.Dataset = MagicMock
    torch.amp.autocast = MagicMock()
    torch.backends.mps = MagicMock()
    torch.backends.mps.is_available = MagicMock(return_value=False)
    torch.backends.cuda = MagicMock()
    torch.backends.cuda.is_available = MagicMock(return_value=False)
    torch.backends.cudnn = MagicMock()
    torch.backends.cudnn.enabled = False
    mocks['torch'] = torch
    for sub in ['nn', 'cuda', 'amp', 'distributed', 'utils', 'utils.data',
                'optim', 'autograd', 'profiler', '_utils', '_C', 'jit',
                'onnx', 'overrides', 'library']:
        mocks.setdefault('torch.' + sub, _mock_mod('torch.' + sub))

    # mmengine
    mocks['mmengine.dist'] = _mock_mod('mmengine.dist',
        is_initialized=MagicMock(return_value=False),
        is_main_process=MagicMock(return_value=True),
        ProcessGroup=MagicMock)
    mocks['mmengine.device'] = _mock_mod('mmengine.device',
        is_npu_available=MagicMock(return_value=False),
        get_device=MagicMock(return_value='cpu'))

    # transformers
    mocks['transformers'] = _mock_mod('transformers',
        AutoTokenizer=MagicMock(), AutoModel=MagicMock(),
        AutoModelForCausalLM=MagicMock(), AutoConfig=MagicMock(),
        StoppingCriteria=type('StoppingCriteria', (), {}))
    mocks['transformers.generation'] = _mock_mod('transformers.generation')
    mocks['transformers.generation.stopping_criteria'] = _mock_mod(
        'transformers.generation.stopping_criteria',
        StoppingCriteria=type('StoppingCriteria', (), {}))

    # other dependencies
    mocks['evaluate'] = _mock_mod('evaluate',
        load=MagicMock(return_value=MagicMock()),
        combine=MagicMock(return_value=MagicMock()))
    mocks['orjson'] = _mock_mod('orjson',
        loads=MagicMock(return_value={}), dumps=MagicMock(return_value=b'{}'))
    mocks['fcntl'] = _mock_mod('fcntl',
        flock=MagicMock(), LOCK_EX=1, LOCK_UN=2, LOCK_SH=4, LOCK_NB=8)
    mocks['resource'] = _mock_mod('resource',
        getrlimit=MagicMock(return_value=(1000, 1000)),
        setrlimit=MagicMock(), RLIMIT_AS=1, RLIMIT_CPU=2, RLIMIT_NOFILE=3)
    for name in ['termios', 'pwd', 'grp']:
        mocks[name] = _mock_mod(name)

    return mocks


_FILE_MOCKS = _setup_file_mocks()
_file_originals = {k: sys.modules.get(k) for k in _FILE_MOCKS}
for _mod_name, _mock_obj in _FILE_MOCKS.items():
    if _mod_name in sys.modules:
        continue
    try:
        importlib.import_module(_mod_name)
    except Exception:
        sys.modules[_mod_name] = _mock_obj


def _restore_file_mocks():
    for mod_name, original in _file_originals.items():
        if original is not None:
            sys.modules[mod_name] = original
        else:
            sys.modules.pop(mod_name, None)


atexit.register(_restore_file_mocks)

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from ais_bench.benchmark.datasets.oneig import OneIGDataset


def _make_dataset(**attrs):
    """Create a OneIGDataset instance bypassing BaseDataset.__init__."""
    ds = OneIGDataset.__new__(OneIGDataset)
    ds.logger = MagicMock()
    for k, v in attrs.items():
        setattr(ds, k, v)
    return ds


class TestOneIGDataset(unittest.TestCase):
    """OneIGDataset 对外API测试"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_oneig_load_model_names_length_mismatch_raises_error(self):
        """model_names长度不匹配时应抛出ValueError"""
        ds = _make_dataset()
        with patch.object(ds, '_load_aux_data', return_value={}), \
             patch.object(ds, '_load_image_data', return_value=[]), \
             patch.object(ds, '_preprocess_data', return_value=[]), \
             patch('ais_bench.benchmark.datasets.oneig.Dataset.from_list'):
            with self.assertRaises(ValueError) as ctx:
                ds.load(path='/tmp/imgs', task_type='alignment',
                        model_names=['m1', 'm2'], image_grids=['2,2'])
            self.assertIn('model_names length', str(ctx.exception))

    def test_oneig_load_default_params(self):
        """默认参数应正确设置"""
        ds = _make_dataset()
        with patch.object(ds, '_load_aux_data', return_value={}), \
             patch.object(ds, '_load_image_data', return_value=[]), \
             patch.object(ds, '_preprocess_data', return_value=[]), \
             patch('ais_bench.benchmark.datasets.oneig.Dataset.from_list') as mock_from_list:
            mock_from_list.return_value = MagicMock()
            ds.load(path='/tmp/imgs')
            self.assertEqual(ds.model_names, ['default'])
            self.assertEqual(ds.image_grids, ['2,2'])
            self.assertEqual(ds.task_type, 'alignment')
            self.assertEqual(ds.mode, 'EN')

    def test_oneig_load_returns_dataset(self):
        """load应返回Dataset对象"""
        ds = _make_dataset()
        fake_dataset = MagicMock()
        with patch.object(ds, '_load_aux_data', return_value={}), \
             patch.object(ds, '_load_image_data', return_value=[]), \
             patch.object(ds, '_preprocess_data', return_value=[]), \
             patch('ais_bench.benchmark.datasets.oneig.Dataset.from_list',
                   return_value=fake_dataset):
            result = ds.load(path='/tmp/imgs')
            self.assertIs(result, fake_dataset)

    def test_oneig_load_aux_data_alignment(self):
        """alignment任务应正确加载question_dependency数据"""
        qd_dir = os.path.join(self.tmp, 'qd')
        os.makedirs(qd_dir)
        anime_data = {'s1': {'question': 'Q1', 'dependency': 'D1'}}
        with open(os.path.join(qd_dir, 'anime.json'), 'w', encoding='utf-8') as f:
            json.dump(anime_data, f)
        ds = _make_dataset(
            task_type='alignment', mode='EN',
            aux_data_paths={'question_dependency_dir': qd_dir})
        result = ds._load_aux_data()
        self.assertIn('alignment_data', result)
        self.assertEqual(result['alignment_data']['anime'], anime_data)

    def test_oneig_load_aux_data_reasoning(self):
        """reasoning任务应正确加载gt_answers数据"""
        gt_path = os.path.join(self.tmp, 'gt.json')
        gt_data = {'s1': '42', 's2': 'hello'}
        with open(gt_path, 'w', encoding='utf-8') as f:
            json.dump(gt_data, f)
        ds = _make_dataset(
            task_type='reasoning', mode='EN',
            aux_data_paths={'gt_answer_path': gt_path})
        result = ds._load_aux_data()
        self.assertEqual(result['gt_answers'], gt_data)

    def test_oneig_load_aux_data_text_valid(self):
        """text任务应正确加载text_contents数据"""
        csv_path = os.path.join(self.tmp, 'text.csv')
        df = pd.DataFrame({'id': ['s1', 's2'], 'text_content': ['hello', 'world']})
        df.to_csv(csv_path, index=False)
        ds = _make_dataset(
            task_type='text', mode='EN',
            aux_data_paths={'text_content_csv_path': csv_path})
        result = ds._load_aux_data()
        self.assertEqual(result['text_contents']['s1'], 'hello')
        self.assertEqual(result['text_contents']['s2'], 'world')

    def test_oneig_load_aux_data_style_valid(self):
        """style任务应正确加载并归一化style_labels数据"""
        csv_path = os.path.join(self.tmp, 'style.csv')
        df = pd.DataFrame({'id': ['s1', 's2'], 'class': ['Anime Style', 'Portrait']})
        df.to_csv(csv_path, index=False)
        ds = _make_dataset(
            task_type='style', mode='EN',
            aux_data_paths={'style_csv_path': csv_path})
        result = ds._load_aux_data()
        self.assertEqual(result['style_labels']['s1'], 'anime_style')
        self.assertEqual(result['style_labels']['s2'], 'portrait')

    def test_oneig_load_image_data_finds_images(self):
        """应正确找到并加载图片数据"""
        img_dir = os.path.join(self.tmp, 'text', 'm1')
        os.makedirs(img_dir, exist_ok=True)
        with open(os.path.join(img_dir, 'abc.png'), 'wb') as f:
            f.write(b'')
        ds = _make_dataset(
            task_type='text', mode='EN',
            model_names=['m1'], image_grids=['2,2'])
        result = ds._load_image_data(self.tmp, {})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['model_name'], 'm1')
        self.assertEqual(result[0]['id'], 'abc')

    def test_oneig_load_image_data_dir_not_found(self):
        """目录不存在时应返回空列表"""
        ds = _make_dataset(
            task_type='text', mode='EN',
            model_names=['m1'], image_grids=['2,2'])
        result = ds._load_image_data('/nonexistent/path', {})
        self.assertEqual(result, [])

    def test_oneig_preprocess_strips_whitespace(self):
        """_preprocess_data应去除字符串字段的首尾空白"""
        ds = _make_dataset()
        data = [
            {'gt_answer': '  42  '},
            {'expected_text': '  hello  '},
            {'style_label': '  anime  '},
        ]
        result = ds._preprocess_data(data)
        self.assertEqual(result[0]['gt_answer'], '42')
        self.assertEqual(result[1]['expected_text'], 'hello')
        self.assertEqual(result[2]['style_label'], 'anime')

    def test_oneig_preprocess_preserves_non_string(self):
        """_preprocess_data应保留非字符串字段"""
        ds = _make_dataset()
        data = [{'gt_answer': 42, 'other': None}]
        result = ds._preprocess_data(data)
        self.assertEqual(result[0]['gt_answer'], 42)
        self.assertIsNone(result[0]['other'])

    def test_oneig_get_task_class_iter_zh_alignment(self):
        """ZH模式alignment应包含multilingualism"""
        ds = _make_dataset(task_type='alignment', mode='ZH')
        result = ds._get_task_class_iter()
        class_names = [c[0] for c in result]
        self.assertIn('Multilingualism', class_names)

    def test_oneig_get_task_class_iter_en_alignment(self):
        """EN模式alignment不应包含multilingualism"""
        ds = _make_dataset(task_type='alignment', mode='EN')
        result = ds._get_task_class_iter()
        class_names = [c[0] for c in result]
        self.assertNotIn('Multilingualism', class_names)

    def test_oneig_load_alignment_task_fields(self):
        """alignment任务应正确设置question和dependency字段"""
        qd_dir = os.path.join(self.tmp, 'qd')
        os.makedirs(qd_dir)
        anime_data = {'abc': {'question': '{"1":"Q1"}', 'dependency': '{"1":[0]}'}}
        with open(os.path.join(qd_dir, 'anime.json'), 'w', encoding='utf-8') as f:
            json.dump(anime_data, f)
        img_dir = os.path.join(self.tmp, 'anime', 'm1')
        os.makedirs(img_dir)
        with open(os.path.join(img_dir, 'abc.png'), 'wb') as f:
            f.write(b'')
        ds = _make_dataset(task_type='alignment', mode='EN')
        with patch('ais_bench.benchmark.datasets.oneig.Dataset.from_list',
                   side_effect=lambda x: x):
            result = ds.load(path=self.tmp, task_type='alignment',
                            model_names=['m1'], image_grids=['2,2'],
                            aux_data_paths={'question_dependency_dir': qd_dir})
        item = result[0]
        self.assertEqual(item['question'], '{"1":"Q1"}')
        self.assertEqual(item['dependency'], '{"1":[0]}')

    def test_oneig_load_reasoning_task_fields(self):
        """reasoning任务应正确设置gt_answer字段"""
        gt_path = os.path.join(self.tmp, 'gt.json')
        with open(gt_path, 'w', encoding='utf-8') as f:
            json.dump({'abc': '42'}, f)
        img_dir = os.path.join(self.tmp, 'reasoning', 'm1')
        os.makedirs(img_dir)
        with open(os.path.join(img_dir, 'abc.png'), 'wb') as f:
            f.write(b'')
        ds = _make_dataset(task_type='reasoning', mode='EN')
        with patch('ais_bench.benchmark.datasets.oneig.Dataset.from_list',
                   side_effect=lambda x: x):
            result = ds.load(path=self.tmp, task_type='reasoning',
                            model_names=['m1'], image_grids=['2,2'],
                            aux_data_paths={'gt_answer_path': gt_path})
        self.assertEqual(result[0]['gt_answer'], '42')

    def test_oneig_load_reasoning_empty_gt_skipped(self):
        """reasoning任务空gt_answer应被跳过"""
        gt_path = os.path.join(self.tmp, 'gt.json')
        with open(gt_path, 'w', encoding='utf-8') as f:
            json.dump({'abc': ''}, f)
        img_dir = os.path.join(self.tmp, 'reasoning', 'm1')
        os.makedirs(img_dir)
        with open(os.path.join(img_dir, 'abc.png'), 'wb') as f:
            f.write(b'')
        ds = _make_dataset(task_type='reasoning', mode='EN')
        with patch('ais_bench.benchmark.datasets.oneig.Dataset.from_list',
                   side_effect=lambda x: x):
            result = ds.load(path=self.tmp, task_type='reasoning',
                            model_names=['m1'], image_grids=['2,2'],
                            aux_data_paths={'gt_answer_path': gt_path})
        self.assertEqual(len(result), 0)

    def test_oneig_is_valid_style_label(self):
        """测试_is_valid_style_label各种输入"""
        ds = _make_dataset(task_type='style')
        import pandas as pd
        self.assertFalse(ds._is_valid_style_label(pd.NA))
        self.assertFalse(ds._is_valid_style_label(''))
        self.assertFalse(ds._is_valid_style_label('nan'))
        self.assertFalse(ds._is_valid_style_label('  '))
        self.assertTrue(ds._is_valid_style_label('impressionism'))
        self.assertTrue(ds._is_valid_style_label('Pop Art'))


if __name__ == '__main__':
    unittest.main()
