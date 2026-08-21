"""Unit tests for OneIGSummarizer - focus on public API."""
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

import unittest
from unittest.mock import patch

from mmengine.config import ConfigDict

from ais_bench.benchmark.summarizers.default import DefaultSummarizer
from ais_bench.benchmark.summarizers.oneig import OneIGSummarizer


class TestOneIGSummarizer(unittest.TestCase):
    """OneIGSummarizer 对外API测试"""

    def test_oneig_register_metric_new_metric(self):
        """注册新metric时应正确更新所有字典"""
        raw_results = {}
        parsed_results = {}
        dataset_metrics = {}
        dataset_eval_mode = {}
        OneIGSummarizer._register_metric(
            'oneig_diversity_anime', 85.0, 'model_a',
            raw_results, parsed_results, dataset_metrics, dataset_eval_mode)
        self.assertEqual(
            raw_results['model_a']['oneig_diversity_anime']['accuracy'], 85.0)
        self.assertEqual(
            parsed_results['model_a']['oneig_diversity_anime']['accuracy'], 85.0)
        self.assertEqual(dataset_metrics['oneig_diversity_anime'], ['accuracy'])
        self.assertEqual(dataset_eval_mode['oneig_diversity_anime'], 'gen')

    def test_oneig_register_metric_existing_metric(self):
        """已存在metric时不应覆盖dataset_metrics"""
        raw_results = {}
        parsed_results = {}
        dataset_metrics = {'oneig_diversity_anime': ['custom_metric']}
        dataset_eval_mode = {}
        OneIGSummarizer._register_metric(
            'oneig_diversity_anime', 85.0, 'model_a',
            raw_results, parsed_results, dataset_metrics, dataset_eval_mode)
        self.assertEqual(
            dataset_metrics['oneig_diversity_anime'], ['custom_metric'])
        self.assertEqual(
            raw_results['model_a']['oneig_diversity_anime']['accuracy'], 85.0)

    def test_oneig_aggregate_diversity_with_class_scores(self):
        """class_scores应正确聚合细粒度指标"""
        summ = OneIGSummarizer.__new__(OneIGSummarizer)
        raw_results = {'model_a': {
            'oneig_diversity': {
                'class_scores': {
                    'anime': [0.8, 0.9],
                    'human': [0.7, 0.6],
                },
            }
        }}
        parsed_results = {}
        dataset_metrics = {}
        dataset_eval_mode = {}
        summ._aggregate_diversity_finegrain(
            'model_a', raw_results, parsed_results,
            dataset_metrics, dataset_eval_mode, raw_results['model_a'])
        # anime: avg = 0.85, *100 = 85.0
        self.assertAlmostEqual(
            raw_results['model_a']['oneig_diversity_anime']['accuracy'], 85.0)
        # human: avg = 0.65, *100 = 65.0
        self.assertAlmostEqual(
            raw_results['model_a']['oneig_diversity_human']['accuracy'], 65.0)
        self.assertIn('oneig_diversity_anime', dataset_metrics)
        self.assertIn('oneig_diversity_human', dataset_metrics)

    def test_oneig_aggregate_diversity_class_scores_with_none(self):
        """class_scores含None时应过滤后计算"""
        summ = OneIGSummarizer.__new__(OneIGSummarizer)
        raw_results = {'model_a': {
            'oneig_diversity': {
                'class_scores': {'anime': [0.8, None, 0.9]},
            }
        }}
        parsed_results = {}
        dataset_metrics = {}
        dataset_eval_mode = {}
        summ._aggregate_diversity_finegrain(
            'model_a', raw_results, parsed_results,
            dataset_metrics, dataset_eval_mode, raw_results['model_a'])
        # valid = [0.8, 0.9], avg = 0.85, *100 = 85.0
        self.assertAlmostEqual(
            raw_results['model_a']['oneig_diversity_anime']['accuracy'], 85.0)

    def test_oneig_aggregate_diversity_fallback_details(self):
        """无class_scores时应回退到details聚合"""
        summ = OneIGSummarizer.__new__(OneIGSummarizer)
        raw_results = {'model_a': {
            'oneig_diversity': {
                'details': [
                    {'class_item': 'anime', 'score': 0.8},
                    {'class_item': 'anime', 'score': 0.9},
                    {'class_item': 'human', 'score': 0.7},
                ],
            }
        }}
        parsed_results = {}
        dataset_metrics = {}
        dataset_eval_mode = {}
        summ._aggregate_diversity_finegrain(
            'model_a', raw_results, parsed_results,
            dataset_metrics, dataset_eval_mode, raw_results['model_a'])
        # anime: avg = 0.85, *100 = 85.0
        self.assertAlmostEqual(
            raw_results['model_a']['oneig_diversity_anime']['accuracy'], 85.0)
        # human: avg = 0.7, *100 = 70.0
        self.assertAlmostEqual(
            raw_results['model_a']['oneig_diversity_human']['accuracy'], 70.0)

    @patch('ais_bench.benchmark.summarizers.default.AISLogger')
    @patch.object(DefaultSummarizer, '_calculate_group_metrics')
    def test_oneig_summarizer_full_flow(self, mock_super, _log):
        """端到端测试：full flow应正确聚合细粒度指标"""
        cfg = ConfigDict({
            'models': [{'abbr': 'model_a'}],
            'datasets': [{'abbr': 'oneig_diversity'}],
            'work_dir': '/tmp',
        })
        summ = OneIGSummarizer(cfg)

        raw_results = {
            'model_a': {
                'oneig_diversity': {
                    'accuracy': 75.0,
                    'class_scores': {
                        'anime': [0.8, 0.9, None],
                        'human': [0.7, 0.6],
                        'object': [None, None],
                    },
                    'details': [],
                }
            }
        }
        parsed_results = {
            'model_a': {'oneig_diversity': {'accuracy': 75.0}},
        }
        dataset_metrics = {'oneig_diversity': ['accuracy']}
        dataset_eval_mode = {'oneig_diversity': 'gen'}

        mock_super.side_effect = lambda rr, pr, dm, dem: (rr, pr, dm, dem)

        raw_results, parsed_results, dataset_metrics, dataset_eval_mode = \
            summ._calculate_group_metrics(
                raw_results, parsed_results, dataset_metrics,
                dataset_eval_mode)

        # anime: valid = [0.8, 0.9], avg = 0.85, *100 = 85.0
        self.assertIn('oneig_diversity_anime', parsed_results['model_a'])
        self.assertAlmostEqual(
            parsed_results['model_a']['oneig_diversity_anime']['accuracy'], 85.0)
        # human: valid = [0.7, 0.6], avg = 0.65, *100 = 65.0
        self.assertAlmostEqual(
            parsed_results['model_a']['oneig_diversity_human']['accuracy'], 65.0)
        # object: valid = [], skipped
        self.assertNotIn('oneig_diversity_object', parsed_results['model_a'])
        self.assertIn('oneig_diversity_anime', dataset_metrics)
        self.assertIn('oneig_diversity_human', dataset_metrics)
        self.assertEqual(dataset_eval_mode['oneig_diversity_anime'], 'gen')


if __name__ == '__main__':
    unittest.main()
