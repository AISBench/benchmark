"""Conftest for OneIG task tests.

Pre-mocks missing heavy dependencies before test collection.
Mocks are scoped to this directory only and restored after session.
"""
import importlib
import importlib.util
import sys
import types
from unittest.mock import MagicMock

import pytest


def _mock_mod(name, **attrs):
    """Create a mock module with __spec__ and optional attributes."""
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    mod.__path__ = []
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _setup_mocks():
    """Set up mock modules for missing heavy dependencies."""
    m = {}

    # Helper: MagicMock with return value
    def _mr(val):
        return MagicMock(return_value=val)

    # ---- torch ----
    torch = _mock_mod('torch',
        Tensor=MagicMock, nn=MagicMock(), optim=MagicMock(),
        utils=MagicMock(), cuda=MagicMock(), device=_mr(MagicMock()),
        no_grad=MagicMock(), manual_seed=MagicMock(), load=MagicMock(),
        float=MagicMock(), float16=MagicMock(name='float16'),
        float32=MagicMock(name='float32'), bfloat16=MagicMock(name='bfloat16'),
        tensor=_mr(MagicMock()), cat=MagicMock(), zeros=_mr(MagicMock()),
        ones=_mr(MagicMock()), max=_mr(MagicMock()), amp=MagicMock(),
        _utils=MagicMock(), _C=MagicMock(), jit=MagicMock(),
        onnx=MagicMock(), overrides=MagicMock(), library=MagicMock(),
        _meta_registrations=MagicMock(), _inductor=MagicMock(),
        compile=_mr(lambda f: f), backends=MagicMock(),
        LongTensor=MagicMock, FloatTensor=MagicMock, IntTensor=MagicMock,
        DoubleTensor=MagicMock, HalfTensor=MagicMock, ByteTensor=MagicMock,
        CharTensor=MagicMock, ShortTensor=MagicMock, BoolTensor=MagicMock,
    )
    torch.cuda.is_available = _mr(False)
    torch.cuda.manual_seed_all = MagicMock()
    torch.cuda.empty_cache = MagicMock()
    torch.amp.autocast = MagicMock()
    torch.backends.mps = MagicMock()
    torch.backends.mps.is_available = _mr(False)
    torch.backends.cuda = MagicMock()
    torch.backends.cuda.is_available = _mr(False)
    torch.backends.cudnn = MagicMock()
    torch.backends.cudnn.enabled = False
    torch.utils.data = MagicMock()
    torch.utils.data.DataLoader = MagicMock
    torch.utils.data.Dataset = MagicMock
    torch.nn.Module = MagicMock
    torch.nn.functional = MagicMock()
    torch._utils._flatten_dense_tensors = MagicMock()
    torch._utils._take_tensors = MagicMock()
    torch._utils._unflatten_dense_tensors = MagicMock()
    m['torch'] = torch

    # torch submodules
    for sub in ['nn', 'cuda', 'amp', 'distributed', 'utils', 'utils.data',
                'optim', 'autograd', 'profiler', '_utils', '_C', 'jit',
                'onnx', 'overrides', 'library']:
        if f'torch.{sub}' not in m:
            m[f'torch.{sub}'] = _mock_mod(f'torch.{sub}')

    # torch.utils.data needs DataLoader/Dataset for imports
    m['torch.utils.data'].DataLoader = MagicMock
    m['torch.utils.data'].Dataset = MagicMock
    m['torch.utils.data'].default_collate = MagicMock()
    torch.utils.data = m['torch.utils.data']
    # torch.utils needs checkpoint
    m['torch.utils'].checkpoint = MagicMock()
    m['torch.utils'].data = m['torch.utils.data']
    # torch.nn needs Module
    m['torch.nn'].Module = MagicMock
    m['torch.nn'].functional = MagicMock()
    # torch.optim needs optimizer classes
    m['torch.optim'].AdamW = MagicMock
    m['torch.optim'].Adam = MagicMock
    m['torch.optim'].SGD = MagicMock
    # torch.autograd needs grad/backward
    m['torch.autograd'].grad = MagicMock()
    m['torch.autograd'].backward = MagicMock()

    # torch.distributed specific
    td = m['torch.distributed']
    td.is_initialized = _mr(False)
    td.is_available = _mr(False)
    td.get_rank = _mr(0)
    td.get_world_size = _mr(1)

    # ---- mmengine ----
    m['mmengine.dist'] = _mock_mod('mmengine.dist',
        is_initialized=_mr(False), is_main_process=_mr(True),
        get_rank=_mr(0), get_world_size=_mr(1), ProcessGroup=MagicMock)
    m['mmengine.device'] = _mock_mod('mmengine.device',
        is_npu_available=_mr(False), get_device=_mr('cpu'),
        get_max_cuda_memory=_mr(0), get_max_musa_memory=_mr(0))

    # ---- transformers ----
    tf = _mock_mod('transformers',
        Qwen3VLForConditionalGeneration=MagicMock(),
        AutoProcessor=MagicMock(), AutoTokenizer=MagicMock(),
        StoppingCriteria=type('StoppingCriteria', (), {}),
        StoppingCriteriaList=_mr([]), PreTrainedTokenizer=MagicMock,
        PreTrainedTokenizerFast=MagicMock, AutoModel=MagicMock(),
        AutoModelForCausalLM=MagicMock(), AutoConfig=MagicMock(),
        GenerationConfig=MagicMock(), BitsAndBytesConfig=MagicMock(),
        pipeline=MagicMock(), Qwen2_5_VLForConditionalGeneration=MagicMock(),
        Qwen2_5OmniProcessor=MagicMock(), Qwen2VLProcessor=MagicMock(),
        Qwen2Tokenizer=MagicMock(), BertTokenizer=MagicMock(),
        BertLMHeadModel=MagicMock(), BertConfig=MagicMock(),
    )
    tf_gen = _mock_mod('transformers.generation', GenerationConfig=MagicMock())
    tf_sc = _mock_mod('transformers.generation.stopping_criteria',
        StoppingCriteria=type('StoppingCriteria', (), {}))
    tf_gen.stopping_criteria = tf_sc
    tf.generation = tf_gen
    m['transformers'] = tf
    m['transformers.generation'] = tf_gen
    m['transformers.generation.stopping_criteria'] = tf_sc

    # ---- torchvision ----
    tv_t = _mock_mod('torchvision.transforms',
        Compose=MagicMock(), Resize=MagicMock(), CenterCrop=MagicMock(),
        ToTensor=MagicMock(), Normalize=MagicMock(), InterpolationMode=MagicMock())
    m['torchvision'] = _mock_mod('torchvision', transforms=tv_t)
    m['torchvision.transforms'] = tv_t

    # ---- other ML/NLP deps ----
    m['qwen_vl_utils'] = _mock_mod('qwen_vl_utils',
        process_vision_info=_mr((None, None)))
    m['dreamsim'] = _mock_mod('dreamsim',
        dreamsim=_mr((MagicMock(), MagicMock())))
    m['evaluate'] = _mock_mod('evaluate',
        load=_mr(MagicMock()), combine=_mr(MagicMock()))
    m['PIL'] = _mock_mod('PIL', Image=_mock_mod('PIL.Image', open=MagicMock()))
    m['PIL.Image'] = m['PIL'].Image

    # numpy
    np_attrs = {k: _mr(MagicMock()) for k in
        ['zeros', 'ones', 'array', 'asarray', 'concatenate',
         'expand_dims', 'stack', 'squeeze', 'reshape']}
    np_attrs.update({
        'ndarray': MagicMock,
        'float32': MagicMock(name='float32'),
        'float64': MagicMock(name='float64'),
        'int32': MagicMock(name='int32'),
        'int64': MagicMock(name='int64'),
        'uint8': MagicMock(name='uint8'),
        'mean': _mr(0.0), 'sum': _mr(0),
        'max': _mr(0), 'min': _mr(0),
    })
    m['numpy'] = _mock_mod('numpy', **np_attrs)

    # scipy
    m['scipy'] = _mock_mod('scipy',
        stats=_mock_mod('scipy.stats', hypergeom=MagicMock()),
        integrate=_mock_mod('scipy.integrate'))
    m['scipy.stats'] = m['scipy'].stats
    m['scipy.integrate'] = m['scipy'].integrate

    # nltk
    nltk_t = _mock_mod('nltk.translate',
        bleu_score=_mock_mod('nltk.translate.bleu_score', sentence_bleu=_mr(0.5)))
    nltk_t.tokenize = _mock_mod('nltk.tokenize', word_tokenize=_mr([]))
    m['nltk'] = _mock_mod('nltk', translate=nltk_t)
    m['nltk.translate'] = nltk_t
    m['nltk.translate.bleu_score'] = nltk_t.bleu_score
    m['nltk.tokenize'] = nltk_t.tokenize

    # simple single-module deps
    m['jieba'] = _mock_mod('jieba', cut=_mr([]), lcut=_mr([]))
    m['rouge_chinese'] = _mock_mod('rouge_chinese', Rouge=MagicMock())
    m['rouge_score'] = _mock_mod('rouge_score',
        rouge_scorer=_mock_mod('rouge_score.rouge_scorer', RougeScorer=MagicMock))
    m['rouge_score.rouge_scorer'] = m['rouge_score'].rouge_scorer
    m['sacrebleu'] = _mock_mod('sacrebleu',
        corpus_bleu=_mr(MagicMock(score=0)), BLEU=MagicMock)
    m['bert_score'] = _mock_mod('bert_score', score=_mr(([], [], [])))
    m['jiwer'] = _mock_mod('jiwer', compute_measures=_mr({}))
    m['Levenshtein'] = _mock_mod('Levenshtein', distance=_mr(0))

    # rapidfuzz
    rf_d = _mock_mod('rapidfuzz.distance',
        Levenshtein=_mock_mod('rapidfuzz.distance.Levenshtein', distance=_mr(0)))
    m['rapidfuzz'] = _mock_mod('rapidfuzz', distance=rf_d)
    m['rapidfuzz.distance'] = rf_d
    m['rapidfuzz.distance.Levenshtein'] = rf_d.Levenshtein

    m['latex2sympy2'] = _mock_mod('latex2sympy2', latex2sympy=_mr(MagicMock()))
    m['sympy'] = _mock_mod('sympy',
        sympify=_mr(MagicMock()), Eq=MagicMock, solve=_mr([]))
    m['lark'] = _mock_mod('lark',
        Lark=MagicMock, Transformer=type('Transformer', (), {}))
    m['sentence_transformers'] = _mock_mod('sentence_transformers',
        SentenceTransformer=MagicMock)
    m['safetensors'] = _mock_mod('safetensors',
        torch=_mock_mod('safetensors.torch', load_file=_mr({})))
    m['safetensors.torch'] = m['safetensors'].torch

    # pycocoevalcap
    pcec = _mock_mod('pycocoevalcap',
        bleu=_mock_mod('pycocoevalcap.bleu', bleu=MagicMock()),
        cider=_mock_mod('pycocoevalcap.cider', cider=MagicMock()),
        rouge=_mock_mod('pycocoevalcap.rouge', rouge=MagicMock()),
        meteor=_mock_mod('pycocoevalcap.meteor', meteor=MagicMock()),
        tokenizer=_mock_mod('pycocoevalcap.tokenizer'))
    m['pycocoevalcap'] = pcec
    for sub in ['bleu', 'cider', 'rouge', 'meteor', 'tokenizer']:
        m[f'pycocoevalcap.{sub}'] = getattr(pcec, sub)

    # pycocotools
    pct = _mock_mod('pycocotools',
        coco=_mock_mod('pycocotools.coco', COCO=MagicMock),
        cocoeval=_mock_mod('pycocotools.cocoeval'))
    m['pycocotools'] = pct
    m['pycocotools.coco'] = pct.coco
    m['pycocotools.cocoeval'] = pct.cocoeval

    # rich
    rich_subs = {s: _mock_mod(f'rich.{s}') for s in
                 ['live', 'console', 'table', 'panel', 'text', 'progress']}
    for s, mod in rich_subs.items():
        cap = s.capitalize() if s != 'live' else 'Live'
        setattr(mod, cap, MagicMock)
    rich = _mock_mod('rich', **rich_subs)
    m['rich'] = rich
    for s, mod in rich_subs.items():
        m[f'rich.{s}'] = mod

    # openai
    m['openai'] = _mock_mod('openai',
        OpenAI=MagicMock, AsyncOpenAI=MagicMock,
        APIError=Exception, APIConnectionError=Exception, RateLimitError=Exception)

    # tiktoken
    m['tiktoken'] = _mock_mod('tiktoken',
        Encoding=MagicMock, get_encoding=_mr(MagicMock()),
        encoding_for_model=_mr(MagicMock()))

    # misc simple modules
    for name in ['accelerate', 'bitsandbytes', 'einops', 'cpm_kernels',
                 'cv2', 'chardet', 'charset_normalizer', 'antlr4']:
        m[name] = _mock_mod(name)

    m['peft'] = _mock_mod('peft', LoraConfig=MagicMock(), get_peft_model=MagicMock())
    m['trl'] = _mock_mod('trl', SFTConfig=MagicMock(), SFTTrainer=MagicMock())
    m['func_timeout'] = _mock_mod('func_timeout',
        func_timeout=MagicMock(), FunctionTimedOut=Exception)

    # fuzzywuzzy
    fw_f = _mock_mod('fuzzywuzzy.fuzz', ratio=_mr(80))
    m['fuzzywuzzy'] = _mock_mod('fuzzywuzzy',
        fuzz=fw_f, process=_mock_mod('fuzzywuzzy.process'))
    m['fuzzywuzzy.fuzz'] = fw_f
    m['fuzzywuzzy.process'] = m['fuzzywuzzy'].process

    m['jsonlines'] = _mock_mod('jsonlines', open=MagicMock())
    m['immutabledict'] = _mock_mod('immutabledict', Immutabledict=dict)
    m['loguru'] = _mock_mod('loguru', logger=MagicMock())
    m['gradio_client'] = _mock_mod('gradio_client', Client=MagicMock())

    # absl
    m['absl'] = _mock_mod('absl',
        flags=_mock_mod('absl.flags'), logging=_mock_mod('absl.logging'))
    m['absl.flags'] = m['absl'].flags
    m['absl.logging'] = m['absl'].logging

    # yaml, toml
    m['yaml'] = _mock_mod('yaml', safe_load=_mr({}), safe_dump=MagicMock())
    m['toml'] = _mock_mod('toml', load=_mr({}), dumps=MagicMock())

    # datasets
    ds = _mock_mod('datasets',
        Dataset=MagicMock(), DatasetDict=MagicMock(),
        load_dataset=_mr(MagicMock()), load_from_disk=_mr(MagicMock()),
        concatenate_datasets=_mr(MagicMock()))
    ds.Dataset.from_list = _mr(MagicMock())
    m['datasets'] = ds

    # Unix-only modules
    m['fcntl'] = _mock_mod('fcntl', flock=MagicMock(),
        LOCK_EX=1, LOCK_UN=2, LOCK_SH=4, LOCK_NB=8)
    m['resource'] = _mock_mod('resource',
        getrlimit=_mr((1000, 1000)), setrlimit=MagicMock(),
        RLIMIT_AS=1, RLIMIT_CPU=2, RLIMIT_NOFILE=3)
    for name in ['termios', 'pwd', 'grp']:
        m[name] = _mock_mod(name)

    # janus
    m['janus'] = _mock_mod('janus', Queue=type('Queue', (), {
        'async_q': MagicMock, 'sync_q': MagicMock}))

    return m


MOCK_MODULES = _setup_mocks()

_original_modules = {k: sys.modules.get(k) for k in MOCK_MODULES}

for mod_name, mock_obj in MOCK_MODULES.items():
    if mod_name in sys.modules:
        continue
    try:
        importlib.import_module(mod_name)
    except Exception:
        sys.modules[mod_name] = mock_obj


@pytest.fixture(scope="session", autouse=True)
def _restore_modules_after_session():
    """Restore original sys.modules state after all OneIG tests complete."""
    yield
    for mod_name, original in _original_modules.items():
        if original is not None:
            sys.modules[mod_name] = original
        else:
            sys.modules.pop(mod_name, None)
