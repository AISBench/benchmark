import sys
import os
import json
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _setup_mocks():
    minisweagent = type('minisweagent', (), {})()
    minisweagent.run = type('run', (), {})()
    minisweagent.run.utils = type('utils', (), {})()
    minisweagent.run.utils.batch_instances = type('batch_instances', (), {})()
    minisweagent.run.utils.batch_instances.BatchInstance = MagicMock
    minisweagent.run.extra = type('extra', (), {})()
    minisweagent.run.extra.utils = type('utils', (), {})()
    minisweagent.run.extra.utils.batch_progress = type('batch_progress', (), {})()
    minisweagent.run.extra.utils.batch_progress.RunBatchProgressManager = MagicMock
    minisweagent.run.extra.run_batch = type('run_batch', (), {})()
    minisweagent.run.extra.run_batch.process_instance = MagicMock()
    minisweagent.run.extra.run_batch.RunBatchConfig = MagicMock()
    minisweagent.config = type('config', (), {})()
    minisweagent.config.get_config_path = MagicMock()

    orjson = MagicMock()
    orjson.loads = MagicMock(return_value={})
    orjson.dumps = MagicMock(return_value=b'{}')

    docker = MagicMock()
    docker.errors = type('errors', (), {})()
    docker.errors.Timeout = Exception
    docker.errors.ImageNotFound = Exception
    docker.from_env = MagicMock()

    transformers = MagicMock()
    transformers.AutoTokenizer = MagicMock()
    transformers.AutoTokenizer.from_pretrained = MagicMock()

    fcntl = MagicMock()
    fcntl.flock = MagicMock()
    fcntl.LOCK_EX = 1
    fcntl.LOCK_UN = 2

    tqdm = MagicMock()
    tqdm.auto = MagicMock()
    tqdm.auto.tqdm = MagicMock()

    datasets = MagicMock()
    datasets.Dataset = MagicMock()
    datasets.DatasetDict = MagicMock()
    datasets.load_dataset = MagicMock()
    datasets.load_from_disk = MagicMock()
    datasets.features = MagicMock()
    datasets.features.Features = MagicMock()
    datasets.features.Value = MagicMock()
    datasets.utils = MagicMock()
    datasets.utils.logging = MagicMock()
    datasets.utils.logging.disable_progress_bar = MagicMock()

    scipy = MagicMock()
    scipy.stats = MagicMock()
    scipy.stats.hypergeom = MagicMock()
    scipy.integrate = MagicMock()

    jieba = MagicMock()
    jieba.cut = MagicMock(return_value=[])
    jieba.lcut = MagicMock(return_value=[])

    rouge_chinese = MagicMock()
    rouge_chinese.Rouge = MagicMock()

    mmengine = MagicMock()
    mmengine.registry = MagicMock()
    mmengine.registry.METRICS = MagicMock()
    mmengine.registry.Registry = type('Registry', (), {
        '__init__': lambda *args, **kwargs: None,
        'register_module': lambda *args, **kwargs: (lambda f: f),
    })
    mmengine.config = MagicMock()
    mmengine.config.ConfigDict = dict
    mmengine.config.Config = MagicMock
    mmengine.fileio = MagicMock()
    mmengine.utils = MagicMock()
    mmengine.logging = MagicMock()
    mmengine.logging.MessageHub = MagicMock
    mmengine.hub = MagicMock()
    mmengine.dist = MagicMock()
    mmengine.device = MagicMock()
    mmengine.visualization = MagicMock()
    mmengine.optim = MagicMock()
    mmengine.runner = MagicMock()
    mmengine.structures = MagicMock()
    mmengine.structures.InstanceData = MagicMock
    mmengine.load = MagicMock()
    mmengine.load.side_effect = lambda path: json.load(open(path))
    mmengine_lite = MagicMock()
    mmengine_lite.logging = MagicMock()
    mmengine_lite.logging.MessageHub = MagicMock

    evaluate = MagicMock()
    evaluate.load = MagicMock(return_value=MagicMock())
    evaluate.combine = MagicMock(return_value=MagicMock())

    nltk = MagicMock()
    nltk.translate = MagicMock()
    nltk.translate.bleu_score = MagicMock()
    nltk.translate.bleu_score.sentence_bleu = MagicMock(return_value=0.5)
    nltk.tokenize = MagicMock()
    nltk.tokenize.word_tokenize = MagicMock(return_value=[])

    accelerate = MagicMock()

    transformers = MagicMock()
    transformers.AutoTokenizer = MagicMock()
    transformers.AutoTokenizer.from_pretrained = MagicMock()
    transformers.AutoModelForCausalLM = MagicMock()
    transformers.AutoModelForCausalLM.from_pretrained = MagicMock()
    transformers.pipeline = MagicMock()
    transformers.BitsAndBytesConfig = MagicMock()
    transformers.GenerationConfig = MagicMock()
    transformers.StopStringCriteria = MagicMock()
    transformers.StoppingCriteriaList = MagicMock()
    transformers.generation = MagicMock()
    transformers.generation.stopping_criteria = MagicMock()
    transformers.generation.stopping_criteria.StoppingCriteria = type('StoppingCriteria', (), {})

    peft = MagicMock()
    peft.LoraConfig = MagicMock()
    peft.get_peft_model = MagicMock()

    bitsandbytes = MagicMock()

    trl = MagicMock()
    trl.SFTConfig = MagicMock()
    trl.SFTTrainer = MagicMock()

    einops = MagicMock()

    func_timeout = MagicMock()
    func_timeout.func_timeout = MagicMock()
    func_timeout.FunctionTimedOut = Exception

    fuzzywuzzy = MagicMock()
    fuzzywuzzy.fuzz = MagicMock()
    fuzzywuzzy.fuzz.ratio = MagicMock(return_value=80)
    fuzzywuzzy.process = MagicMock()

    jsonlines = MagicMock()
    jsonlines.open = MagicMock()

    absl = MagicMock()
    absl.flags = MagicMock()
    absl.logging = MagicMock()

    cpm_kernels = MagicMock()

    gradio_client = MagicMock()
    gradio_client.Client = MagicMock()

    immutabledict = MagicMock()
    immutabledict.Immutabledict = dict

    janus = MagicMock()
    janus.Queue = type('Queue', (), {
        'async_q': MagicMock,
        'sync_q': MagicMock,
    })

    loguru = MagicMock()
    loguru.logger = MagicMock()

    rich = MagicMock()
    rich.live = MagicMock()
    rich.live.Live = MagicMock
    rich.console = MagicMock()
    rich.console.Console = MagicMock
    rich.table = MagicMock()
    rich.table.Table = MagicMock
    rich.panel = MagicMock()
    rich.panel.Panel = MagicMock
    rich.text = MagicMock()
    rich.text.Text = MagicMock
    rich.progress = MagicMock()
    rich.progress.Progress = MagicMock

    openai = MagicMock()
    openai.OpenAI = MagicMock
    openai.AsyncOpenAI = MagicMock
    openai.APIError = Exception
    openai.APIConnectionError = Exception
    openai.RateLimitError = Exception

    requests = MagicMock()
    requests.get = MagicMock()
    requests.post = MagicMock()
    requests.Session = MagicMock

    yaml = MagicMock()
    yaml.safe_load = MagicMock(return_value={})
    yaml.safe_dump = MagicMock()

    toml = MagicMock()
    toml.load = MagicMock(return_value={})
    toml.dumps = MagicMock()

    PIL = MagicMock()
    PIL.Image = MagicMock()
    PIL.Image.open = MagicMock()

    cv2 = MagicMock()

    torch = MagicMock()
    torch.Tensor = MagicMock
    torch.nn = MagicMock()
    torch.nn.Module = MagicMock
    torch.optim = MagicMock()
    torch.utils = MagicMock()
    torch.utils.data = MagicMock()
    torch.utils.data.DataLoader = MagicMock
    torch.utils.data.Dataset = MagicMock
    torch.cuda = MagicMock()
    torch.cuda.is_available = MagicMock(return_value=False)
    torch.device = MagicMock
    torch.no_grad = MagicMock()

    safetensors = MagicMock()
    safetensors.torch = MagicMock()
    safetensors.torch.load_file = MagicMock(return_value={})

    sentence_transformers = MagicMock()
    sentence_transformers.SentenceTransformer = MagicMock

    rouge_score = MagicMock()
    rouge_score.rouge_scorer = MagicMock()
    rouge_score.rouge_scorer.RougeScorer = MagicMock

    prompt_toolkit = MagicMock()
    prompt_toolkit.shortcuts = MagicMock()

    python_multipart = MagicMock()

    aiohttp = MagicMock()
    aiohttp.ClientSession = MagicMock

    httpx = MagicMock()
    httpx.Client = MagicMock
    httpx.AsyncClient = MagicMock

    tiktoken = MagicMock()
    tiktoken.Encoding = MagicMock
    tiktoken.get_encoding = MagicMock(return_value=MagicMock())
    tiktoken.encoding_for_model = MagicMock(return_value=MagicMock())

    lark = MagicMock()
    lark.Lark = MagicMock
    lark.Transformer = type('Transformer', (), {})

    sympy = MagicMock()
    sympy.sympify = MagicMock(return_value=MagicMock())
    sympy.Eq = MagicMock
    sympy.solve = MagicMock(return_value=[])

    Levenshtein = MagicMock()
    Levenshtein.distance = MagicMock(return_value=0)

    rapidfuzz = MagicMock()
    rapidfuzz.distance = MagicMock()
    rapidfuzz.distance.Levenshtein = MagicMock()
    rapidfuzz.distance.Levenshtein.distance = MagicMock(return_value=0)

    latex2sympy2 = MagicMock()
    latex2sympy2.latex2sympy = MagicMock(return_value=MagicMock())

    antlr4 = MagicMock()

    pycocoevalcap = MagicMock()
    pycocoevalcap.bleu = MagicMock()
    pycocoevalcap.bleu.bleu = MagicMock()
    pycocoevalcap.cider = MagicMock()
    pycocoevalcap.cider.cider = MagicMock()
    pycocoevalcap.rouge = MagicMock()
    pycocoevalcap.rouge.rouge = MagicMock()
    pycocoevalcap.meteor = MagicMock()
    pycocoevalcap.meteor.meteor = MagicMock()
    pycocoevalcap.tokenizer = MagicMock()

    pycocotools = MagicMock()
    pycocotools.coco = MagicMock()
    pycocotools.coco.COCO = MagicMock
    pycocotools.cocoeval = MagicMock()

    jiwer = MagicMock()
    jiwer.compute_measures = MagicMock(return_value={})

    sacrebleu = MagicMock()
    sacrebleu.corpus_bleu = MagicMock(return_value=MagicMock(score=0))
    sacrebleu.BLEU = MagicMock

    bert_score = MagicMock()
    bert_score.score = MagicMock(return_value=([], [], []))

    resource = MagicMock()
    resource.getrlimit = MagicMock(return_value=(1000, 1000))
    resource.setrlimit = MagicMock()
    resource.RLIMIT_AS = 1
    resource.RLIMIT_CPU = 2

    rouge = MagicMock()
    rouge.Rouge = MagicMock

    chardet = MagicMock()

    charset_normalizer = MagicMock()

    MOCK_MODULES = {
        'minisweagent': minisweagent,
        'minisweagent.run': minisweagent.run,
        'minisweagent.run.utils': minisweagent.run.utils,
        'minisweagent.run.utils.batch_instances': minisweagent.run.utils.batch_instances,
        'minisweagent.run.extra': minisweagent.run.extra,
        'minisweagent.run.extra.utils': minisweagent.run.extra.utils,
        'minisweagent.run.extra.utils.batch_progress': minisweagent.run.extra.utils.batch_progress,
        'minisweagent.run.extra.run_batch': minisweagent.run.extra.run_batch,
        'minisweagent.config': minisweagent.config,
        'orjson': orjson,
        'docker': docker,
        'docker.errors': docker.errors,
        'transformers': transformers,
        'transformers.generation': transformers.generation,
        'transformers.generation.stopping_criteria': transformers.generation.stopping_criteria,
        'fcntl': fcntl,
        'tqdm': tqdm,
        'tqdm.auto': tqdm.auto,
        'datasets': datasets,
        'datasets.features': datasets.features,
        'datasets.utils': datasets.utils,
        'datasets.utils.logging': datasets.utils.logging,
        'scipy': scipy,
        'scipy.stats': scipy.stats,
        'scipy.integrate': scipy.integrate,
        'jieba': jieba,
        'rouge_chinese': rouge_chinese,
        'mmengine': mmengine,
        'mmengine.registry': mmengine.registry,
        'mmengine.config': mmengine.config,
        'mmengine.fileio': mmengine.fileio,
        'mmengine.utils': mmengine.utils,
        'mmengine.logging': mmengine.logging,
        'mmengine.hub': mmengine.hub,
        'mmengine.dist': mmengine.dist,
        'mmengine.device': mmengine.device,
        'mmengine.visualization': mmengine.visualization,
        'mmengine.optim': mmengine.optim,
        'mmengine.runner': mmengine.runner,
        'mmengine.structures': mmengine.structures,
        'mmengine_lite': mmengine_lite,
        'mmengine_lite.logging': mmengine_lite.logging,
        'evaluate': evaluate,
        'nltk': nltk,
        'nltk.translate': nltk.translate,
        'nltk.translate.bleu_score': nltk.translate.bleu_score,
        'nltk.tokenize': nltk.tokenize,
        'accelerate': accelerate,
        'peft': peft,
        'bitsandbytes': bitsandbytes,
        'trl': trl,
        'einops': einops,
        'func_timeout': func_timeout,
        'fuzzywuzzy': fuzzywuzzy,
        'fuzzywuzzy.fuzz': fuzzywuzzy.fuzz,
        'fuzzywuzzy.process': fuzzywuzzy.process,
        'jsonlines': jsonlines,
        'absl': absl,
        'absl.flags': absl.flags,
        'absl.logging': absl.logging,
        'cpm_kernels': cpm_kernels,
        'gradio_client': gradio_client,
        'immutabledict': immutabledict,
        'janus': janus,
        'loguru': loguru,
        'rich': rich,
        'rich.live': rich.live,
        'rich.console': rich.console,
        'rich.table': rich.table,
        'rich.panel': rich.panel,
        'rich.text': rich.text,
        'rich.progress': rich.progress,
        'openai': openai,
        'requests': requests,
        'yaml': yaml,
        'toml': toml,
        'PIL': PIL,
        'PIL.Image': PIL.Image,
        'cv2': cv2,
        'torch': torch,
        'torch.nn': torch.nn,
        'torch.optim': torch.optim,
        'torch.utils': torch.utils,
        'torch.utils.data': torch.utils.data,
        'torch.cuda': torch.cuda,
        'safetensors': safetensors,
        'safetensors.torch': safetensors.torch,
        'sentence_transformers': sentence_transformers,
        'rouge_score': rouge_score,
        'rouge_score.rouge_scorer': rouge_score.rouge_scorer,
        'prompt_toolkit': prompt_toolkit,
        'prompt_toolkit.shortcuts': prompt_toolkit.shortcuts,
        'python_multipart': python_multipart,
        'aiohttp': aiohttp,
        'httpx': httpx,
        'tiktoken': tiktoken,
        'lark': lark,
        'sympy': sympy,
        'Levenshtein': Levenshtein,
        'rapidfuzz': rapidfuzz,
        'rapidfuzz.distance': rapidfuzz.distance,
        'rapidfuzz.distance.Levenshtein': rapidfuzz.distance.Levenshtein,
        'latex2sympy2': latex2sympy2,
        'antlr4': antlr4,
        'pycocoevalcap': pycocoevalcap,
        'pycocoevalcap.bleu': pycocoevalcap.bleu,
        'pycocoevalcap.cider': pycocoevalcap.cider,
        'pycocoevalcap.rouge': pycocoevalcap.rouge,
        'pycocoevalcap.meteor': pycocoevalcap.meteor,
        'pycocoevalcap.tokenizer': pycocoevalcap.tokenizer,
        'pycocotools': pycocotools,
        'pycocotools.coco': pycocotools.coco,
        'pycocotools.cocoeval': pycocotools.cocoeval,
        'jiwer': jiwer,
        'sacrebleu': sacrebleu,
        'bert_score': bert_score,
        'resource': resource,
        'rouge': rouge,
        'chardet': chardet,
        'charset_normalizer': charset_normalizer,
    }

    for mod_name, mock_obj in MOCK_MODULES.items():
        if mod_name not in sys.modules:
            sys.modules[mod_name] = mock_obj

    return len(MOCK_MODULES)


num_mocked = _setup_mocks()
print(f"[conftest] Mocked {num_mocked} modules at module load time", flush=True)
