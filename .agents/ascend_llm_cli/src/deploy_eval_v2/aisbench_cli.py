"""Build a standard AISBench perf config using external data or AISBench native synthetic data."""
import csv
import json
import hashlib
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


def dataset_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def synthetic_config(case):
    """Translate confirmed targets only; AISBench owns all sample generation."""
    from decimal import Decimal
    import math
    size=case.get('input_tokens');count=case.get('formal_requests')
    ratio=case.get('shared_prefix_ratio',0)
    if type(size) is not int or not 1 <= size < 2**20:
        raise ValueError('AISBench synthetic requires input_tokens in [1, 2**20)')
    if type(count) is not int or not 1 <= count < 2**20:
        raise ValueError('AISBench synthetic requires formal_requests in [1, 2**20)')
    if type(ratio) not in (int,float) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError('Invalid synthetic shared_prefix_ratio')
    prefix=int(Decimal(size)*Decimal(str(ratio)))
    return {'Type':'tokenid','RequestCount':count,'TrustRemoteCode':False,
            'TokenIdConfig':{'RequestSize':size,'PrefixLen':prefix}}


def check_synthetic_support(case):
    """Check the installed AISBench implementation, without generating another sample set."""
    import inspect
    from ais_bench.benchmark.datasets import SyntheticDataset
    config=synthetic_config(case)
    SyntheticDataset._check_synthetic_config(config)
    source=inspect.getsource(SyntheticDataset.load)
    if 'PrefixLen' not in source:
        raise ValueError('Installed AISBench SyntheticDataset has no inspectable PrefixLen support')
    module=Path(inspect.getfile(SyntheticDataset))
    return {'type':'aisbench_synthetic','config':config,'module':str(module),
            'module_sha256':dataset_digest(module),'status':'CONFIG_CHECKED',
            'sample_generation':'PENDING_NATIVE_LOADER_VALIDATION'}


def audit_synthetic_result(folder, case, model_path, tokenizer=None):
    """Compatibility entry point for the bounded postflight auditor."""
    from .native_postflight import audit
    return audit(folder,case,model_path,tokenizer=tokenizer)


def dataset_for(benchmark, case, *, check_files=True):
    key = case.get('dataset_id', 'default')
    datasets = benchmark.get('datasets', {})
    if key not in datasets:
        raise ValueError('Missing benchmark.datasets binding: '+key)
    dataset = dict(datasets[key])
    if dataset.get('type') == 'aisbench_synthetic':
        if set(dataset) != {'type'}:
            raise ValueError('AISBench synthetic binding must not contain external paths or generator settings')
        return {**dataset,'config':synthetic_config(case),'input_column':'question','output_column':'answer'}
    path = Path(dataset.get('path', ''))
    absolute = path.is_absolute() if check_files else PurePosixPath(str(path).replace('\\', '/')).is_absolute()
    if not absolute or (check_files and not path.is_file()) or path.suffix.lower() not in ('.jsonl', '.csv'):
        raise ValueError('Dataset must be an existing absolute JSONL/CSV path: '+str(path))
    dataset.setdefault('input_column', 'question')
    dataset.setdefault('output_column', 'answer')
    # The binding itself is knowable even when data lives only on the remote host.
    # Do not create containers merely to discover a missing evidence path.
    manifest_value = dataset.get('generation_manifest')
    if 'shared_prefix_ratio' in case and not manifest_value:
        raise ValueError('Shared prefix case requires generation manifest evidence')
    if manifest_value:
        if not isinstance(manifest_value, str):
            raise ValueError('Generation manifest must be an absolute path string')
        manifest_absolute = (Path(manifest_value).is_absolute() if check_files else
                             PurePosixPath(manifest_value.replace('\\', '/')).is_absolute())
        if not manifest_absolute:
            raise ValueError('Generation manifest must be absolute')
    if not check_files:
        return dataset
    if dataset.get('generation_manifest'):
        manifest_path=Path(dataset['generation_manifest'])
        if not manifest_path.is_absolute():
            raise ValueError('Generation manifest must be absolute')
        from .json_io import load_json
        manifest=load_json(manifest_path, 16*1024*1024)
        if dataset_digest(path)!=manifest['dataset_sha256']:
            raise ValueError('Generated dataset digest mismatch')
        evidence=manifest['audit']
        if not evidence['exact_lengths']:
            raise ValueError('Generated dataset has unresolved token lengths')
        if case.get('input_tokens',evidence['target_tokens'])!=evidence['target_tokens']:
            raise ValueError('Generated dataset input length differs from case')
        if len(evidence['token_lengths'])<case.get('formal_requests',1):
            raise ValueError('Generated dataset has insufficient audited requests')
    if 'shared_prefix_ratio' in case and manifest['parameters']['prefix_ratio']!=case['shared_prefix_ratio']:
        raise ValueError('Generated prefix target differs from case')
    for name in ('input_column', 'output_column'):
        if not isinstance(dataset[name], str) or not dataset[name].isidentifier():
            raise ValueError('Dataset column must be an identifier: '+name)
    return dataset


def check_dataset(dataset, count):
    """Read and validate the requested slice without copying or modifying data."""
    with Path(dataset['path']).open(encoding='utf-8-sig', newline='') as stream:
        def bounded_lines():
            while True:
                line=stream.readline(1024*1024+1)
                if not line:
                    break
                if len(line.encode('utf-8'))>1024*1024:
                    raise ValueError('Dataset row exceeds 1 MiB')
                yield line
        rows = csv.DictReader(bounded_lines()) if Path(dataset['path']).suffix.lower()=='.csv' else (json.loads(line) for line in bounded_lines())
        seen = 0
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get(dataset['input_column']), str) or dataset['output_column'] not in row:
                raise ValueError('Dataset row lacks configured input/output columns')
            seen += 1
            if seen >= count: break
    if seen < count:
        raise ValueError(f'Dataset has {seen} rows; requested {count}')


def write_config(spec, case, service, folder, *, check_files=True):
    require_supported_mode(case)
    dataset = dataset_for(spec['benchmark'], case, check_files=check_files)
    synthetic=dataset.get('type')=='aisbench_synthetic'
    if check_files and not synthetic:
        check_dataset(dataset, case['formal_requests'])
    endpoint = urlsplit(service['endpoint'])
    if endpoint.scheme not in ('http', 'https') or not endpoint.hostname:
        raise ValueError('Invalid service endpoint')
    summarizer = 'stable_stage' if case.get('steady_state', False) else 'default_perf'
    # AISBench 3.1 counts total attempts (range(self.retry)); one sends the
    # initial request with no internal retry. Coordinator max_retries is separate.
    fields = dict(attr='service', abbr='service', path=spec['model']['path'],
                  model=spec['model']['served_name'], stream=True, request_rate=case.get('request_rate', 0),
                  retry=1, host_ip=endpoint.hostname, host_port=endpoint.port or (443 if endpoint.scheme=='https' else 80),
                  url=service['endpoint'].rstrip('/')+'/', max_out_len=case['output_tokens'],
                  batch_size=case['concurrency'], trust_remote_code=False,
                  generation_kwargs={'temperature':0,'ignore_eos':False})
    dataset_class='SyntheticDataset' if synthetic else 'CustomDataset'
    dataset_args=(f'config={dataset["config"]!r}, model_path={spec["model"]["path"]!r}' if synthetic else
                  f'path={dataset["path"]!r}, meta_path=""')
    config = folder/'aisbench-config.py'
    text = (
        'from mmengine.config import read_base\n'
        'from ais_bench.benchmark.models import VLLMCustomAPI\n'
        f'from ais_bench.benchmark.datasets import {dataset_class}\n'
        'from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate\n'
        'from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever\n'
        'from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer\n'
        'from ais_bench.benchmark.partitioners import NaivePartitioner\n'
        'from ais_bench.benchmark.runners.local import LocalRunner\n'
        'from ais_bench.benchmark.tasks import OpenICLApiInferTask\n'
        f'with read_base():\n    from ais_bench.benchmark.configs.summarizers.perf.{summarizer} import summarizer\n'
        f'datasets = [dict(type={dataset_class}, abbr="dataset", {dataset_args}, '
        f'reader_cfg=dict(input_columns={[dataset["input_column"]]!r}, output_column={dataset["output_column"]!r}), '
        f'infer_cfg=dict(prompt_template=dict(type=PromptTemplate, template={("{"+dataset["input_column"]+"}")!r}), '
        'retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer)))]\n'
        'models = [dict(type=VLLMCustomAPI, '+', '.join(f'{k}={v!r}' for k,v in fields.items())+')]\n'
        'infer = dict(partitioner=dict(type=NaivePartitioner), runner=dict(type=LocalRunner, max_num_workers=1, task=dict(type=OpenICLApiInferTask)))\n'
    )
    config.write_text(text, encoding='utf-8')
    (folder/'dataset.json').write_text(json.dumps(dataset, indent=2), encoding='utf-8')
    return config


def commands(executable, config, case, folder):
    require_supported_mode(case)
    warmups = case.get('num_warmups', 1) if case.get('warmup', True) else 0
    # Normal mode writes AISBench worker logs under --work-dir; never use --debug.
    return [('benchmark', [executable, str(config), '--mode', 'perf',
             '--work-dir', str(folder/'outputs'), '--num-prompts', str(case['formal_requests']),
             '--num-warmups', str(warmups)])]


def require_supported_mode(case):
    """Task modes are independent of the coordinator FULL/scripts mode."""
    mode = case.get('mode', 'perf')
    if mode != 'perf':
        raise ValueError('AISBench test mode '+repr(mode)+' is reserved but not implemented; no perf fallback')


def validate_native_result(folder, case):
    """Compatibility entry point for the streaming native result validator."""
    from .native_results import validate_native_result as validate
    return validate(folder,case)
