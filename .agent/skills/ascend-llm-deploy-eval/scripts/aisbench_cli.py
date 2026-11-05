"""Build a standard AISBench perf config using an existing user dataset."""
import csv
import json
import hashlib
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit


def dataset_for(benchmark, case, *, check_files=True):
    key = case.get('dataset_id', 'default')
    datasets = benchmark.get('datasets', {})
    if key not in datasets:
        raise ValueError('Missing benchmark.datasets binding: '+key)
    dataset = dict(datasets[key])
    path = Path(dataset.get('path', ''))
    absolute = path.is_absolute() if check_files else PurePosixPath(str(path).replace('\\', '/')).is_absolute()
    if not absolute or (check_files and not path.is_file()) or path.suffix.lower() not in ('.jsonl', '.csv'):
        raise ValueError('Dataset must be an existing absolute JSONL/CSV path: '+str(path))
    dataset.setdefault('input_column', 'question')
    dataset.setdefault('output_column', 'answer')
    if not check_files:
        return dataset
    if dataset.get('generation_manifest'):
        manifest_path=Path(dataset['generation_manifest'])
        if not manifest_path.is_absolute():
            raise ValueError('Generation manifest must be absolute')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['dataset_sha256']:
            raise ValueError('Generated dataset digest mismatch')
        evidence=manifest['audit']
        if not evidence['exact_lengths']:
            raise ValueError('Generated dataset has unresolved token lengths')
        if case.get('input_tokens',evidence['target_tokens'])!=evidence['target_tokens']:
            raise ValueError('Generated dataset input length differs from case')
        if len(evidence['token_lengths'])<case.get('formal_requests',1):
            raise ValueError('Generated dataset has insufficient audited requests')
    elif 'shared_prefix_ratio' in case:
        raise ValueError('Shared prefix case requires generation manifest evidence')
    if 'shared_prefix_ratio' in case and manifest['parameters']['prefix_ratio']!=case['shared_prefix_ratio']:
        raise ValueError('Generated prefix target differs from case')
    for name in ('input_column', 'output_column'):
        if not isinstance(dataset[name], str) or not dataset[name].isidentifier():
            raise ValueError('Dataset column must be an identifier: '+name)
    return dataset


def check_dataset(dataset, count):
    """Read and validate the requested slice without copying or modifying data."""
    with Path(dataset['path']).open(encoding='utf-8-sig', newline='') as stream:
        rows = csv.DictReader(stream) if Path(dataset['path']).suffix.lower()=='.csv' else (json.loads(line) for line in stream)
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
    if check_files:
        check_dataset(dataset, case['formal_requests'])
    endpoint = urlsplit(service['endpoint'])
    if endpoint.scheme not in ('http', 'https') or not endpoint.hostname:
        raise ValueError('Invalid service endpoint')
    summarizer = 'stable_stage' if case.get('steady_state', False) else 'default_perf'
    fields = dict(attr='service', abbr='service', path=spec['model']['path'],
                  model=spec['model']['served_name'], stream=True, request_rate=case.get('request_rate', 0),
                  retry=0, host_ip=endpoint.hostname, host_port=endpoint.port or (443 if endpoint.scheme=='https' else 80),
                  url=service['endpoint'].rstrip('/')+'/', max_out_len=case['output_tokens'],
                  batch_size=case['concurrency'], trust_remote_code=False,
                  generation_kwargs={'temperature':0,'ignore_eos':False})
    config = folder/'aisbench-config.py'
    text = (
        'from mmengine.config import read_base\n'
        'from ais_bench.benchmark.models import VLLMCustomAPI\n'
        'from ais_bench.benchmark.datasets import CustomDataset\n'
        'from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate\n'
        'from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever\n'
        'from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer\n'
        'from ais_bench.benchmark.partitioners import NaivePartitioner\n'
        'from ais_bench.benchmark.runners.local import LocalRunner\n'
        'from ais_bench.benchmark.tasks import OpenICLApiInferTask\n'
        f'with read_base():\n    from ais_bench.benchmark.configs.summarizers.perf.{summarizer} import summarizer\n'
        f'datasets = [dict(type=CustomDataset, abbr="dataset", path={dataset["path"]!r}, meta_path="", '
        f'reader_cfg=dict(input_columns={[dataset["input_column"]]!r}, output_column={dataset["output_column"]!r}), '
        f'infer_cfg=dict(prompt_template=dict(type=PromptTemplate, template={("{"+dataset["input_column"]+"}")!r}), '
        'retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer)))]\n'
        'models = [dict(type=VLLMCustomAPI, '+', '.join(f'{k}={v!r}' for k,v in fields.items())+')]\n'
        'infer = dict(partitioner=dict(type=NaivePartitioner), runner=dict(type=LocalRunner, max_num_workers=1, task=dict(type=OpenICLApiInferTask)))\n'
        f'work_dir = {str(folder/"outputs")!r}\n'
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
    """Keep native multi-model/dataset results; verify every emitted request group."""
    execution=json.loads((folder/'execution.json').read_text(encoding='utf-8'))
    if execution.get('returncode') != 0:
        raise ValueError('AISBench did not exit successfully')
    files=list((folder/'outputs').glob('*/performances/*/*_details.jsonl'))
    if not files:
        raise ValueError('Missing native AISBench request details')
    for path in files:
        seen=set()
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if row['id'] in seen or row.get('success') is not True or row.get('error_info'):
                raise ValueError('Failed or duplicate native AISBench request')
            seen.add(row['id'])
        if len(seen)!=case['formal_requests']:
            raise ValueError('Native AISBench request count mismatch')
    if not any(p.is_file() for p in (folder/'outputs').glob('*/logs/**/*')):
        raise ValueError('Missing native AISBench worker logs')
    (folder/'validated.json').write_text(json.dumps({'status':'PASSED','mode':'perf',
        'config':'native','request_groups':len(files)},indent=2),encoding='utf-8')
