"""Translate a frozen case to the repository's native Prefix Cache CLI."""
import json
from pathlib import Path


def write_scenario(spec, case, service, folder):
    benchmark = spec['benchmark']
    corpus = Path(benchmark['corpus_path'])
    if not corpus.is_absolute() or not corpus.is_file():
        raise ValueError('Native CLI requires a readable, absolute corpus_path')
    if not case.get('reset_cache', True):
        raise ValueError('Native prefix workflow requires reset_cache=true')
    config = folder/'aisbench-config.py'
    summarizer = 'stable_stage' if case.get('steady_state', False) else 'default_perf'
    config.write_text(
        'import os\n'
        'from ais_bench.benchmark.partitioners import NaivePartitioner\n'
        'from ais_bench.benchmark.runners.local import LocalRunner\n'
        'from ais_bench.benchmark.tasks import OpenICLApiInferTask\n'
        'from ais_bench_prefix_cache.config import build_dataset_config, build_model_config\n'
        f'from ais_bench.benchmark.configs.summarizers.perf.{summarizer} import summarizer\n'
        'scenario = os.environ["AISBENCH_PREFIX_CACHE_SCENARIO"]\n'
        'datasets = [build_dataset_config(scenario)]\n'
        'models = [build_model_config(scenario)]\n'
        'infer = dict(partitioner=dict(type=NaivePartitioner), runner=dict(type=LocalRunner, max_num_workers=1, task=dict(type=OpenICLApiInferTask)))\n'
        'work_dir = os.environ["AISBENCH_PREFIX_CACHE_WORK_DIR"]\n', encoding='utf-8')
    prefill = (service.get('prefill_url') or f"http://127.0.0.1:{service['prefill_port']}").rstrip('/')
    data = {
        'schema_version': '1.0',
        'run': {'run_id': case['id'], 'random_seed': case.get('seed', 1), 'output_dir': str(folder/'prefix')},
        'tokenizer': {'path': spec['model']['path'], 'block_size': benchmark['block_size']},
        'corpus': {'path': str(corpus), 'field': benchmark.get('corpus_field', 'question')},
        'requests': {'count': case['formal_requests'], 'input_length': {'mode': 'fixed', 'value': case['input_tokens']},
                     'output_length': {'mode': 'fixed', 'value': case['output_tokens']}},
        'prefix_cache': {'mode': 'warmup' if case.get('warmup', True) else 'cold',
                         'target_hit_rate': case['prefix_ratio'], 'groups': {'count': case.get('prefix_num', 1)}},
        'service': {'inference_url': service['endpoint'].rstrip('/')+'/v1/completions',
                    'metrics_url': prefill+'/metrics', 'reset_url': prefill+'/reset_prefix_cache',
                    'model': spec['model']['served_name'], 'dp_size': benchmark['prefill_dp'],
                    'timeout_seconds': case['timeout_seconds'], 'assume_empty_cache': False},
        'aisbench': {'config': str(config), 'work_dir': str(folder/'outputs'),
                     'extra_args': {'--num-warmups': 0, '--request-rate': case.get('request_rate', 0)},
                     'model': {'batch_size': case['concurrency'], 'retry': 0,
                               'generation_kwargs': {'temperature': 0, 'ignore_eos': True}}},
    }
    path = folder/'scenario.json'
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return path


def commands(executable, scenario):
    return [('prepare', [executable, 'prepare', '--mode', 'text', '--scenario', str(scenario)]),
            ('benchmark', [executable, 'run', '--scenario', str(scenario)])]
