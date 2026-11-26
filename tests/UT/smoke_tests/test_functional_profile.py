"""Isolated simulator/profile regressions; no benchmark install or real sleeps."""
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import call, patch


ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / 'smoke_tests/test-case'
PERF = CASES / 'run_server_performance'
SIMULATOR = ROOT / 'tools/infer_serve_simulator'
MARKER = 'aisbench-smoke-functional/qwen'
MODEL_CASES = {
    'diff_dataset_kinds': (
        'performance_ceval', 'performance_dataset_custom_with_data_sample',
        'performance_gsm8k', 'performance_mooncake_trace', 'performance_synthetic_gen',
        'preformance_dataset_custom', 'preformance_dataset_custom_with_max_tokens',
    ),
    'stream_apis': (
        'perf_vllm_api_stream_chat_gsm8k', 'perf_vllm_api_stream_chat_multiturn_mtbench',
        'perf_vllm_api_stream_chat_multiturn_sharegpt', 'pref_vllm_api_general_stream_gsm8k',
    ),
}
DATA_CASES = (
    ('diff_dataset_kinds', 'performance_ceval', 'ceval', 'ceval_datasets'),
    ('diff_dataset_kinds', 'performance_gsm8k', 'gsm8k', 'gsm8k_datasets'),
    ('stream_apis', 'perf_vllm_api_stream_chat_gsm8k', 'gsm8k', 'gsm8k_datasets'),
    ('stream_apis', 'pref_vllm_api_general_stream_gsm8k', 'gsm8k', 'gsm8k_datasets'),
)


def config_path(group, case, kind='models', backend='vllm_api'):
    return PERF / group / case / 'ais_bench_configs' / kind / backend / f'{case}.py'


def execute_config(path, fast, **fixtures):
    """Replace imported read_base fixtures via AST, retaining all config logic."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.Import, ast.ImportFrom, ast.With))]
    namespace = deepcopy(fixtures)
    environment = {} if fast is None else {'AISBENCH_SMOKE_FAST': fast}
    with patch.dict(os.environ, environment, clear=True):
        exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


class FunctionalProfileTest(unittest.TestCase):
    def setUp(self):
        # Keep Flask's context modules stable while isolating the generic api package.
        importlib.import_module('flask')
        importlib.import_module('yaml')
        with patch.dict(sys.modules), patch.object(sys, 'path', [str(SIMULATOR)] + sys.path):
            for name in list(sys.modules):
                if name == 'api' or name.startswith('api.'):
                    del sys.modules[name]
            spec = importlib.util.spec_from_file_location('_functional_smoke_service',
                                                         SIMULATOR / 'flask_service.py')
            self.service = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.service)
            self.base = sys.modules['api.base_api']
        self.service.app.config['TESTING'] = True
        self.client = self.service.app.test_client()
        self.config = self.base.load_yaml_config()
        # Nondefault fixture values prove rollback reads YAML, not hardcoded defaults.
        self.config['stream_latency'] = dict(ttft=0.37, tpot=0.023)
        self.config['text_latency'] = dict(e2el=0.71)
        self.config['general']['enable_mtp'] = False
        self.config['random_dataset']['random_content'] = False
        self.original = deepcopy(self.config)
        self.yaml_bytes = (SIMULATOR / 'api/api_config.yaml').read_bytes()
        self.loader = patch.object(self.base, 'load_yaml_config', return_value=self.config)
        self.loader.start()
        self.addCleanup(self.loader.stop)

    def post(self, endpoint, model=MARKER, stream=False, **kwargs):
        body = dict(max_tokens=4, ignore_eos=True, stream=stream)
        if model is not None:
            body['model'] = model
        body.update(kwargs.pop('body', {}))
        return self.client.post(endpoint, json=body, **kwargs)

    def test_exact_marker_and_opt_in_are_both_required(self):
        for endpoint in ('/v1/chat/completions', '/v1/completions'):
            for fast in (None, '0', '1', 'true'):
                for model in (None, 'qwen', MARKER, MARKER + '-other'):
                    for stream in (False, True):
                        with self.subTest(endpoint=endpoint, fast=fast, model=model, stream=stream):
                            environment = {} if fast is None else {'AISBENCH_SMOKE_FAST': fast}
                            enabled = fast == '1' and model == MARKER
                            with patch.dict(os.environ, environment, clear=True), patch('time.sleep') as sleep:
                                response = self.post(endpoint, model, stream, buffered=True)
                            self.assertEqual(response.status_code, 200)
                            expected = ([0.005] + [0.002] * 3 if enabled else [0.37] + [0.046] * 3) \
                                if stream else [0.005 if enabled else 0.71]
                            self.assertEqual(sleep.call_args_list, [call(value) for value in expected])
                            self.assertEqual(self.config, self.original)
        self.assertEqual((SIMULATOR / 'api/api_config.yaml').read_bytes(), self.yaml_bytes)

    def test_text_and_stream_content_chunks_and_usage_are_preserved(self):
        for endpoint in ('/v1/chat/completions', '/v1/completions'):
            for mtp in (False, True):
                self.config['general']['enable_mtp'] = mtp
                self.config['random_dataset']['tokens_per_chunk'] = 2
                texts, streams = [], []
                for fast in ('0', '1'):
                    with patch.dict(os.environ, {'AISBENCH_SMOKE_FAST': fast}), patch('time.sleep'):
                        texts.append(self.post(endpoint).get_json())
                        streams.append(self.post(endpoint, stream=True, buffered=True).get_data(as_text=True))
                self.assertEqual(streams[0], streams[1])
                chunks = streams[1].split('\n\n')
                self.assertEqual(chunks[-1], 'data: [DONE]')
                events = [json.loads(chunk.removeprefix('data: ')) for chunk in chunks[:-1]]
                self.assertEqual(len(events), 2 if mtp else 4)
                self.assertEqual(events[-1]['usage'], {'completion_tokens': 4})
                self.assertTrue(all('usage' not in event for event in events[:-1]))
                if endpoint == '/v1/chat/completions':
                    for result, e2el in zip(texts, (0.71, 0.005)):
                        self.assertEqual(result.pop('prefill_time'), e2el / 4)
                        self.assertEqual(result.pop('decode_time_arr'), [e2el / 4] * 3)
                        self.assertEqual(result['usage'], {'completion_tokens': 4})
                    content = ''.join(event['choices'][0]['delta']['content'] for event in events)
                    self.assertEqual(texts[0]['choices'][0]['message']['content'], content)
                else:
                    content = ''.join(event['choices'][0]['text'] for event in events)
                    self.assertEqual(texts[0]['choices'][0]['text'], content)
                    self.assertNotIn('usage', texts[0])  # Preserve existing completion response shape.
                self.assertEqual(content, 'A ' * 4)
                self.assertEqual(texts[0], texts[1])

    def test_interleaved_requests_do_not_share_profile_or_mutate_yaml(self):
        for endpoint in ('/v1/chat/completions', '/v1/completions'):
            with self.subTest(endpoint=endpoint), \
                    patch.dict(os.environ, {'AISBENCH_SMOKE_FAST': '1'}), patch('time.sleep') as sleep:
                stream = self.post(endpoint, stream=True, buffered=False)
                self.post(endpoint, model='qwen')
                stream.get_data()  # Finish marked stream after a normal request intervenes.
                stream.close()
                self.post(endpoint, model='qwen')
                self.assertEqual(sleep.call_args_list,
                                 [call(0.005), call(0.71)] + [call(0.002)] * 3 + [call(0.71)])
                self.assertEqual(self.config, self.original)

    def test_unmarked_function_call_keeps_original_timing_and_shape(self):
        with patch.dict(os.environ, {'AISBENCH_SMOKE_FAST': '1'}), patch('time.sleep') as sleep:
            response = self.post('/v1/chat/completions', model='qwen', body={'tools': []})
        sleep.assert_called_once_with(0.71)
        message = response.get_json()['choices'][0]['message']
        self.assertEqual(message['tool_calls'][0]['function'], {'name': 'func', 'arguments': 'A ' * 4})

    def test_non_openai_handlers_ignore_marker_even_when_enabled(self):
        routes = ('/generate', '/generate_stream', '/v2/models/qwen/generate',
                  '/v2/models/qwen/generate_stream', '/infer', '/infer_token')
        for route in routes:
            for stream in (False, True):
                with self.subTest(route=route, stream=stream), \
                        patch.dict(os.environ, {'AISBENCH_SMOKE_FAST': '1'}), patch('time.sleep') as sleep:
                    response = self.client.post(route, json=dict(
                        model=MARKER, stream=stream,
                        parameters=dict(max_new_tokens=4, ignore_eos=True)), buffered=True)
                    self.assertEqual(response.status_code, 200)
                    is_stream = route.endswith('_stream') or (route.startswith('/infer') and stream)
                    expected = [0.37] + [0.046] * 3 if is_stream else [0.71]
                    self.assertEqual(sleep.call_args_list, [call(value) for value in expected])
                    self.assertEqual(self.config, self.original)


class FunctionalConfigTest(unittest.TestCase):
    def test_selected_models_roll_back_fully_and_only_cap_output_length(self):
        for group, cases in MODEL_CASES.items():
            for case in cases:
                original_length = 20 if case == 'performance_synthetic_gen' else \
                    300 if case == 'performance_dataset_custom_with_data_sample' else 200
                original_batch = 100 if case == 'performance_synthetic_gen' else \
                    1 if case == 'performance_dataset_custom_with_data_sample' else 200
                expected = dict(sentinel='base', model='qwen', request_rate=50,
                                max_out_len=original_length, batch_size=original_batch,
                                generation_kwargs=dict(ignore_eos=True))
                for fast in (None, '0', '1', 'true'):
                    with self.subTest(case=case, fast=fast):
                        result = execute_config(config_path(group, case), fast,
                                                models=[dict(sentinel='base')])['models'][0]
                        wanted = dict(expected)
                        if fast == '1':
                            wanted.update(model=MARKER, max_out_len=min(original_length, 32))
                        self.assertEqual(result, wanted)

    def test_sample_reductions_and_ceval_subjects_are_deterministic_and_reversible(self):
        for group, case, dataset, variable in DATA_CASES:
            for fast in (None, '0', '1', 'true'):
                with self.subTest(case=case, fast=fast):
                    subjects = [dict(name=f'subject-{i}', reader_cfg={})
                                for i in range(5 if dataset == 'ceval' else 1)]
                    result = execute_config(config_path(group, case, 'datasets', dataset), fast,
                                            **{variable: subjects})[variable]
                    expected = deepcopy(subjects[:3] if dataset == 'ceval' and fast == '1' else subjects)
                    for subject in expected:
                        subject['reader_cfg']['test_range'] = '[0:20]' if fast == '1' else '[0:100]'
                    self.assertEqual(result, expected)

    def test_markers_and_workload_switches_are_confined_to_approved_configs(self):
        allowed = {config_path(group, case) for group, cases in MODEL_CASES.items() for case in cases}
        allowed.update(config_path(group, case, 'datasets', dataset)
                       for group, case, dataset, _ in DATA_CASES)
        switched = set()
        for path in CASES.rglob('*.py'):
            if 'ais_bench_configs' not in path.parts:
                continue
            source = path.read_text(encoding='utf-8')
            if MARKER in source or 'AISBENCH_SMOKE_FAST' in source:
                switched.add(path)
        # Protect reuse/interrupt, pressure, high concurrency, optional capabilities,
        # and non-OpenAI models/datasets, not just the explicitly selected subtrees.
        self.assertEqual(switched, allowed)

    def test_protected_models_keep_full_workloads_with_fast_enabled(self):
        roots = (PERF / 'high_concurrency', PERF / 'run_with_reuse',
                 CASES / 'run_server_pressure')
        paths = [path for root in roots for path in root.rglob('*.py') if 'models' in path.parts]
        paths += [path for path in (PERF / 'stream_apis').rglob('*.py')
                  if 'models' in path.parts and 'vllm_api' not in path.parts]
        self.assertGreaterEqual(len(paths), 12)
        for path in paths:
            with self.subTest(path=path):
                normal = execute_config(path, '0', models=[{}])['models']
                fast = execute_config(path, '1', models=[{}])['models']
                self.assertEqual(normal, fast)
                self.assertNotEqual(fast[0].get('model'), MARKER)
                self.assertEqual(fast[0]['generation_kwargs'], dict(ignore_eos=True))
                if 'high_concurrency' in path.parts:
                    self.assertEqual((fast[0]['max_out_len'], fast[0]['batch_size']), (8, 800))
                elif path.stem.startswith('default_ctrl_c'):
                    self.assertEqual((fast[0]['max_out_len'], fast[0]['batch_size']), (100, 10))
                elif path.stem == 'perf_recalculate_stable_stage':
                    self.assertEqual((fast[0]['max_out_len'], fast[0]['batch_size']), (200, 20))
                else:
                    self.assertEqual((fast[0]['max_out_len'], fast[0]['batch_size']), (200, 200))


if __name__ == '__main__':
    unittest.main()
