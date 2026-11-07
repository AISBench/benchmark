"""Exercise worker control flow in pytest's process, without GPU or upstream checkout.

The subprocess tests remain useful integration checks. These tests also make the
same execution paths visible to the repository's ordinary pytest-cov CI job.
"""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from ais_bench.benchmark.tasks.vabench import worker
from ais_bench.benchmark.utils.vabench.data import read_json, write_json


@pytest.mark.parametrize('scenario', [
    'ok', 'npu', 'exception', 'named_error', 'global_error', 'duplicate', 'invalid', 'missing',
])
def test_worker_execute_preserves_scores_failures_and_device_evidence(tmp_path, monkeypatch, scenario):
    source, work, cache = (tmp_path / name for name in ('source', 'work', 'cache'))
    for folder in (source, work, cache):
        folder.mkdir()
    monkeypatch.setattr(sys, 'path', list(sys.path))
    for name in ('VABENCH_CACHE_DIR', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        monkeypatch.setenv(name, 'before-test')
    monkeypatch.setitem(sys.modules, 'torch', ModuleType('torch'))
    package = ModuleType('vabench'); package.__path__ = []
    monkeypatch.setitem(sys.modules, 'vabench', package)
    device = 'npu:0' if scenario == 'npu' else 'cpu'
    monkeypatch.setattr(worker, 'initialize_device', lambda requested: device)
    released = []
    omni = ModuleType('vabench.omni_runtime')
    model = SimpleNamespace(parameters=lambda: [SimpleNamespace(device=device)])
    omni.get_omni_model_and_processor = lambda *a, **kw: (model, object())
    omni.release_omni = lambda: released.append(True)
    monkeypatch.setitem(sys.modules, omni.__name__, omni)
    dimensions = ['third_alignment', 'fourth_qa_audio']
    metadata = [dict(prompt_en=idx, video_list=[str(tmp_path / (idx + '.mp4'))],
                     audio_list=[str(tmp_path / (idx + '.wav'))]) for idx in ('a', 'b')]
    expected = [dict(idx=idx, category='scene') for idx in ('a', 'b')]
    calls = []

    def compute(dimension):
        def score(file, actual_device, submodules, **params):
            items = read_json(file)
            calls.append((dimension, params))
            assert actual_device == device
            assert params['qwen_omni_ckpt'] == str(cache / 'Qwen2.5_omni_7B')
            assert params['local_files_only'] is True
            assert params['temperature'] == (0.2 if dimension.startswith('third_') else 0.0)
            omni.get_omni_model_and_processor()
            if scenario == 'exception':
                raise RuntimeError('fixture inference failed')
            if scenario == 'named_error':
                print('Error processing ' + items[0]['video_list'][0] + ': decode failed')
            if scenario == 'global_error':
                print('FFmpeg failed for /temporary/unknown.mp4: decode failed')
            raw = [dict(video_path=item['video_list'][0], video_results=(
                99 if scenario == 'invalid' else 3 if dimension.startswith('third_') else 0.5))
                for item in items]
            if scenario == 'duplicate':
                raw.append(raw[0])
            return None, [] if scenario == 'missing' else raw
        return score

    for dimension in dimensions:
        module = ModuleType('vabench.' + dimension)
        module.MODEL_CACHE = model
        setattr(module, 'compute_' + dimension, compute(dimension))
        monkeypatch.setitem(sys.modules, module.__name__, module)
    request = dict(source_root=str(source), cache_dir=str(cache), work_dir=str(work),
                   device=device, dimensions=dimensions, grouped={'scene': metadata}, records=expected,
                   params={'temperature': 0.0}, dim_params={'third_alignment': {'temperature': 0.2}},
                   mode='adapter')
    previous_cwd = Path.cwd()
    result = worker.execute(request)
    assert Path.cwd() == previous_cwd
    assert released == [True] and len(calls) == 2
    assert set(read_json(work / 'partial.json')) == set(dimensions)
    assert result['environment']['python'] == sys.version
    for dimension in dimensions:
        detail = result['details'][dimension]
        rows = detail['records']
        assert [row['idx'] for row in rows] == ['a', 'b']
        if scenario in ('ok', 'npu'):
            assert [row['status'] for row in rows] == ['ok', 'ok']
            assert detail['device_observations'][0]['observed_torch_devices'] == [device]
            assert detail['observed_omni_devices'] == [device]
        elif scenario == 'named_error':
            assert [row['status'] for row in rows] == ['backend_error', 'ok']
            assert rows[0]['score'] is None and rows[1]['score'] is not None
            assert detail['errors'][0]['samples'] == ['a']
        else:
            status = {'invalid': 'invalid_score', 'missing': 'missing'}.get(scenario, 'backend_error')
            assert [row['status'] for row in rows] == [status, status]
            assert all(row['score'] is None for row in rows)
    if scenario == 'npu':
        assert all(params['device_map'] == {'': device} and params['attn_implementation'] == 'eager'
                   for _, params in calls)


@pytest.mark.parametrize('hardlink_fallback', [False, True])
def test_official_wrapper_uses_mapping_local_resources_and_unchanged_media(tmp_path, monkeypatch,
                                                                        hardlink_fallback):
    root, sandbox = tmp_path / 'official', tmp_path / 'sandbox'
    (root / 'mapping').mkdir(parents=True)
    sandbox.mkdir()
    (root / 'mapping/final_idx_to_prompt.csv').write_text(
        'idx\tprompt\na\tFull prompt\n', encoding='utf-8')
    media = {}
    for name, suffix in [('video', '.mp4'), ('audio', '.wav')]:
        path = tmp_path / ('a' + suffix); path.write_bytes(b'original media')
        media[name + '_list'] = [str(path)]
    package = ModuleType('vabench'); package.__path__ = []
    utils = ModuleType('vabench.utils'); utils.load_json = read_json
    observations = []

    class VABench:
        def __init__(self, device, info, output):
            assert device == 'cpu' and Path(output) == sandbox
            assert self.load_prompt_mapping() == {'a': 'Full prompt'}
        def evaluate(self, video, audio, name, dimension_list, local, **params):
            assert Path(video, 'a.mp4').read_bytes() == b'original media'
            assert Path(audio, 'a.wav').read_bytes() == b'original media'
            observations.append((package.init_submodules(dimension_list), params, local))
            write_json(sandbox / (name + '_eval_results.json'), {
                dimension_list[0]: {'video_results': [{'video_path': str(Path(video, 'a.mp4')),
                                                      'video_results': 0.5}]}})
    package.VABench = VABench; package.utils = utils
    monkeypatch.setitem(sys.modules, 'vabench', package)
    monkeypatch.setitem(sys.modules, 'vabench.utils', utils)
    if hardlink_fallback:
        def denied(*a, **kw):
            raise OSError('fixture: symlink unavailable')
        monkeypatch.setattr(Path, 'symlink_to', denied)
    request = dict(source_root=str(root), cache_dir=str(tmp_path / 'cache'))
    for _ in range(2):  # Existing links are reused on the second call.
        result = worker._official_scene(request, 'second_clap', [media], sandbox, 'cpu', {'batch': 2})
        assert result[0]['video_results'] == 0.5
    assert observations[0][1:] == ({'batch': 2}, True)
    assert 'second_clap' in observations[0][0]


@pytest.mark.parametrize('failed', [False, True])
def test_worker_entry_writes_success_or_traceback_response(tmp_path, monkeypatch, failed):
    response = tmp_path / 'response.json'
    request = tmp_path / 'request.json'
    write_json(request, {'response': str(response)})
    monkeypatch.setattr(sys, 'argv', ['worker', str(request)])
    def execute(data):
        if failed:
            raise RuntimeError('fixture fatal error')
        return {'details': {}}
    monkeypatch.setattr(worker, 'execute', execute)
    assert worker.main() == int(failed)
    result = read_json(response)
    if failed:
        assert 'fixture fatal error' in result['error']
    else:
        assert result == {'details': {}}
