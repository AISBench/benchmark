"""CPU-only contract tests: real FFmpeg, real subprocess and data audits."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from ais_bench.benchmark.utils.vabench import DIMENSIONS, MLLM_DIMENSIONS
from ais_bench.benchmark.utils.vabench.data import (
    PROTOCOL_PATH, catalog, catalog_videos, digest, media_tool, prepare, read_json,
    select_indices, select_prepared, verify_prepared, write_json)
from ais_bench.benchmark.tasks.vabench.engine import run, successful
from ais_bench.benchmark.summarizers.vabench.results import (
    build_report, compare, normalize_records, summarize_dimension)
from ais_bench.benchmark.tasks.vabench.worker import failed_sample_ids, has_backend_error


SOURCE = Path(__file__).resolve().parents[4] / 'VABench'


def test_pinned_catalog():
    if not SOURCE.exists():
        pytest.skip('Local official checkout is not available')
    standard = catalog(SOURCE)
    assert len(standard) == 778
    assert len({e['idx'] for e in standard}) == 778
    assert len({e['category'] for e in standard}) == 24
    assert {'easy_human_language', 'hard_human_language', 'hard_Exact_Time_Synchronization'} <= {
        e['category'] for e in standard}


def test_pinned_data_rejects_drift(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    with pytest.raises(ValueError, match='differs from pinned'):
        catalog(root)


def test_explicit_single_sample_smoke_catalog():
    if not SOURCE.exists():
        pytest.skip('Local official checkout is not available')
    entries = catalog(SOURCE, 'smoke', sample_id='kitten curled up meowing')
    assert len(entries) == 1 and entries[0]['idx'] == 'kitten curled up meowing'
    assert len(catalog(SOURCE, 'standard')) == 778
    with pytest.raises(ValueError, match='requires one sample_id'):
        catalog(SOURCE, 'smoke')
    with pytest.raises(ValueError, match='requires one sample_id'):
        catalog(SOURCE, 'smoke', sample_id='not-an-official-prompt')
    with pytest.raises(ValueError, match='only supported'):
        catalog(SOURCE, 'standard', sample_id='kitten curled up meowing')


def test_catalog_videos_selects_pinned_samples(tmp_path):
    if not SOURCE.exists():
        pytest.skip('Local official checkout is not available')
    idx = catalog(SOURCE)[0]['idx']
    (tmp_path / f'{idx}.mp4').write_bytes(b'')
    selected = catalog_videos(SOURCE, tmp_path)
    assert [entry['idx'] for entry in selected] == [idx]
    (tmp_path / 'not an official sample.mp4').write_bytes(b'')
    with pytest.raises(ValueError, match='outside the pinned 778'):
        catalog_videos(SOURCE, tmp_path)


def test_catalog_videos_rejects_ambiguous_stems(tmp_path):
    if not SOURCE.exists():
        pytest.skip('Local official checkout is not available')
    idx = catalog(SOURCE)[0]['idx']
    for name in ('a', 'b'):
        folder = tmp_path / name
        folder.mkdir()
        (folder / f'{idx}.mp4').write_bytes(b'')
    with pytest.raises(ValueError, match='Ambiguous MP4 stems'):
        catalog_videos(SOURCE, tmp_path)


@pytest.fixture
def entry():
    return dict(idx='中文 sample with spaces', category='scene', metadata=dict(
        prompt_en='A test tone with a visual scene.', prompt_audio='A tone.',
        prompt_vision='A color image.', dimension=list(DIMENSIONS),
        auxiliary_info_audio=['Is a tone audible?'], auxiliary_info_vision=['Is color visible?']))


@pytest.fixture
def video(tmp_path, entry):
    if not shutil.which(media_tool('ffmpeg')) or not shutil.which(media_tool('ffprobe')):
        pytest.skip('FFmpeg and ffprobe are required for media tests')
    folder = tmp_path / 'videos'
    folder.mkdir()
    file = folder / (entry['idx'] + '.mp4')
    subprocess.run([
        media_tool('ffmpeg'), '-v', 'error', '-y', '-f', 'lavfi', '-i',
        'color=c=blue:s=64x64:r=24:d=0.3', '-f', 'lavfi', '-i',
        'sine=frequency=440:sample_rate=48000:duration=0.3',
        '-c:v', 'mpeg4', '-c:a', 'aac', '-shortest', str(file)], check=True)
    return file


def test_real_media_preparation(tmp_path, entry, video):
    out = tmp_path / 'prepared'
    manifest = prepare([entry], video.parent, out)
    assert manifest['subset_size'] == 1
    assert manifest['records'][0]['short_video'] is True
    verified, grouped = verify_prepared(out)
    assert verified['input_signature'] == manifest['input_signature']
    assert grouped['scene'][0]['prompt_en'] == entry['metadata']['prompt_en']
    wave = Path(manifest['records'][0]['audio_path'])
    import wave as wave_module
    with wave_module.open(str(wave)) as audio:
        assert audio.getframerate() == 48000
        assert audio.getsampwidth() == 2
    with pytest.raises(FileExistsError):
        prepare([entry], video.parent, out)
    wave.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Input changed'):
        verify_prepared(out)


def test_verify_prepared_hashes_only_selected_records(tmp_path, entry, video):
    second = copy.deepcopy(entry)
    second['idx'] = 'unselected sample'
    shutil.copyfile(video, video.parent / (second['idx'] + '.mp4'))
    prepared = tmp_path / 'prepared'
    prepare([entry, second], video.parent, prepared, subset='custom')
    (video.parent / (second['idx'] + '.mp4')).unlink()
    with pytest.raises(FileNotFoundError):
        verify_prepared(prepared)
    manifest, _ = verify_prepared(prepared, '[:1]')
    assert [record['idx'] for record in manifest['records']] == [entry['idx'], second['idx']]


def test_smoke_cli_labels_single_sample(tmp_path, entry, video, monkeypatch):
    import ais_bench.tools.vabench.cli as cli
    monkeypatch.setattr(cli, 'catalog', lambda *a, **kw: [entry])
    out = tmp_path / 'smoke'
    assert cli.main(['prepare', '--source-root', 'fixture', '--subset', 'smoke',
                     '--sample-id', entry['idx'], '--videos', str(video.parent),
                     '--output', str(out)]) == 0
    manifest, grouped = verify_prepared(out)
    assert manifest['evaluation_scope'] == 'smoke'
    assert manifest['subset_size'] == 1
    report = build_report(manifest, {}, {})
    assert report['evaluation_scope'] == 'smoke'
    assert 'not full 778-set' in report['scope_note']
    with pytest.raises(ValueError, match='exactly one'):
        prepare([entry, entry], video.parent, tmp_path / 'invalid', subset='smoke')


def test_missing_ambiguous_and_silent_input(tmp_path, entry, video):
    with pytest.raises(ValueError, match='Missing MP4'):
        prepare([dict(entry, idx='missing')], video.parent, tmp_path / 'out1')
    extra = video.parent / 'duplicate'
    extra.mkdir()
    shutil.copyfile(video, extra / video.name)
    with pytest.raises(ValueError, match='ambiguous MP4'):
        prepare([entry], video.parent, tmp_path / 'out2')
    (extra / video.name).unlink()
    silent = video.parent / 'silent.mp4'
    subprocess.run([media_tool('ffmpeg'), '-v', 'error', '-y', '-i', str(video),
                    '-an', '-c:v', 'copy', str(silent)], check=True)
    with pytest.raises(ValueError, match='must contain video and audio'):
        prepare([dict(entry, idx='silent')], video.parent, tmp_path / 'out3')


def _report(score=0.0, status='ok', dim='second_clap'):
    records = [dict(idx='x', category='scene', status=status, score=score, raw={})]
    return build_report(dict(input_signature='same', records=[{'idx': 'x'}]),
                        {dim: {'records': records}},
                        {k: 'same' for k in ('protocol_sha256', 'source_signature', 'integration_signature',
                                             'weights_signature', 'scoring_signature')})


def test_comparison_zero_negative_and_threshold():
    reference = _report(0)
    assert compare(reference, _report(0), ['second_clap'])['passed']
    assert not compare(reference, _report(0.01), ['second_clap'])['passed']
    assert compare(_report(-0.5), _report(-0.502), ['second_clap'])['passed']
    # Boundary uses strict < 1%, rather than <= 1%.
    assert not compare(_report(100), _report(101), ['second_clap'])['passed']
    assert not compare(_report(), _report())['passed']  # fifteen dims required by default
    with pytest.raises(ValueError, match='Invalid tolerance'):
        compare(reference, reference, relative=float('inf'))


@pytest.mark.parametrize('mutation', ['missing', 'failed', 'weights', 'media', 'duplicate', 'nan'])
def test_alignment_cannot_pass_invalid_runs(mutation):
    left, right = _report(0.5), _report(0.5)
    if mutation == 'missing':
        right['details']['second_clap']['records'][0]['status'] = 'missing'
    elif mutation == 'failed':
        for report in (left, right):
            report['details']['second_clap']['records'][0]['status'] = 'backend_error'
    elif mutation == 'weights':
        right['provenance']['weights_signature'] = 'different'
    elif mutation == 'media':
        right['input_signature'] = 'different'
    elif mutation == 'duplicate':
        right['details']['second_clap']['records'] *= 2
    elif mutation == 'nan':
        right['details']['second_clap']['records'][0]['score'] = float('nan')
    assert not compare(left, right, ['second_clap'])['passed']


def test_statuses_and_weighted_mean():
    expected = [dict(idx='x', category='a'), dict(idx='y', category='b')]
    rows = normalize_records('second_lsa', [{'video_path': 'x.mp4', 'video_results': 0}], expected)
    assert [r['status'] for r in rows] == ['ok', 'missing']
    summary = summarize_dimension(rows)
    assert summary['mean'] == 0 and summary['missing'] == 1
    rows = [dict(idx=str(i), category=cat, status='ok', score=score)
            for i, (cat, score) in enumerate([('a', 0), ('a', 0), ('b', 3)])]
    assert summarize_dimension(rows)['mean'] == 1  # not (0+3)/2
    assert normalize_records('second_clap', [{'video_path': 'x.mp4', 'video_results': 0}],
                             expected, backend_errors=True)[0]['status'] == 'backend_error'
    assert has_backend_error('Error processing clip: failed')
    assert has_backend_error('Traceback (most recent call last):')
    assert not has_backend_error('Evaluation completed normally')
    assert not has_backend_error('人脸检测失败或评估失败: example.mp4')


def test_named_sample_failure_does_not_poison_scene():
    expected = [dict(idx='bad', category='c'), dict(idx='good', category='c')]
    rows = normalize_records('second_desync', [
        dict(video_path='/v/bad.mp4', video_results=1.0),
        dict(video_path='/v/good.mp4', video_results=0.25)],
        expected, failed_ids={'bad'})
    assert [r['status'] for r in rows] == ['backend_error', 'ok']
    assert rows[0]['score'] is None and rows[1]['score'] == 0.25
    assert failed_sample_ids(
        'Error processing /v/bad.mp4: tensor mismatch\nTraceback (most recent call last):\n') == {'bad'}
    # tqdm may write its progress bar and the error on the same line.
    assert failed_sample_ids(' 80%|####  | 4/5]Error processing /v/bad.mp4: boom\n') == {'bad'}
    assert failed_sample_ids('Error processing sample /v/bad.mp4: boom\n') == {'bad'}
    assert failed_sample_ids('Error in second_desync evaluation: boom\n') == set()


def test_mixed_decoder_errors_are_all_attributed_before_releasing_scene():
    from ais_bench.benchmark.tasks.vabench.worker import attribute_backend_errors
    expected = [dict(idx=x, category='c') for x in ('a', 'b', 'good')]
    text = ('Error processing /audio/a.wav: model failed\n'
            'FFmpeg failed for /audio/b.wav: decoder failed\n')
    attribution = attribute_backend_errors(text, expected)
    assert attribution['failed_ids'] == {'a', 'b'}
    assert attribution['scene_unverified'] is False
    raw = [dict(video_path=x + '.mp4', video_results=v)
           for x, v in [('a', 0), ('b', 0), ('good', 0.6)]]
    rows = normalize_records('second_clap', raw, expected,
                             backend_errors=attribution['scene_unverified'],
                             failed_ids=attribution['failed_ids'])
    assert [r['status'] for r in rows] == ['backend_error', 'backend_error', 'ok']
    assert summarize_dimension(rows)['mean'] == 0.6


@pytest.mark.parametrize('extra', [
    'SoundFile read failed for /tmp/random_audio.wav: invalid header\n',
    'Error in second_clap evaluation: model state invalid\n',
    ' 80%|###| 4/5 [00:01<00:01]Error in second_clap evaluation: invalid state\n',
    'Completed next sample\nTraceback (most recent call last):\n'])
def test_unattributed_error_keeps_scene_protection_even_with_named_error(extra):
    from ais_bench.benchmark.tasks.vabench.worker import attribute_backend_errors
    result = attribute_backend_errors('Error processing /audio/a.wav: failed\n' + extra,
                                      [dict(idx='a'), dict(idx='good')])
    assert result['failed_ids'] == {'a'}
    assert result['scene_unverified'] is True


def test_named_traceback_does_not_poison_other_samples():
    from ais_bench.benchmark.tasks.vabench.worker import attribute_backend_errors
    result = attribute_backend_errors(
        'Error processing /video/bad.mp4: failed\nTraceback (most recent call last):\n'
        '  File "score.py", line 1\nRuntimeError: mismatch\n', [dict(idx='bad'), dict(idx='good')])
    assert result['failed_ids'] == {'bad'} and not result['scene_unverified']


def test_metric_group_count():
    from ais_bench.benchmark.tasks.vabench.eval import _metric_group_count
    assert _metric_group_count(DIMENSIONS) == 9
    assert _metric_group_count(MLLM_DIMENSIONS) == 1
    assert _metric_group_count(['second_clap']) == 1
    assert _metric_group_count([]) == 0


def test_num_prompts_range_selects_prefix():
    assert select_indices(7, '[:2]') == [0, 1]
    assert select_indices(7, '[4:5]') == [4]
    assert select_indices(7, None) == list(range(7))
    assert select_indices(7, '[bad') == list(range(7))
    assert select_indices(7, '[:0]') == []


def test_select_prepared_filters_rows_and_scene_metadata(tmp_path, entry, video):
    second = copy.deepcopy(entry)
    second['idx'] = 'second selection sample'
    shutil.copyfile(video, video.parent / (second['idx'] + '.mp4'))
    prepared = tmp_path / 'prepared'
    prepare([entry, second], video.parent, prepared, subset='custom')
    manifest, grouped = verify_prepared(prepared)
    selected_manifest, selected_grouped, selection = select_prepared(
        manifest, grouped, '[:1]')
    assert [record['idx'] for record in selected_manifest['records']] == [entry['idx']]
    assert selection == dict(test_range='[:1]', selected=1, total=2, applied=True)
    assert selected_manifest['evaluation_scope'] == 'custom'
    assert selected_manifest['subset_size'] == 1
    assert manifest['subset_size'] == 2
    assert selected_manifest['selection']['source_input_signature'] == manifest['input_signature']
    assert [item['prompt_en'] for item in selected_grouped['scene']] == [
        entry['metadata']['prompt_en']]
    with pytest.raises(ValueError, match='selected no VABench samples'):
        select_prepared(manifest, grouped, '[:0]')


def test_task_progress_uses_sample_count(tmp_path, entry, video, monkeypatch):
    from ais_bench.benchmark.tasks.vabench import eval as task_module
    prepared = tmp_path / 'prepared'
    second = copy.deepcopy(entry)
    second['idx'] = 'second progress sample'
    shutil.copyfile(video, video.parent / (second['idx'] + '.mp4'))
    prepare([entry, second], video.parent, prepared, subset='custom')

    def fake_run(prepared, source, cache, destination, progress=None, **options):
        progress(1, 2, ['second_desync'])
        progress(2, 2, ['third_alignment'])
        return dict(requested_dimensions=['second_desync'], details={}, metrics={})

    monkeypatch.setattr(task_module, 'run', fake_run)
    monkeypatch.setattr(task_module, 'successful', lambda report: True)

    class Recorder:
        def __init__(self):
            self.states = []

        def update_task_state(self, state):
            self.states.append(dict(state))

    recorder = Recorder()
    cfg = dict(
        models=[dict(abbr='model', type='VABenchEvalPlaceholder', attr='local')],
        datasets=[[dict(abbr='vabench_t2av', path=str(prepared),
                        eval_cfg=dict(source_root='src', cache_dir='cache',
                                      dimensions=['second_desync', 'third_alignment']))]],
        work_dir=str(tmp_path / 'work'), cli_args={})
    task_module.VABenchEvalTask(cfg).run(recorder)
    assert [(s['finish_count'], s['total_count']) for s in recorder.states] == [
        (0, 2), (1, 2), (2, 2), (2, 2)]
    assert recorder.states[0]['progress_description'] == 'none'
    assert recorder.states[-1]['other_kwargs'] == {'dimension': 'done', 'sample_count': 2}


def test_explicit_cpu_onnx_provider_and_observation(monkeypatch):
    from types import ModuleType
    from ais_bench.benchmark.tasks.vabench.worker import instrument_models
    class Session:
        def __init__(self):
            self.providers = ['CUDAExecutionProvider']
        def set_providers(self, providers):
            self.providers = providers
        def get_providers(self):
            return self.providers
    class Scorer:
        def __init__(self):
            self.session = Session()
    module = ModuleType('vabench.first_dnsmos')
    module._load_dns_mos_model = lambda: Scorer()
    observations = []
    instrument_models(module, observations, 'cpu')
    scorer = module._load_dns_mos_model()
    assert scorer.session.providers == ['CPUExecutionProvider']
    assert observations[0]['observed_onnx_providers'] == ['CPUExecutionProvider']


def test_device_observer_tolerates_ctypes_library_loader():
    import ctypes
    from types import ModuleType, SimpleNamespace
    from ais_bench.benchmark.tasks.vabench.worker import _observe_devices
    module = ModuleType('vabench.probe')
    module.MODEL_CACHE = SimpleNamespace(loader=ctypes.cdll)
    assert _observe_devices(module) == {
        'observed_torch_devices': [], 'observed_onnx_providers': []}


def test_logging_handlers_are_rebound_after_redirect_close(tmp_path):
    import logging
    from contextlib import redirect_stderr
    from ais_bench.benchmark.tasks.vabench.worker import _rebind_stream_handlers
    logger = logging.getLogger('vabench-test-rebind')
    logger.handlers = []
    logger.propagate = False
    redirected = (tmp_path / 'backend.log').open('w')
    try:
        with redirect_stderr(redirected):
            handler = logging.StreamHandler()
            logger.addHandler(handler)
    finally:
        redirected.close()
    try:
        assert handler.stream.closed
        _rebind_stream_handlers(redirected, sys.stderr)
        assert handler.stream is sys.stderr
        logger.warning('this must not raise a logging error')
    finally:
        logger.removeHandler(handler)


def test_custom_split_validation(monkeypatch):
    import ais_bench.benchmark.tasks.vabench.engine as engine
    monkeypatch.setattr(engine, 'catalog', lambda root, subset: [{'idx': 'a'}, {'idx': 'b'}])
    engine._validate_prepared_split(
        dict(evaluation_scope='custom', records=[dict(idx='b')]), 'unused')
    with pytest.raises(ValueError, match='outside the pinned 778'):
        engine._validate_prepared_split(
            dict(evaluation_scope='custom', records=[dict(idx='z')]), 'unused')
    with pytest.raises(ValueError, match='exactly one'):
        engine._validate_prepared_split(
            dict(evaluation_scope='smoke', records=[dict(idx='a'), dict(idx='b')]), 'unused')
    with pytest.raises(ValueError, match='pinned 778 split'):
        engine._validate_prepared_split(
            dict(evaluation_scope='standard', records=[dict(idx='a')]), 'unused')


def test_unknown_scopes_cannot_pass_as_standard(monkeypatch, tmp_path):
    import ais_bench.benchmark.tasks.vabench.engine as engine
    monkeypatch.setattr(engine, 'catalog', lambda *args: [{'idx': 'a'}])
    engine._validate_prepared_split(
        dict(evaluation_scope='standard', records=[dict(idx='a')]), 'unused')
    with pytest.raises(ValueError, match='pinned 778 split'):
        engine._validate_prepared_split(
            dict(evaluation_scope='unsupported', records=[dict(idx='a')]), 'unused')
    with pytest.raises(ValueError, match='subset must be standard or smoke'):
        catalog('unused', 'unsupported')
    with pytest.raises(ValueError, match='Preparation subset'):
        prepare([], tmp_path, tmp_path / 'out', subset='unsupported')


def test_standard_catalog_export_has_only_current_fields(tmp_path):
    if not SOURCE.exists():
        pytest.skip('Local official checkout is not available')
    from ais_bench.tools.vabench.cli import main
    output = tmp_path / 'prompts.json'
    assert main(['catalog', '--source-root', str(SOURCE), '--output', str(output)]) == 0
    exported = read_json(output)
    assert set(exported) == {'subset', 'count', 'upstream_commit', 'entries'}
    assert exported['subset'] == 'standard' and exported['count'] == 778
    assert set(read_json(PROTOCOL_PATH)) == {
        'upstream', 'commit', 'description', 'data_sha256', 'standard_ids'}


def test_selected_ten_percent_catalog_stays_fixed():
    if not SOURCE.exists():
        pytest.skip('Local official checkout is not available')
    selected = read_json(Path('ais_bench/configs/vabench_examples/mini10pct.json'))
    standard = {entry['idx']: entry for entry in catalog(SOURCE)}
    assert selected['count'] == len(selected['entries']) == 78
    assert len({entry['idx'] for entry in selected['entries']}) == 78
    assert len({entry['category'] for entry in selected['entries']}) == 24
    assert all(standard[entry['idx']]['category'] == entry['category'] for entry in selected['entries'])


def test_custom_scope_is_labeled_non_official():
    from ais_bench.benchmark.summarizers.vabench.results import build_report
    scores = dict(first_dnsmos=3, first_nisqa=3, first_audiobox=2.75,
                  second_viclip=0, second_clap=0, second_desync=1, second_lsa=5,
                  second_imagebind=0, third_alignment=3, third_audio_reality=3,
                  third_visual_reality=3, third_expressiveness=3, third_artistry=3,
                  fourth_qa_audio=0.5, fourth_qa_vision=0.5)
    details = {dim: dict(records=[dict(idx='x', category='a', status='ok', score=score,
                                       original_score=score, raw={})], errors=[])
               for dim, score in scores.items()}
    manifest = dict(input_signature='sig', evaluation_scope='custom',
                    records=[dict(idx='x', category='a')])
    report = build_report(manifest, details, {})
    assert report['evaluation_scope'] == 'custom'
    assert 'Custom subset of 1' in report['scope_note']
    assert 'not full 778-set' in report['scope_note']
    assert report['total']['status'] == 'formal'
    assert report['total']['label'] == '自定义子集总分（非官方）'


def test_lsa_missing_sets_must_match_and_not_be_empty():
    left = _report(2.0, dim='second_lsa')
    left['details']['second_lsa']['records'].append(dict(idx='y', category='scene',
                                                       status='missing', score=None))
    left['sample_ids'] = ['x', 'y']
    assert compare(left, copy.deepcopy(left), ['second_lsa'])['passed']
    right = copy.deepcopy(left)
    right['details']['second_lsa']['records'][1]['status'] = 'ok'
    right['details']['second_lsa']['records'][1]['score'] = 2.0
    assert not compare(left, right, ['second_lsa'])['passed']


def test_real_worker_subprocess_and_mllm_grouping(tmp_path, entry, video, monkeypatch):
    """A fake scorer tests process/protocol plumbing, not model accuracy."""
    import ais_bench.benchmark.tasks.vabench.engine as engine
    prepared = tmp_path / 'prepared'
    prepare([entry], video.parent, prepared, subset='smoke')
    source = tmp_path / 'fake_source'
    package = source / 'vabench'
    package.mkdir(parents=True)
    (source / 'torch.py').write_text('def device(x): return x\n', encoding='utf-8')
    (package / '__init__.py').write_text('', encoding='utf-8')
    (package / 'omni_runtime.py').write_text(
        'def release_omni(): pass\n'
        'def get_omni_model_and_processor(*a, **kw): raise RuntimeError("unused")\n', encoding='utf-8')
    dims = ['third_alignment', 'fourth_qa_audio']
    for dim in dims:
        fixture_score = 1.0 if dim.startswith('fourth_') else 3.0
        (package / f'{dim}.py').write_text(
            'import json\n'
            f'def compute_{dim}(file, device, submodules, **kw):\n'
            '    items = json.load(open(file, encoding="utf-8"))\n'
            f'    return None, [{{"video_path": x["video_list"][0], "video_results": {fixture_score}}} for x in items]\n',
            encoding='utf-8')
    cache = tmp_path / 'cache'
    checkpoint = cache / 'Qwen2.5_omni_7B'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'dummy').write_text('test fixture, not a model')
    monkeypatch.setattr(engine, 'catalog', lambda *args: [entry])
    progress = []
    report = run(prepared, source, cache, tmp_path / 'run', dimensions=dims,
                 device='cpu', python=sys.executable,
                 progress=lambda done, total, names: progress.append((done, total, tuple(names))))
    assert successful(report)
    assert progress == [(1, 1, tuple(dims))]
    assert report['evaluation_scope'] == 'smoke'
    assert report['metrics']['third_alignment']['mean'] == 3
    assert len(list((tmp_path / 'run').glob('job_*'))) == 1
    assert len(list((tmp_path / 'run' / 'official_eval_results').glob('*/*_eval_results.json'))) == 1
    assert len(list((tmp_path / 'run' / 'official_eval_results').glob('*/*_eval_results.csv'))) == 1
    request = read_json(tmp_path / 'run/job_00/request.json')
    assert request['dimensions'] == dims
    assert compare(report, report, dims)['passed']

    # Exercise the real upstream VABench orchestration class, with the same
    # fake compute_* scorers. Only model inference is substituted.
    if SOURCE.exists():
        (package / '__init__.py').write_text(
            (SOURCE / 'vabench/__init__.py').read_text(encoding='utf-8'), encoding='utf-8')
        (source / 'torch.py').write_text(
            'def device(x): return x\n'
            'class cuda:\n'
            '    @staticmethod\n'
            '    def is_available(): return False\n', encoding='utf-8')
        (package / 'utils.py').write_text(
            'import json\n'
            'def init_submodules(*a, **kw): return {}\n'
            'def load_json(p):\n'
            '    with open(p, encoding="utf-8") as f: return json.load(f)\n'
            'def save_json(obj, p):\n'
            '    with open(p, "w", encoding="utf-8") as f: json.dump(obj, f)\n', encoding='utf-8')
        mapping_dir = source / 'mapping'
        mapping_dir.mkdir()
        (mapping_dir / 'final_idx_to_prompt.csv').write_text(
            'idx\tprompt\n' + entry['idx'] + '\t' + entry['metadata']['prompt_en'] + '\n',
            encoding='utf-8')
        # The source signature changed to install the real orchestrator;
        # recompute both modes against this identical source snapshot.
        adapter = run(prepared, source, cache, tmp_path / 'adapter_real_class',
                      dimensions=dims, device='cpu')
        official = run(prepared, source, cache, tmp_path / 'official_real_class',
                       dimensions=dims, device='cpu', mode='official')
        assert successful(official)
        assert compare(official, adapter, dims)['passed']
