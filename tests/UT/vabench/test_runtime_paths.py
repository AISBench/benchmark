"""Runtime boundary contracts: no pip/apt, weight downloads or real GPU calls."""
import io
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from ais_bench.tools.vabench import environment as envmod
from ais_bench.benchmark.utils.vabench.data import read_json, write_json


@pytest.mark.parametrize('scenario', ['clean', 'decord_tag', 'decode_failed', 'missing_version',
                                     'missing_git', 'bad_nisqa', 'missing_resources', 'ffmpeg_failed'])
def test_environment_audit_checks_versions_origins_probes_and_exact_exceptions(tmp_path, monkeypatch, scenario):
    pins = {'torch': '2.6.0', 'torchvision': '0.21.0', 'torchaudio': '2.6.0', 'requests': '2.32.5'}
    monkeypatch.setattr(envmod, 'expected_packages', lambda *a: pins)
    monkeypatch.setattr(envmod, 'requirement_lines', lambda *a: [
        'imagebind @ git+https://example.com/repo.git@' + 'a' * 40])
    def version(name):
        if scenario == 'missing_version' and name == 'requests':
            raise envmod.metadata.PackageNotFoundError(name)
        return pins[name] + ('+cu124' if name == 'torch' else '')
    def distribution(name):
        if name == 'imagebind':
            if scenario == 'missing_git':
                raise envmod.metadata.PackageNotFoundError(name)
            return SimpleNamespace(read_text=lambda _: json.dumps({'vcs_info': {'commit_id': 'a' * 40}}))
        op = '==' if scenario == 'bad_nisqa' else '>='
        return SimpleNamespace(version='2.0.post2', requires=[n + op + v for n, v in envmod.NISQA_PINS.items()])
    monkeypatch.setattr(envmod.metadata, 'version', version)
    monkeypatch.setattr(envmod.metadata, 'distribution', distribution)
    probes = []
    def probe(code, env):
        assert env['HF_HUB_OFFLINE'] == env['TRANSFORMERS_OFFLINE'] == '1'
        assert env['VABENCH_CACHE_DIR'] == str((tmp_path / 'cache').resolve())
        probes.append(code)
        return dict(passed=not (scenario == 'decode_failed' and code.startswith('import decord')), log='fixture')
    monkeypatch.setattr(envmod, '_probe', probe)
    monkeypatch.setattr(envmod.shutil, 'which', lambda n: '/bin/' + n)
    monkeypatch.setattr(envmod.sys, 'version_info', (3, 11, 0))
    monkeypatch.setattr(envmod, 'preflight', lambda *a, **kw: ['missing'] if scenario == 'missing_resources' else [])
    def run(command, **kwargs):
        if command[0] == 'ffmpeg':
            if scenario == 'ffmpeg_failed':
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)
        warning = scenario in ('decord_tag', 'decode_failed')
        return subprocess.CompletedProcess(command, int(warning),
            'decord 0.6.0 is not supported on this platform\n' if warning else 'No broken requirements found.\n', '')
    monkeypatch.setattr(envmod.subprocess, 'run', run)
    result = envmod.check_environment(tmp_path / 'source', tmp_path / 'cache')
    assert result['passed'] == (scenario in ('clean', 'decord_tag'))
    assert result['model_inference_verified'] is False
    assert len(probes) >= 20
    assert result['pip_check']['accepted_decord_tag_warning'] == (scenario == 'decord_tag')


@pytest.mark.parametrize('failure', [None, 'stage', 'check'])
def test_installer_writes_audit_and_preserves_failed_status(tmp_path, monkeypatch, failure):
    cache = tmp_path / 'cache'; cache.mkdir()
    work = tmp_path / 'audit'
    monkeypatch.setattr(envmod, 'runtime_guard', lambda **kw: None)
    monkeypatch.setattr(envmod, 'system_install_guard', lambda: None)
    monkeypatch.setattr(envmod, 'catalog', lambda *a: [])
    monkeypatch.setattr(envmod, 'install_plan', lambda *a: [['apt-get', 'update'], ['fixture-pip']])
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if command[0] == 'apt-get':
            assert kwargs['env']['DEBIAN_FRONTEND'] == 'noninteractive'
        if failure == 'stage':
            raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(envmod.subprocess, 'run', run)
    monkeypatch.setattr(envmod.subprocess, 'check_output', lambda *a, **kw: 'fixture==1\n')
    monkeypatch.setattr(envmod, 'patch_nisqa', lambda *a: (tmp_path / 'patched.whl', {'package_code_unchanged': True}))
    monkeypatch.setattr(envmod, 'prepare_external', lambda *a: {'dnsmos_sha256': 'fixture'})
    monkeypatch.setattr(envmod, 'check_environment', lambda *a: {'passed': failure != 'check'})
    args = ['install', '--source-root', 'source', '--cache-dir', str(cache),
            '--work-dir', str(work), '--install-system-deps']
    if failure == 'stage':
        with pytest.raises(subprocess.CalledProcessError):
            envmod.main(args)
        assert read_json(work / 'install.json')['status'] == 'failed'
    else:
        assert envmod.main(args) == int(failure == 'check')
        audit = read_json(work / 'install.json')
        assert audit['status'] == ('checks_failed' if failure else 'checks_passed')
        assert audit['nisqa']['package_code_unchanged']
        assert '--no-deps' in commands[-1]
        assert (work / 'installed.freeze.txt').read_text() == 'fixture==1\n'


def test_source_preparation_downloads_only_missing_source_and_links_cache(tmp_path, monkeypatch):
    source, cache = tmp_path / 'source', tmp_path / 'cache'
    (source / 'third_party').mkdir(parents=True)
    (cache / 'synchformer/Synchformer').mkdir(parents=True)
    (cache / 'checkpoints/auxiliary/buffalo_l').mkdir(parents=True)
    monkeypatch.setattr(envmod, 'catalog', lambda *a: [])
    monkeypatch.setattr(envmod.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(b'class ComputeScore: pass\n'))
    links = []
    def link(path, target, **kwargs):
        links.append((path, target)); path.mkdir()
    monkeypatch.setattr(Path, 'symlink_to', link)
    result = envmod.prepare_external(source, cache)
    assert len(links) == 2 and result['dnsmos_sha256']
    dns = source / 'third_party/DNS-Challenge/DNSMOS/dnsmos_local.py'
    assert dns.read_bytes() == b'class ComputeScore: pass\n'
    dns.unlink()
    monkeypatch.setattr(envmod.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(b'error response'))
    with pytest.raises(ValueError, match='Unexpected DNSMOS'):
        envmod.prepare_external(source, cache)
    assert not dns.exists()


def test_small_probe_returns_process_errors_and_timeouts(monkeypatch):
    monkeypatch.setattr(envmod.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess([], 2, 'out', 'err'))
    assert envmod._probe('unused', {}) == dict(passed=False, returncode=2, log='outerr')
    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired('fixture', 3)
    monkeypatch.setattr(envmod.subprocess, 'run', timeout)
    assert envmod._probe('unused', {}, 3)['passed'] is False


def test_dataset_loader_reads_prepared_rows(tmp_path):
    from ais_bench.benchmark.datasets.vabench import VABenchDataset
    write_json(tmp_path / 'manifest.json', {'records': [{'idx': 'a'}, {'idx': 'b'}]})
    dataset = VABenchDataset(str(tmp_path)).dataset
    assert len(dataset) == 2 and dataset[1]['idx'] == 'b'


@pytest.mark.parametrize('fails', [False, True])
def test_task_entrypoint_reports_final_state_and_joins_monitor(tmp_path, monkeypatch, fails):
    import ais_bench.benchmark.tasks.vabench.eval as task
    states, joined = [], []
    cfg = {'work_dir': str(tmp_path), 'cli_args': {}}
    monkeypatch.setattr(task.Config, 'fromfile', lambda *a: cfg)
    monkeypatch.setattr(task, 'task_abbr_from_cfg', lambda *a: 'fixture')
    monkeypatch.setattr(task, 'TaskStateManager', lambda **kw: SimpleNamespace(
        update_task_state=lambda state: states.append(state), launch=lambda: None))
    monkeypatch.setattr(task.threading, 'Thread', lambda **kw: SimpleNamespace(
        start=lambda: None, join=lambda **kw: joined.append(True)))
    def evaluate(manager):
        if fails:
            raise RuntimeError('fixture backend failure')
    monkeypatch.setattr(task, 'VABenchEvalTask', lambda cfg: SimpleNamespace(run=evaluate))
    monkeypatch.setattr(sys, 'argv', ['vabench_eval', 'fixture.py'])
    if fails:
        with pytest.raises(RuntimeError):
            task.main()
    else:
        task.main()
    assert states[-1]['status'] == ('error' if fails else 'finish') and joined


def test_cli_catalog_custom_and_comparison_paths(tmp_path, monkeypatch):
    from ais_bench.tools.vabench import cli
    entry = {'idx': 'a', 'category': 'one'}
    monkeypatch.setattr(cli, 'catalog', lambda *a, **kw: [entry])
    output = tmp_path / 'catalog.json'
    assert cli.main(['catalog', '--source-root', 'source', '--output', str(output)]) == 0
    assert read_json(output)['count'] == 1
    monkeypatch.setattr(cli, 'catalog_videos', lambda *a: [entry])
    monkeypatch.setattr(cli, 'prepare', lambda *a, **kw: {'subset_size': 1})
    assert cli.main(['prepare', '--source-root', 'source', '--subset', 'custom',
                     '--videos', 'videos', '--output', str(tmp_path / 'prepared')]) == 0
    with pytest.raises(SystemExit):
        cli.main(['prepare', '--source-root', 'source', '--subset', 'custom',
                  '--sample-id', 'a', '--videos', 'videos', '--output', 'out'])
    from ais_bench.benchmark.summarizers.vabench.results import build_report
    provenance = {k: 'same' for k in ['protocol_sha256', 'source_signature',
                                    'integration_signature', 'weights_signature', 'scoring_signature']}
    report = build_report({'input_signature': 'same', 'records': [entry]}, {
        'second_clap': {'records': [dict(entry, score=0.0, status='ok', raw={})]}}, provenance)
    path = tmp_path / 'report.json'; write_json(path, report)
    assert cli.main(['compare', '--reference', str(path), '--candidate', str(path),
                     '--dimensions', 'second_clap', '--output', str(tmp_path / 'comparison.json')]) == 0
    assert read_json(tmp_path / 'comparison.json')['passed']
    monkeypatch.setattr(cli, 'run', lambda *a, **kw: report)
    assert cli.main(['run', '--source-root', 'source', '--cache-dir', 'cache',
                     '--prepared', 'prepared', '--dimensions', 'second_clap', '--output', 'output']) == 0
    bad_options = tmp_path / 'bad.json'; write_json(bad_options, {'unknown': 1})
    with pytest.raises(SystemExit):
        cli.main(['run', '--source-root', 'source', '--cache-dir', 'cache', '--prepared', 'prepared',
                  '--output', 'output', '--options', str(bad_options)])


@pytest.mark.parametrize('missing', [False, True])
def test_cli_doctor_uses_declared_modules_and_missing_resources(tmp_path, monkeypatch, missing):
    from ais_bench.tools.vabench import cli
    torch = ModuleType('torch'); torch.__version__ = 'fixture'
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.device = lambda device: device
    torch.ones = lambda *a, device: SimpleNamespace(device=device, cpu=lambda: None)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    real_spec = cli.importlib.util.find_spec
    monkeypatch.setattr(cli.importlib.util, 'find_spec', lambda n: None if n == 'torch_npu' else
                        SimpleNamespace() if n in ('torch', 'transformers', 'qwen_omni_utils', 'nisqa',
                        'audiobox_aesthetics', 'onnxruntime', 'decord', 'imagebind') else real_spec(n))
    monkeypatch.setattr(cli.shutil, 'which', lambda n: None if n == 'npu-smi' else '/bin/' + n)
    monkeypatch.setattr(cli, 'preflight', lambda *a, **kw: ['weight'] if missing else [])
    assert cli.main(['doctor', '--source-root', 'source', '--cache-dir', 'cache',
                     '--dimensions', 'first_dnsmos', '--device', 'cpu']) == int(missing)


@pytest.mark.parametrize('available,allocation_error', [(False, False), (True, True), (True, False)])
def test_doctor_gpu_target_must_really_be_usable(monkeypatch, capsys, available, allocation_error):
    from ais_bench.tools.vabench import cli
    torch = ModuleType('torch')
    torch.__version__ = 'fixture'
    torch.device = lambda device: device
    torch.cuda = SimpleNamespace(is_available=lambda: available, set_device=lambda device: None)
    def ones(*args, device):
        if allocation_error:
            raise RuntimeError('CUDA allocation failed')
        return SimpleNamespace(device=device, cpu=lambda: None)
    torch.ones = ones
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(cli.importlib.util, 'find_spec', lambda name: None if name == 'torch_npu' else object())
    monkeypatch.setattr(cli.shutil, 'which', lambda name: None if name == 'npu-smi' else '/bin/' + name)
    monkeypatch.setattr(cli, 'preflight', lambda *a, **kw: [])
    code = cli.main(['doctor', '--source-root', 'source', '--cache-dir', 'cache',
                     '--dimensions', 'first_dnsmos'])
    assert code == int(not available or allocation_error)
    info = json.loads(capsys.readouterr().out)
    assert ('device_error' in info) == bool(code)
    if code == 0:
        assert info['device_probe'] == dict(passed=True, observed_device='cuda:0')


@pytest.mark.parametrize('device,available', [('cpu', True), ('cuda:1', True), ('cuda:0', False),
                                            ('npu:0', True), ('npu:0', False), ('bad', True)])
def test_device_initialization_respects_explicit_target(monkeypatch, device, available):
    from ais_bench.benchmark.tasks.vabench.worker import initialize_device
    selected = []
    torch = ModuleType('torch'); torch.device = lambda d: d
    accelerator = SimpleNamespace(is_available=lambda: available, set_device=selected.append)
    torch.cuda = torch.npu = accelerator
    monkeypatch.setitem(sys.modules, 'torch', torch)
    npu = ModuleType('torch_npu'); npu.__path__ = []
    contrib = ModuleType('torch_npu.contrib'); contrib.transfer_to_npu = object()
    monkeypatch.setitem(sys.modules, 'torch_npu', npu)
    monkeypatch.setitem(sys.modules, 'torch_npu.contrib', contrib)
    if device == 'bad':
        with pytest.raises(ValueError): initialize_device(device)
    elif device != 'cpu' and not available:
        with pytest.raises(RuntimeError): initialize_device(device)
    else:
        assert initialize_device(device) == device
        assert selected == ([] if device == 'cpu' else [device])


def test_audiobox_observation_places_predictor_and_is_idempotent(monkeypatch):
    from ais_bench.benchmark.tasks.vabench.worker import instrument_models, _observe_devices
    placed = []
    torch = ModuleType('torch'); torch.device = lambda d: d
    monkeypatch.setitem(sys.modules, 'torch', torch)
    model = SimpleNamespace(to=placed.append, parameters=lambda: [SimpleNamespace(device='cpu')])
    predictor = SimpleNamespace(model=model)
    module = ModuleType('vabench.first_audiobox')
    module.initialize_predictor = lambda: predictor
    module._get_test_model = lambda: model
    observations = []
    instrument_models(module, observations, 'cpu')
    factory = module.initialize_predictor
    instrument_models(module, observations, 'cpu')
    assert module.initialize_predictor is factory
    assert module.initialize_predictor().device == 'cpu' and placed == ['cpu']
    module._get_test_model()
    assert observations[-1]['observed_torch_devices'] == ['cpu']
    def broken(): raise RuntimeError('observational failure')
    module.MODEL_CACHE = SimpleNamespace(parameters=broken, get_providers=broken, objects=[1, None])
    assert _observe_devices(module)['observed_torch_devices'] == []


def test_lsa_cpu_subcomponent_preserves_detector_configuration(monkeypatch):
    from ais_bench.benchmark.tasks.vabench import worker
    calls = []
    class Detector: pass
    app = SimpleNamespace(models={'det': SimpleNamespace(session=SimpleNamespace(
        get_providers=lambda: ['CPUExecutionProvider']))}, prepare=lambda **kw: calls.append(kw))
    face = SimpleNamespace(FaceDetector=Detector, FaceAnalysis=lambda **kw: calls.append(kw) or app,
                           _INSIGHTFACE_ROOT='fixture', INSIGHTFACE_DETECT_SIZE=512)
    original_import = worker.importlib.import_module
    monkeypatch.setattr(worker.importlib, 'import_module', lambda name: face if name.endswith('face_detector')
                        else original_import(name))
    observations = []
    worker.instrument_models(ModuleType('vabench.second_lsa'), observations, 'cpu')
    detector = Detector()
    assert detector.app is app
    assert calls[0]['allowed_modules'] == ['detection', 'landmark_2d_106']
    assert calls[0]['providers'] == ['CPUExecutionProvider']
    assert calls[1] == {'ctx_id': -1, 'det_size': (512, 512)}
    assert observations[-1]['observed_onnx_providers'] == ['CPUExecutionProvider']


def test_resource_preflight_and_fingerprints_reject_incomplete_cache(tmp_path):
    from ais_bench.benchmark.tasks.vabench import resources
    source, cache = tmp_path / 'source', tmp_path / 'cache'
    source.mkdir(); cache.mkdir()
    with pytest.raises(FileNotFoundError): resources.source_signature(source)
    dims = ['first_dnsmos', 'second_desync', 'second_clap', 'third_alignment']
    missing = resources.preflight(source, cache, dims, strict=True)
    assert any('dnsmos_local.py' in p for p in missing)
    assert any('train_utils.py' in p for p in missing)
    for folder in ['clap/laion__clap-htsat-unfused', 'Qwen2.5_omni_7B']:
        path = cache / folder; path.mkdir(parents=True)
        write_json(path / 'config.json', {})
        write_json(path / 'tokenizer_config.json', {})
        (path / 'model-00001-of-00002.safetensors').touch()
        write_json(path / 'model.safetensors.index.json', {'weight_map': {}})
    missing = resources.preflight(source, cache, dims, strict=True)
    assert any('(empty file)' in p for p in missing)
    assert any('(invalid weight index)' in p for p in missing)
    weights, missing = resources.weight_fingerprints(['second_clap'], cache)
    assert weights and not missing
    (cache / 'clap/laion__clap-htsat-unfused/model.safetensors.index.json').unlink()
    missing = resources.preflight(source, cache, ['second_clap'], strict=True)
    assert any('<sharded model weight index>' in p for p in missing)
    weights, missing = resources.weight_fingerprints(['first_nisqa'], cache)
    assert missing and not weights


@pytest.mark.parametrize('scenario', ['missing', 'timeout', 'exit'])
def test_engine_keeps_failure_reports_and_ignores_progress_callback_errors(tmp_path, monkeypatch, scenario):
    from ais_bench.benchmark.tasks.vabench import engine
    manifest = dict(input_signature='fixture', evaluation_scope='smoke',
                    records=[dict(idx='a', category='scene')])
    monkeypatch.setattr(engine, 'verify_prepared', lambda *a: (manifest, {}))
    monkeypatch.setattr(engine, '_validate_prepared_split', lambda *a: None)
    monkeypatch.setattr(engine, 'source_signature', lambda *a: 'source')
    monkeypatch.setattr(engine, 'weight_fingerprints', lambda *a: ({}, []))
    monkeypatch.setattr(engine, 'preflight', lambda *a: ['missing_weight'] if scenario == 'missing' else [])
    def run(command, **kwargs):
        if scenario == 'timeout': raise subprocess.TimeoutExpired(command, 1)
        request = read_json(command[-1]); write_json(request['response'], {})
        return subprocess.CompletedProcess(command, 2)
    monkeypatch.setattr(engine.subprocess, 'run', run)
    def callback(*args): raise RuntimeError('UI must not affect scoring')
    output = tmp_path / 'output'
    report = engine.run('prepared', 'source', 'cache', output, ['second_clap'], device='cpu', progress=callback)
    assert report['metrics']['second_clap']['failed'] == 1
    assert report['total']['score'] is None and not engine.successful(report)
    assert (output / 'partial_report.json').exists() and (output / 'report.json').exists()
