"""Dependency installation contracts; no pip installation or model downloads."""
import base64
import csv
import hashlib
import io
from pathlib import Path
import zipfile

import pytest
from packaging.requirements import Requirement

from ais_bench.tools.vabench import environment
from ais_bench.benchmark.utils.vabench.data import digest, read_json, write_json
from ais_bench.benchmark.tasks.vabench.resources import preflight


def original_wheel(tmp_path, missing=None):
    file = tmp_path / 'nisqa-2.0.post2-py3-none-any.whl'
    pins = dict(environment.NISQA_PINS, pillow='10.2.0', numpy='1.26.4')
    text = 'Metadata-Version: 2.1\nName: nisqa\nVersion: 2.0.post2\n'
    text += ''.join(f'Requires-Dist: {n} =={v}\n' for n, v in pins.items() if n != missing)
    with zipfile.ZipFile(file, 'w') as archive:
        archive.writestr('nisqa-2.0.post2.dist-info/METADATA', text)
        archive.writestr('nisqa-2.0.post2.dist-info/RECORD', '')
        archive.writestr('nisqa/NISQA_model.py', '# unchanged model code\n')
        archive.writestr('nisqa-2.0.post2.dist-info/LICENSE', 'test fixture license')
        archive.writestr('nisqa-2.0.post2.dist-info/WHEEL', 'Wheel-Version: 1.0\n')
    return file


def test_nisqa_patch_keeps_code_and_rebuilds_record(tmp_path):
    original = original_wheel(tmp_path)
    before = digest(original)
    patched, audit = environment.patch_nisqa(original, tmp_path / 'patched', expected_sha=before)
    assert digest(original) == before
    assert audit['original_sha256'] != audit['patched_sha256']
    assert read_json(patched.parent / 'nisqa_patch.json') == audit
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(patched) as result:
        for name in source.namelist():
            if name not in audit['changed_members']:
                assert source.read(name) == result.read(name)
        text = result.read('nisqa-2.0.post2.dist-info/METADATA').decode()
        reqs = [Requirement(s.split(': ', 1)[1]) for s in text.splitlines() if s.startswith('Requires-Dist: ')]
        values = {r.name: str(r.specifier) for r in reqs}
        assert values['pillow'] == '==10.2.0' and values['numpy'] == '==1.26.4'
        assert all(values[k] == '>=' + v for k, v in environment.NISQA_PINS.items())
        record = result.read('nisqa-2.0.post2.dist-info/RECORD').decode()
        for name, hash_value, size in csv.reader(io.StringIO(record)):
            if name.endswith('/RECORD'):
                assert hash_value == size == ''
            else:
                content = result.read(name)
                expected = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b'=').decode()
                assert hash_value == 'sha256=' + expected and int(size) == len(content)


def test_nisqa_patch_refuses_wrong_artifact_and_unexpected_metadata(tmp_path):
    original = original_wheel(tmp_path, missing='torch')
    with pytest.raises(ValueError, match='SHA256'):
        environment.patch_nisqa(original, tmp_path / 'patched')
    with pytest.raises(ValueError, match='exactly one NISQA pin for torch'):
        environment.patch_nisqa(original, tmp_path / 'patched', expected_sha=digest(original))


def test_install_plan_keeps_special_packages_out_of_dependency_resolution(tmp_path):
    plan = environment.install_plan(environment.REQUIREMENTS, tmp_path)
    assert 'vabench-build.txt' in plan[0][plan[0].index('-r') + 1]
    assert '--no-build-isolation' in plan[1]
    assert '--no-deps' in plan[2] and '--force-reinstall' in plan[2]
    assert 'vabench-git.txt' in plan[2][plan[2].index('-r') + 1]
    assert 'download' in plan[3] and '--no-deps' in plan[3]
    mirror = 'https://example.com/simple'
    assert all(c[-2:] == ['--index-url', mirror] for c in environment.install_plan(
        environment.REQUIREMENTS, tmp_path, mirror))


def test_plan_is_read_only(tmp_path, capsys, monkeypatch):
    def no_commands(*args, **kwargs):
        raise AssertionError('A plan must not install or download anything')
    monkeypatch.setattr(environment.subprocess, 'run', no_commands)
    monkeypatch.setattr(environment.urllib.request, 'urlopen', no_commands)
    work = tmp_path / 'not-created'
    assert environment.main(['plan', '--source-root', 'source', '--cache-dir', 'cache',
                             '--work-dir', str(work)]) == 0
    assert not work.exists()
    assert 'nisqa==2.0.post2' in capsys.readouterr().out


def test_system_packages_are_explicit_and_precede_python_dependencies(tmp_path):
    default = environment.install_plan(environment.REQUIREMENTS, tmp_path)
    assert not any(c[0] == 'apt-get' for c in default)
    commands = environment.install_plan(environment.REQUIREMENTS, tmp_path,
                                        'https://example.com/simple', system_deps=True)
    assert commands[0] == ['apt-get', 'update']
    assert commands[1][:4] == ['apt-get', 'install', '-y', '--no-install-recommends']
    assert 'ffmpeg' in commands[1] and 'build-essential' in commands[1]
    assert all('--index-url' in c for c in commands[2:])
    assert '--index-url' not in commands[0] + commands[1]


def test_system_install_guard_rejects_nonroot_and_unsupported_distros(monkeypatch):
    monkeypatch.setattr(environment.os, 'geteuid', lambda: 1000, raising=False)
    with pytest.raises(RuntimeError, match='requires root'):
        environment.system_install_guard()
    monkeypatch.setattr(environment.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(environment.platform, 'freedesktop_os_release', lambda: {'ID': 'alpine'})
    with pytest.raises(RuntimeError, match='Ubuntu/Debian'):
        environment.system_install_guard()
    monkeypatch.setattr(environment.platform, 'freedesktop_os_release', lambda: {'ID': 'ubuntu'})
    monkeypatch.setattr(environment.shutil, 'which', lambda name: '/usr/bin/' + name)
    environment.system_install_guard()


def test_plan_with_system_option_still_does_not_install(tmp_path, capsys, monkeypatch):
    def no_commands(*args, **kwargs):
        raise AssertionError('A plan must not run apt or pip')
    monkeypatch.setattr(environment.subprocess, 'run', no_commands)
    work = tmp_path / 'not-created'
    assert environment.main(['plan', '--install-system-deps', '--source-root', 'source',
                             '--cache-dir', 'cache', '--work-dir', str(work)]) == 0
    assert not work.exists() and 'apt-get' in capsys.readouterr().out


def test_runtime_guard_protects_unsupported_targets(monkeypatch):
    monkeypatch.setattr(environment.platform, 'system', lambda: 'Windows')
    with pytest.raises(RuntimeError, match='Linux x86_64 Python 3.11'):
        environment.runtime_guard()


def test_runtime_guard_protects_aisbench_main_environment(monkeypatch):
    monkeypatch.setattr(environment.sys, 'version_info', (3, 11, 14))
    monkeypatch.setattr(environment.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(environment.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(environment.shutil, 'which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr(environment.metadata, 'version', lambda name: 'installed')
    with pytest.raises(RuntimeError, match='AISBench is installed'):
        environment.runtime_guard()


def test_check_writes_failure_report_on_unexpected_probe_error(tmp_path, monkeypatch):
    def broken_probe(*args, **kwargs):
        raise RuntimeError('broken environment')
    monkeypatch.setattr(environment, 'check_environment', broken_probe)
    assert environment.main(['check', '--source-root', 'source', '--cache-dir', 'cache',
                             '--work-dir', str(tmp_path)]) == 1
    result = read_json(tmp_path / 'environment_check.json')
    assert result['passed'] is False and result['model_inference_verified'] is False
    assert result['error'] == 'RuntimeError: broken environment'


def test_declarations_have_only_one_opencv_and_fixed_git_revisions():
    pins = environment.expected_packages(environment.REQUIREMENTS)
    assert pins['torch'] == '2.6.0' and pins['numpy'] == '1.26.4'
    assert pins['opencv-python-headless'] == '4.9.0.80'
    assert 'opencv-python' not in pins and 'nisqa' not in pins and 'pytorchvideo' not in pins
    for line in environment.requirement_lines(environment.REQUIREMENTS / 'vabench-git.txt'):
        req = Requirement(line)
        assert len(req.url.rsplit('@', 1)[1]) == 40


def test_strict_preflight_rejects_partial_model_and_face_directories(tmp_path):
    root, cache = tmp_path / 'source', tmp_path / 'cache'
    for dim in ('third_alignment', 'second_lsa'):
        file = root / 'vabench' / (dim + '.py')
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('# source')
    qwen = cache / 'Qwen2.5_omni_7B'
    qwen.mkdir(parents=True)
    write_json(qwen / 'config.json', {})
    write_json(qwen / 'tokenizer_config.json', {})
    (qwen / 'model-00001.safetensors').write_bytes(b'fixture, not a model')
    write_json(qwen / 'model.safetensors.index.json', {'weight_map': {
        'a': 'model-00001.safetensors', 'b': 'model-00002.safetensors'}})
    missing = preflight(root, cache, ['third_alignment', 'second_lsa'], strict=True)
    assert str(qwen / 'model-00002.safetensors') in missing
    assert any('det_10g.onnx' in name for name in missing)
    assert not any('tokenizer_config.json' in name for name in missing)


def test_external_layout_does_not_overwrite_existing_code(tmp_path, monkeypatch):
    root, cache = tmp_path / 'source', tmp_path / 'cache'
    dns = root / 'third_party/DNS-Challenge/DNSMOS/dnsmos_local.py'
    dns.parent.mkdir(parents=True)
    dns.write_text('# existing source, preserve it', encoding='utf-8')
    monkeypatch.setattr(environment, 'catalog', lambda *args: [])
    def no_download(*args, **kwargs):
        raise AssertionError('Must retain existing source')
    monkeypatch.setattr(environment.urllib.request, 'urlopen', no_download)
    result = environment.prepare_external(root, cache)
    assert dns.read_text() == '# existing source, preserve it'
    assert result['dnsmos_sha256'] == digest(dns)
