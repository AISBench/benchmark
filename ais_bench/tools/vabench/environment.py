"""Prepare and audit the external GPU scoring environment (stdlib at entry).

Run from an AISBench source checkout using the scoring Conda's python3.
No install/download occurs when importing this module, planning, or checking.
"""
import argparse
import base64
import csv
import hashlib
import importlib.metadata as metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

from ais_bench.benchmark.utils.vabench import DIMENSIONS
from ais_bench.benchmark.utils.vabench.data import PACKAGE_ROOT, catalog, digest, write_json
from ais_bench.benchmark.tasks.vabench.resources import preflight


REQUIREMENTS = PACKAGE_ROOT.parent / 'requirements/datasets'
NISQA_SHA256 = '386205dc930aa98114a689f301112852dfe661d81eb4e6e630675ae9acd2c593'
NISQA_PINS = {'torch': '2.2.1', 'torchvision': '0.17.1', 'scipy': '1.12.0',
              'scikit-learn': '1.4.1.post1', 'matplotlib': '3.8.3'}
DNSMOS_URL = ('https://raw.githubusercontent.com/microsoft/DNS-Challenge/'
              '591184a9fcb2cbdec02520fed81a32bbbf9d73ff/DNSMOS/dnsmos_local.py')
SYSTEM_PACKAGES = ('git', 'ffmpeg', 'libsndfile1', 'build-essential', 'ca-certificates')


def requirement_lines(path):
    return [s.split('#', 1)[0].strip() for s in Path(path).read_text(
        encoding='utf-8').splitlines() if s.split('#', 1)[0].strip()]


def install_plan(requirements, work, index_url=None, system_deps=False):
    """Argument arrays; never interpolate an executable shell command."""
    requirements, work = Path(requirements), Path(work)
    pip = [sys.executable, '-m', 'pip']
    index = ['--index-url', index_url] if index_url else []
    commands = [
        pip + ['install', '-r', str(requirements / 'vabench-build.txt')] + index,
        pip + ['install', '--no-build-isolation', '-r', str(requirements / 'vabench.txt')] + index,
        pip + ['install', '--no-deps', '--no-build-isolation', '--force-reinstall',
               '-r', str(requirements / 'vabench-git.txt')] + index,
        pip + ['download', '--only-binary=:all:', '--no-deps', 'nisqa==2.0.post2',
               '--dest', str(work / 'original')] + index,
    ]
    if system_deps:
        commands = [['apt-get', 'update'],
                    ['apt-get', 'install', '-y', '--no-install-recommends', *SYSTEM_PACKAGES]] + commands
    return commands


def patch_nisqa(wheel, destination, expected_sha=NISQA_SHA256):
    """Change exactly five Requires-Dist pins and regenerate wheel RECORD.

    Package code, weights, version, and the other dependency pins stay intact.
    The locally patched distribution is identified by its audit hashes.
    """
    wheel, destination = Path(wheel), Path(destination)
    if digest(wheel) != expected_sha:
        raise ValueError('NISQA original wheel SHA256 differs; refusing to patch')
    with zipfile.ZipFile(wheel) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate wheel members')
        meta_names = [n for n in names if n.endswith('.dist-info/METADATA')]
        if meta_names != ['nisqa-2.0.post2.dist-info/METADATA']:
            raise ValueError('Unexpected NISQA wheel metadata')
        meta_name = meta_names[0]
        text = source.read(meta_name).decode('utf-8')
        changes = {}
        for name, version in NISQA_PINS.items():
            pattern = (r'(?m)^(Requires-Dist: ' + re.escape(name)
                       + r')\s*==\s*' + re.escape(version) + r'[ \t]*$')
            text, count = re.subn(pattern, lambda m: m[1] + '>=' + version, text)
            if count != 1:
                raise ValueError(f'Expected exactly one NISQA pin for {name}')
            changes[name] = {'from': '==' + version, 'to': '>=' + version}
        record_name = 'nisqa-2.0.post2.dist-info/RECORD'
        data = {n: source.read(n) for n in names if n != record_name}
        data[meta_name] = text.encode('utf-8')
        rows = []
        for name, content in data.items():
            sha = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b'=').decode()
            rows.append([name, 'sha256=' + sha, len(content)])
        rows.append([record_name, '', ''])
        stream = io.StringIO(newline='')
        csv.writer(stream).writerows(rows)
        data[record_name] = stream.getvalue().encode('utf-8')
        destination.mkdir(parents=True, exist_ok=True)
        output = destination / wheel.name
        if output.resolve() == wheel.resolve():
            raise ValueError('Patched wheel must not overwrite original')
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as target:
            for info in source.infolist():
                target.writestr(info, data[info.filename])
    audit = dict(original_sha256=expected_sha, patched_sha256=digest(output),
                 changed_members=[meta_name, record_name], dependency_changes=changes,
                 package_code_unchanged=True, distribution='nisqa==2.0.post2')
    write_json(destination / 'nisqa_patch.json', audit)
    return output, audit


def prepare_external(root, cache):
    """Add missing external sources/layout, never overwrite official files."""
    root, cache = Path(root).resolve(), Path(cache).resolve()
    catalog(root, 'standard')  # Verify the pinned data before adding dependencies.
    dns = root / 'third_party/DNS-Challenge/DNSMOS/dnsmos_local.py'
    if not dns.exists():
        dns.parent.mkdir(parents=True, exist_ok=True)
        # Finish the transfer before exposing an importable source file.
        with urllib.request.urlopen(DNSMOS_URL, timeout=120) as response:
            content = response.read()
        if b'class ComputeScore' not in content:
            raise ValueError('Unexpected DNSMOS source response')
        dns.write_bytes(content)
    synch = root / 'third_party/Synchformer'
    cached_source = cache / 'synchformer/Synchformer'
    if not os.path.lexists(synch) and cached_source.is_dir():
        synch.symlink_to(cached_source, target_is_directory=True)
    buffalo = cache / 'checkpoints/auxiliary/models/buffalo_l'
    alternate = cache / 'checkpoints/auxiliary/buffalo_l'
    if not os.path.lexists(buffalo) and alternate.is_dir():
        buffalo.parent.mkdir(parents=True, exist_ok=True)
        buffalo.symlink_to(alternate, target_is_directory=True)
    return dict(dnsmos_path=str(dns), dnsmos_sha256=digest(dns),
                dnsmos_download_url=DNSMOS_URL,
                note='Existing DNSMOS source retained; inspect its hash for baseline parity.')


def runtime_guard(require_git=True):
    if sys.version_info[:2] != (3, 11) or platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Install target must be Linux x86_64 Python 3.11 (activate the scoring Conda).')
    if require_git and not shutil.which('git'):
        raise RuntimeError('Install system git before installing pinned Git dependencies.')
    # NISQA pins Pillow 10.2; AISBench runtime pins 11.2. Do not silently break it.
    try:
        metadata.version('ais-bench-benchmark')
    except metadata.PackageNotFoundError:
        return
    raise RuntimeError('AISBench is installed in this environment. Use a separate scoring Conda '
                       'and select its python3 with BACKEND_PYTHON/--python.')


def system_install_guard():
    """Only the explicit CLI option may opt into OS package installation."""
    if getattr(os, 'geteuid', lambda: -1)() != 0:
        raise RuntimeError('--install-system-deps requires root inside the target container.')
    try:
        distro = platform.freedesktop_os_release().get('ID')
    except OSError as exc:
        raise RuntimeError('Cannot identify the Linux distribution for apt installation.') from exc
    if distro not in ('ubuntu', 'debian') or not shutil.which('apt-get'):
        raise RuntimeError('--install-system-deps supports Ubuntu/Debian with apt-get only; '
                           'install equivalent system tools using your distribution package manager.')


def _probe(code, env, timeout=120):
    try:
        result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True,
                                text=True, encoding='utf-8', errors='replace', timeout=timeout)
        return dict(passed=result.returncode == 0, returncode=result.returncode,
                    log=(result.stdout + result.stderr)[-6000:])
    except subprocess.TimeoutExpired:
        return dict(passed=False, log=f'Probe timed out after {timeout}s')


def expected_packages(requirements):
    pins = {}
    for name in ('vabench-build.txt', 'vabench.txt'):
        for line in requirement_lines(Path(requirements) / name):
            package, version = line.split('==')
            pins[package.lower().replace('_', '-')] = version
    return pins


def check_environment(root, cache, requirements=REQUIREMENTS, video=None):
    """Offline imports, package/source checks and tiny CUDA/decode probes.

    Imports alone do not run model inference and do not certify all 15 scores.
    """
    root, cache = Path(root).resolve(), Path(cache).resolve()
    env = dict(os.environ, VABENCH_CACHE_DIR=str(cache), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1')
    prefix = 'import sys; sys.path.insert(0, ' + repr(str(root)) + '); '
    versions = {}
    for name, expected in expected_packages(requirements).items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
        versions[name] = dict(expected=expected, actual=actual, passed=actual == expected)
    # PyPI's torch version may have a local CUDA suffix (2.6.0+cu124).
    for name in ('torch', 'torchaudio', 'torchvision'):
        item = versions[name]
        item['passed'] = bool(item['actual'] and item['actual'].split('+')[0] == item['expected'])
    git = {}
    for line in requirement_lines(Path(requirements) / 'vabench-git.txt'):
        name, url = line.split(' @ ')
        expected = url.rsplit('@', 1)[1]
        try:
            dist = metadata.distribution(name)
            origin = json.loads(dist.read_text('direct_url.json') or '{}')
            actual = origin.get('vcs_info', {}).get('commit_id')
        except metadata.PackageNotFoundError:
            actual = None
        git[name] = dict(expected_commit=expected, actual_commit=actual, passed=actual == expected)
    nisqa = dict(passed=False)
    try:
        from packaging.requirements import Requirement
        dist = metadata.distribution('nisqa')
        deps = {r.name.lower(): str(r.specifier) for r in map(Requirement, dist.requires or [])}
        nisqa = dict(version=dist.version, patched_pins={n: deps.get(n) for n in NISQA_PINS},
                     passed=dist.version == '2.0.post2' and all(
                         deps.get(n) == '>=' + v for n, v in NISQA_PINS.items()))
    except (metadata.PackageNotFoundError, ImportError):
        pass
    modules = {}
    for dimension in DIMENSIONS:
        modules[dimension] = _probe(prefix + f'import vabench.{dimension}', env)
    modules['dnsmos_external'] = _probe(
        'import runpy; r=runpy.run_path(' + repr(str(root / 'third_party/DNS-Challenge/DNSMOS/dnsmos_local.py'))
        + '); assert callable(r["ComputeScore"])', env)
    modules['imagebind_data'] = _probe('from imagebind import data; from pytorchvideo import transforms', env)
    modules['qwen_classes'] = _probe(
        'from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor; '
        'from qwen_omni_utils import process_mm_info', env)
    modules['synchformer_lazy_imports'] = _probe(
        prefix + 'import vabench.second_desync as m; assert m._lazy_import_synchformer()', env)
    modules['onnx_providers'] = _probe(
        'import onnxruntime as ort; from insightface.app import FaceAnalysis; '
        'print(ort.get_available_providers()); assert "CUDAExecutionProvider" in ort.get_available_providers()', env)
    modules['cuda'] = _probe(
        'import torch, torchvision; assert torch.cuda.is_available(); '
        'print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); '
        'x=torch.ones((2,2),device="cuda"); print(x@x); '
        'print(torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]],device="cuda"),'
        'torch.tensor([1.],device="cuda"),0.5))', env)
    media = {n: shutil.which(n) for n in ('ffmpeg', 'ffprobe', 'git', 'gcc', 'g++', 'make')}
    # A decord platform-tag warning is accepted only after an actual decode.
    with tempfile.TemporaryDirectory(prefix='vabench_decode_') as temp:
        clip = Path(video).resolve() if video else Path(temp) / 'probe.mp4'
        try:
            if video is None:
                subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-f', 'lavfi',
                                '-i', 'color=c=blue:s=64x64:r=25:d=0.4', '-c:v', 'mpeg4', str(clip)],
                               capture_output=True, check=True, timeout=30)
            modules['decord_decode'] = _probe(
                'import decord; v=decord.VideoReader(' + repr(str(clip))
                + '); print(v[0].asnumpy().shape); assert len(v)>0', env)
        except (OSError, subprocess.SubprocessError) as exc:
            modules['decord_decode'] = dict(passed=False, log=str(exc))
    checked = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True,
                             text=True, encoding='utf-8', errors='replace', timeout=120)
    lines = [s for s in (checked.stdout + checked.stderr).splitlines() if s.strip()]
    decord_warning_only = (checked.returncode == 1 and bool(lines) and
                          all(s.strip() == 'decord 0.6.0 is not supported on this platform' for s in lines)
                          and modules['decord_decode']['passed'])
    pip_check = dict(passed=checked.returncode == 0 or decord_warning_only,
                     accepted_decord_tag_warning=decord_warning_only, log='\n'.join(lines))
    missing = preflight(root, cache, DIMENSIONS, strict=True)
    result = dict(python=sys.version, executable=sys.executable, versions=versions, git=git,
                  nisqa=nisqa, probes=modules, tools=media, missing_resources=missing, pip_check=pip_check,
                  model_inference_verified=False)
    result['passed'] = (sys.version_info[:2] == (3, 11) and all(media.values()) and not missing
                        and all(v['passed'] for v in versions.values()) and all(v['passed'] for v in git.values())
                        and nisqa['passed'] and pip_check['passed'] and all(v['passed'] for v in modules.values()))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan', 'install', 'check'))
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--cache-dir', required=True)
    parser.add_argument('--work-dir', required=True, help='Installation audit and patched wheel directory')
    parser.add_argument('--requirements-root', default=str(REQUIREMENTS))
    parser.add_argument('--index-url', help='Optional PyPI mirror; CA/proxy settings are inherited')
    parser.add_argument('--video', help='Existing video for the offline decord decode check')
    parser.add_argument('--install-system-deps', action='store_true',
                        help='Opt in to apt installation of system tools (Ubuntu/Debian container root)')
    args = parser.parse_args(argv)
    work = Path(args.work_dir).resolve()
    requirements = Path(args.requirements_root).resolve()
    if args.install_system_deps and args.command == 'check':
        parser.error('--install-system-deps is available only with plan/install')
    plan = install_plan(requirements, work, args.index_url, args.install_system_deps)
    if args.command == 'plan':
        print(json.dumps(dict(commands=plan, next_steps=[
            'Verify and patch original NISQA wheel; install it with --no-deps.',
            'Prepare missing DNSMOS/Synchformer sources and buffalo_l layout without overwrites.',
            'Run offline imports, CUDA/decode probes and pip check; save audit JSON.']), indent=2))
        return 0
    work.mkdir(parents=True, exist_ok=True)
    if args.command == 'install':
        runtime_guard(require_git=not args.install_system_deps)
        if args.install_system_deps:
            system_install_guard()
        if not Path(args.cache_dir).is_dir():
            parser.error('Prepare the model cache first; pip requirements do not contain weights.')
        catalog(args.source_root, 'standard')
        manifest = dict(status='installing', executable=sys.executable, python=sys.version,
                        system_packages_requested=list(SYSTEM_PACKAGES) if args.install_system_deps else [],
                        requirement_sha256={p.name: digest(p) for p in requirements.glob('vabench*.txt')})
        write_json(work / 'install.json', manifest)
        try:
            for step, command in enumerate(plan, 1):
                print(f'Installing dependency stage {step}/{len(plan)}', flush=True)
                env = dict(os.environ, DEBIAN_FRONTEND='noninteractive') if command[0] == 'apt-get' else None
                subprocess.run(command, check=True, env=env)
            wheel = work / 'original/nisqa-2.0.post2-py3-none-any.whl'
            patched, audit = patch_nisqa(wheel, work / 'patched')
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                            '--force-reinstall', str(patched)], check=True)
            manifest['nisqa'] = audit
            manifest['external_sources'] = prepare_external(args.source_root, args.cache_dir)
            manifest['status'] = 'installed_pending_checks'
        except Exception as exc:
            manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            write_json(work / 'install.json', manifest)
            raise
        write_json(work / 'install.json', manifest)
        frozen = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True)
        (work / 'installed.freeze.txt').write_text(frozen, encoding='utf-8')
    try:
        report = check_environment(args.source_root, args.cache_dir, requirements, args.video)
    except Exception as exc:
        report = dict(passed=False, model_inference_verified=False,
                      error=f'{type(exc).__name__}: {exc}')
    write_json(work / 'environment_check.json', report)
    if args.command == 'install':
        manifest['status'] = 'checks_passed' if report['passed'] else 'checks_failed'
        write_json(work / 'install.json', manifest)
    print(f'Environment checks: {"PASS" if report["passed"] else "FAIL"}; '
          f'report: {work / "environment_check.json"}', flush=True)
    return int(not report['passed'])


if __name__ == '__main__':
    raise SystemExit(main())
