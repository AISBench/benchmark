#!/usr/bin/env python3
"""Cache successful smoke dependency setup, never smoke run output.

Use --force or AISBENCH_SMOKE_FORCE_INSTALL=1 to reinstall. Only local
requirements includes are cacheable; remote includes still install normally.
The lock coordinates this helper across repositories sharing one interpreter;
external pip/conda changes must not run concurrently with smoke setup.
"""

import argparse
import ast
from contextlib import contextmanager
import hashlib
from importlib import metadata
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time


REQUIREMENTS = (
    'smoke_tests/requirements.txt', 'requirements.txt',
    'requirements/runtime.txt', 'requirements/extra.txt',
    'requirements/datasets/bfcl_dependencies.txt',
    'tools/infer_serve_simulator/requirements.txt',
)
METADATA_INPUTS = (
    'setup.py', 'setup.cfg', 'pyproject.toml', 'MANIFEST.in',
    'README.md', 'ais_bench/benchmark/__init__.py',
)
INCLUDE = re.compile(r'^(?:--(?:requirement|constraint)(?:=|\s+)|-[rc]\s*)(.+)$')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def environment_identity():
    return {'executable': os.path.realpath(sys.executable),
            'prefix': os.path.realpath(sys.prefix),
            'base_prefix': os.path.realpath(sys.base_prefix),
            'version': sys.version}


def cache_path(repo, identity):
    scope = digest({'repo': str(repo.resolve()), 'python': identity})
    return repo / 'smoke_tests' / '.cache' / 'dependencies' / (scope + '.json')


def input_fingerprint(repo):
    """Hash setup inputs and recursive pip -r/-c includes, not git HEAD/source."""
    # External requirements may refer to mutable remote/local inputs outside the
    # repo. Be conservative rather than ignoring constraints on a warm run.
    if any(os.environ.get(key) for key in (
            'PIP_CONSTRAINT', 'PIP_BUILD_CONSTRAINT', 'PIP_REQUIREMENT', 'PIP_CONFIG_FILE')):
        raise ValueError('external pip requirements/configuration; cache bypassed')
    files = {'pip_environment': digest({key: value for key, value in os.environ.items()
                                        if key.startswith('PIP_')})}
    home = Path.home()
    config_dirs = os.environ.get('XDG_CONFIG_DIRS', '/etc/xdg').split(os.pathsep)
    configs = [Path(directory) / 'pip/pip.conf' for directory in config_dirs if directory]
    configs += [Path('/etc/pip.conf'), home / '.pip/pip.conf',
                Path(os.environ.get('XDG_CONFIG_HOME', home / '.config')) / 'pip/pip.conf',
                Path(sys.prefix) / 'pip.conf', Path(sys.prefix) / 'pip.ini', home / 'pip/pip.ini']
    for variable in ('APPDATA', 'PROGRAMDATA'):
        if os.environ.get(variable):
            configs.append(Path(os.environ[variable]) / 'pip/pip.ini')
    for config in configs:
        if config.is_file():
            data = config.read_bytes()
            if re.search(rb'(?im)^\s*(?:build-constraint|constraint|requirement)\s*=', data):
                raise ValueError('pip config specifies external requirements; cache bypassed')
            files[str(config)] = hashlib.sha256(data).hexdigest()

    def requirements(path):
        path = path.resolve()
        if str(path) in files:
            return
        data = path.read_bytes()
        files[str(path)] = hashlib.sha256(data).hexdigest()
        text = re.sub(r'\\\r?\n', '', data.decode('utf-8-sig'))
        for line in text.splitlines():
            match = INCLUDE.match(line.strip())
            if not match:
                continue
            names = shlex.split(match.group(1), comments=True)
            if len(names) != 1:
                raise ValueError('unsupported requirements include: ' + line)
            name = os.path.expandvars(names[0])
            if '://' in name or '${' in name:
                raise ValueError('non-local/unresolved requirements include: ' + name)
            requirements(path.parent / name)

    for name in REQUIREMENTS:
        requirements(repo / name)
    # setup.py also reads extras while generating editable package metadata.
    setup = repo / 'setup.py'
    for node in ast.walk(ast.parse(setup.read_text(encoding='utf-8'))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith('requirements/') and node.value.endswith('.txt'):
                requirements(repo / node.value)
    for name in METADATA_INPUTS:
        path = repo / name
        files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    files['helper'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return digest(files)


def installed_state(repo):
    """Return None on incomplete metadata or a non-editable/wrong root install."""
    records, root_found = [], False
    importlib.invalidate_caches()
    try:
        for dist in metadata.distributions():
            info = dist.read_text('METADATA') or dist.read_text('PKG-INFO')
            name, version = dist.metadata.get('Name'), dist.metadata.get('Version')
            if not info or not name or not version:
                return None
            direct_url = dist.read_text('direct_url.json')
            if re.sub(r'[-_.]+', '-', name).lower() == 'ais-bench-benchmark' and direct_url:
                # Editable installs can also expose source .egg-info with no
                # direct_url. Require at least one real editable-install proof.
                direct = json.loads(direct_url)
                if (direct.get('url') != repo.as_uri()
                        or direct.get('dir_info', {}).get('editable') is not True):
                    return None
                root_found = True
            records.append((name.lower(), version, str(Path(dist.locate_file('')).resolve()),
                            info, direct_url or ''))
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    return digest(sorted(records)) if root_found else None


@contextmanager
def environment_lock(identity):
    # Common to all worktrees for this interpreter; do not unlink lock files:
    # waiters must continue locking the same inode after another process exits.
    path = Path(tempfile.gettempdir()) / ('aisbench-smoke-' + digest(identity) + '.lock')
    with path.open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            if lock.tell() == 0:
                lock.write(b'\0')
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def install(repo):
    stages = (
        ['-r', 'smoke_tests/requirements.txt'],
        ['-e', '.', '--use-pep517'],
        ['-r', 'requirements/extra.txt'],
        ['-r', 'requirements/datasets/bfcl_dependencies.txt', '--no-deps'],
        ['-r', 'tools/infer_serve_simulator/requirements.txt'],
    )
    for index, args in enumerate(stages, 1):
        started = time.monotonic()
        print('[dependencies] stage {}/5: {}'.format(index, ' '.join(args)), flush=True)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *args], cwd=repo)
        print('[dependencies] stage {} finished in {:.2f}s'.format(
            index, time.monotonic() - started), flush=True)


def write_stamp(path, stamp):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(stamp, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare(repo, force=False):
    repo = Path(repo).resolve()
    identity = environment_identity()
    path = cache_path(repo, identity)
    with environment_lock(identity):
        try:
            fingerprint = input_fingerprint(repo)
        except ValueError as error:
            print('[dependencies] cache disabled:', error, flush=True)
            fingerprint = None
        except (OSError, SyntaxError):
            path.unlink(missing_ok=True)
            raise
        try:
            stamp = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            stamp = None
        state = installed_state(repo)
        expected = {'schema': 1, 'inputs': fingerprint, 'installed': state}
        if not force and fingerprint and state and stamp == expected:
            print('[dependencies] cache hit; skipping all five pip stages', flush=True)
            return True
        # Remove even an old successful stamp before any potentially partial install.
        path.unlink(missing_ok=True)
        print('[dependencies] installing all five pip stages', flush=True)
        install(repo)
        state = installed_state(repo)
        if fingerprint and state and input_fingerprint(repo) == fingerprint:
            write_stamp(path, {'schema': 1, 'inputs': fingerprint, 'installed': state})
        else:
            print('[dependencies] no stamp: inputs changed or metadata incomplete', flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    force = args.force or os.environ.get('AISBENCH_SMOKE_FORCE_INSTALL', '').lower() in (
        '1', 'true', 'yes', 'on')
    try:
        prepare(args.repo, force)
    except (OSError, ValueError, SyntaxError, subprocess.CalledProcessError) as error:
        print('[dependencies] setup failed: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
