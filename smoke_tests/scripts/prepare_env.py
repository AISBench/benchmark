#!/usr/bin/env python3
"""Cache successful smoke dependency setup, never smoke run output.

Use --force or AISBENCH_SMOKE_FORCE_INSTALL=1 to reinstall. Only local
requirements includes are cacheable; remote includes still install normally.
Stamps live outside the checkout in the user cache (XDG_CACHE_HOME or the
platform default); AISBENCH_SMOKE_CACHE_DIR overrides that directory. Cache
paths resolving inside the repository are rejected because git clean removes
them. AISBENCH_SMOKE_SETUP_LOCK_TIMEOUT bounds lock waiting (default 120s).
The lock coordinates this helper across repositories sharing one interpreter;
external pip/conda changes must not run concurrently with smoke setup.
"""

import argparse
import ast
from contextlib import contextmanager
import errno
import hashlib
from importlib import metadata
import importlib
import json
import math
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


def cache_root(repo=None):
    """Select a persistent user cache, never a git-cleanable checkout path."""
    configured = os.environ.get('AISBENCH_SMOKE_CACHE_DIR')
    if configured is not None:
        if not configured.strip():
            raise ValueError('AISBENCH_SMOKE_CACHE_DIR must not be empty')
        root = Path(configured)
    else:
        xdg = os.environ.get('XDG_CACHE_HOME')
        if xdg:
            base = Path(xdg)
        elif os.name == 'nt':
            base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData/Local')
        elif sys.platform == 'darwin':
            base = Path.home() / 'Library/Caches'
        else:
            base = Path.home() / '.cache'
        root = base / 'aisbench/smoke/dependencies'
    root = root.expanduser().resolve()
    if repo is not None and root.is_relative_to(Path(repo).resolve()):
        raise ValueError('smoke cache must be outside the repository: ' + str(root))
    return root


def cache_scope(repo, identity):
    return digest({'repo': str(Path(repo).resolve()), 'python': identity})


def cache_path(repo, identity):
    return cache_root(repo) / (cache_scope(repo, identity) + '.json')


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
                raise ValueError('unsupported requirements include; cache bypassed')
            name = os.path.expandvars(names[0])
            if '://' in name or '${' in name:
                raise ValueError('non-local/unresolved requirements include; cache bypassed')
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
    """Hash real installs; ignore only the matching in-tree editable duplicate."""
    repo = Path(repo).resolve()
    records, source_duplicates, editable_versions = [], [], set()
    importlib.invalidate_caches()
    try:
        for dist in metadata.distributions():
            info = dist.read_text('METADATA') or dist.read_text('PKG-INFO')
            name, version = dist.metadata.get('Name'), dist.metadata.get('Version')
            if not info or not name or not version:
                return None
            direct_url = dist.read_text('direct_url.json')
            project = re.sub(r'[-_.]+', '-', name).lower() == 'ais-bench-benchmark'
            if project and direct_url:
                # Source egg-info alone is not proof of an installed editable.
                direct = json.loads(direct_url)
                if (direct.get('url') != repo.as_uri()
                        or direct.get('dir_info', {}).get('editable') is not True):
                    return None
                editable_versions.add(version)
            record = (name.lower(), version, str(Path(dist.locate_file('')).resolve()),
                      info, direct_url or '')
            # PathDistribution exposes its metadata directory as _path. If a
            # provider cannot prove this is source-root egg-info, hash it
            # normally, including project installs in an in-tree virtualenv.
            location = getattr(dist, '_path', None)
            if project and not direct_url and location is not None:
                location = Path(location).resolve()
                if (location.name.endswith('.egg-info') and location.is_relative_to(repo)
                        and record[2] == str(repo)):
                    source_duplicates.append(record)
                    continue
            records.append(record)
        if not editable_versions:
            return None
        # git clean can delete generated source metadata without changing the
        # installed dist-info. A different source version is not a duplicate.
        records.extend(record for record in source_duplicates
                       if record[1] not in editable_versions)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None
    return digest(sorted(records))


@contextmanager
def environment_lock(identity):
    # Common to all worktrees for this interpreter; do not unlink lock files:
    # waiters must continue locking the same inode after another process exits.
    timeout = float(os.environ.get('AISBENCH_SMOKE_SETUP_LOCK_TIMEOUT', '120'))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('AISBENCH_SMOKE_SETUP_LOCK_TIMEOUT must be positive finite seconds')
    path = Path(tempfile.gettempdir()) / ('aisbench-smoke-' + digest(identity) + '.lock')
    with path.open('a+b') as lock:
        if os.name == 'nt':
            import msvcrt
            if lock.tell() == 0:
                lock.write(b'\0')
                lock.flush()

            def acquire():
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

        deadline = time.monotonic() + timeout
        while True:
            try:
                acquire()
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('smoke setup lock timed out after {:.2f}s: {}; '
                                       'adjust AISBENCH_SMOKE_SETUP_LOCK_TIMEOUT'.format(
                                           timeout, path)) from error
                time.sleep(min(0.1, remaining))
        try:
            yield
        finally:
            release()


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
    started = time.monotonic()
    try:
        repo = Path(repo).resolve()
        identity = environment_identity()
        path = cache_path(repo, identity)
        print('[dependencies] cache path: ' + str(path), flush=True)
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
                print('[dependencies] cache hit: setup inputs and installed metadata match; '
                      'skipping all five pip stages', flush=True)
                return True
            if force:
                reason = 'forced reinstall'
            elif not fingerprint:
                reason = 'uncacheable setup inputs'
            elif not state:
                reason = 'installed metadata incomplete or editable install unverified'
            elif stamp is None:
                reason = 'no readable successful stamp'
            elif not isinstance(stamp, dict) or stamp.get('schema') != 1:
                reason = 'unsupported stamp schema'
            elif stamp.get('inputs') != fingerprint:
                reason = 'setup inputs changed'
            elif stamp.get('installed') != state:
                reason = 'installed distributions changed'
            else:
                reason = 'invalid successful stamp'
            # Remove even an old stamp before any potentially partial install.
            path.unlink(missing_ok=True)
            print('[dependencies] cache miss: ' + reason + '; installing all five pip stages',
                  flush=True)
            install(repo)
            state = installed_state(repo)
            if fingerprint and state and input_fingerprint(repo) == fingerprint:
                write_stamp(path, {'schema': 1, 'inputs': fingerprint, 'installed': state})
            else:
                print('[dependencies] no stamp: inputs changed or metadata incomplete', flush=True)
            return False
    finally:
        print('[dependencies] total setup time: {:.2f}s'.format(time.monotonic() - started),
              flush=True)


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
