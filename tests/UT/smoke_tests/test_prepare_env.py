"""Dependency-free tests: no pip installation or AISBench imports."""

from contextlib import ExitStack, nullcontext, redirect_stdout
import errno
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[3] / 'smoke_tests/scripts/prepare_env.py'
SPEC = importlib.util.spec_from_file_location('smoke_prepare_env', SCRIPT)
env = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(env)
REAL_LOCK = env.environment_lock


class PrepareEnvTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name).resolve() / 'repo'
        self.repo.mkdir()
        for name in env.REQUIREMENTS:
            self.put(name, 'example==1\n')
        self.put('setup.py', "extras = 'requirements/extra.txt'\n")
        self.put('README.md', 'description\n')
        self.put('ais_bench/benchmark/__init__.py', '__version__ = "1"\n')
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.output = io.StringIO()
        stack.enter_context(redirect_stdout(self.output))
        # Every default/override cache and pip configuration is isolated from the
        # developer/runner account, including tests exercising platform fallback.
        self.home = self.repo.parent / 'home'
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(('PIP_', 'AISBENCH_SMOKE_'))}
        environment.update({
            'XDG_CACHE_HOME': str(self.repo.parent / 'cache'),
            'LOCALAPPDATA': str(self.repo.parent / 'localappdata'),
            'HOME': str(self.home),
            'USERPROFILE': str(self.home),
            'XDG_CONFIG_HOME': str(self.repo.parent / 'config'),
            'XDG_CONFIG_DIRS': str(self.repo.parent / 'config-dirs'),
            'APPDATA': str(self.repo.parent / 'appdata'),
            'PROGRAMDATA': str(self.repo.parent / 'programdata'),
        })
        stack.enter_context(mock.patch.dict(env.os.environ, environment, clear=True))
        stack.enter_context(mock.patch.object(env.Path, 'home', return_value=self.home))
        self.pip = stack.enter_context(mock.patch.object(env.subprocess, 'check_call'))
        self.lock = stack.enter_context(mock.patch.object(
            env, 'environment_lock', side_effect=lambda identity: nullcontext()))
        self.distributions = [self.dist('ais_bench_benchmark'), self.dist('example')]
        stack.enter_context(mock.patch.object(
            env.metadata, 'distributions', side_effect=lambda: self.distributions))

    def put(self, name, content):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return path

    def dist(self, name, version='1', repo=None):
        repo = repo or self.repo
        texts = {'METADATA': 'Name: {}\nVersion: {}\n'.format(name, version)}
        if name == 'ais_bench_benchmark':
            texts['direct_url.json'] = json.dumps(
                {'url': repo.as_uri(), 'dir_info': {'editable': True}})
        site = self.repo.parent / 'site'
        return SimpleNamespace(metadata={'Name': name, 'Version': version},
                               read_text=texts.get, locate_file=lambda _: site,
                               _path=site / (name + '-' + version + '.dist-info'))

    @property
    def stamp(self):
        return env.cache_path(self.repo, env.environment_identity())

    def test_cold_then_warm_zero_pip_and_original_stages(self):
        self.assertFalse(env.prepare(self.repo))
        self.assertTrue(self.stamp.is_file())
        calls = self.pip.call_args_list
        self.assertEqual(len(calls), 5)
        self.assertEqual(calls[1].args[0][4:], ['-e', '.', '--use-pep517'])
        self.assertEqual(calls[3].args[0][-1], '--no-deps')
        self.assertEqual(calls[4].args[0][-1], 'tools/infer_serve_simulator/requirements.txt')
        for call in calls:
            self.assertEqual(call.args[0][:4], [sys.executable, '-m', 'pip', 'install'])
            self.assertEqual(call.kwargs['cwd'], self.repo)
            self.assertNotIn('--no-build-isolation', call.args[0])
        self.pip.reset_mock()
        self.assertTrue(env.prepare(self.repo))
        self.pip.assert_not_called()
        self.assertEqual(self.lock.call_count, 2)

    def source_dist(self, version='1'):
        source = self.dist('ais_bench_benchmark', version)
        source.read_text = lambda key: ('Name: ais_bench_benchmark\nVersion: ' + version
                                        if key == 'PKG-INFO' else None)
        source._path = self.repo / 'ais_bench_benchmark.egg-info'
        source.locate_file = lambda _: self.repo
        return source

    def test_removed_source_egg_info_duplicate_keeps_warm_cache(self):
        source = self.source_dist()
        self.distributions.insert(0, source)
        env.prepare(self.repo)
        original_state = env.installed_state(self.repo)
        self.distributions.remove(source)
        self.assertEqual(original_state, env.installed_state(self.repo))
        self.pip.reset_mock()
        self.assertTrue(env.prepare(self.repo))
        self.pip.assert_not_called()

    def test_only_matching_in_tree_project_egg_info_is_ignored(self):
        original_state = env.installed_state(self.repo)
        candidates = [self.source_dist('2'), self.source_dist(), self.source_dist(),
                      self.dist('example'), self.source_dist()]
        candidates[1]._path = self.repo.parent / 'external/ais_bench_benchmark.egg-info'
        candidates[2]._path = self.repo / 'ais_bench_benchmark.dist-info'
        candidates[3]._path = self.repo / 'example.egg-info'
        candidates[4]._path = self.repo / '.venv/site/ais_bench_benchmark.egg-info'
        candidates[4].locate_file = lambda _: self.repo / '.venv/site'
        for source in candidates:
            with self.subTest(path=source._path):
                self.distributions.append(source)
                self.assertNotEqual(original_state, env.installed_state(self.repo))
                self.distributions.pop()
        self.distributions[:] = [self.source_dist()]
        self.assertIsNone(env.installed_state(self.repo))

    def test_force_reinstalls(self):
        env.prepare(self.repo)
        self.pip.reset_mock()
        self.assertFalse(env.prepare(self.repo, force=True))
        self.assertEqual(self.pip.call_count, 5)

    def test_every_input_and_recursive_requirements_invalidate(self):
        self.put('smoke_tests/requirements.txt', '-r nested.txt\n')
        self.put('smoke_tests/nested.txt', '--constraint=limits.txt\n')
        self.put('smoke_tests/limits.txt', '-rrequirements.txt\nexample==1\n')
        self.put('setup.py', "extras = 'requirements/custom.txt'\n")
        self.put('requirements/custom.txt', 'example==1\n')
        env.prepare(self.repo)
        names = (*env.REQUIREMENTS, *env.METADATA_INPUTS,
                 'smoke_tests/nested.txt', 'smoke_tests/limits.txt', 'requirements/custom.txt')
        for name in names:
            with self.subTest(name=name):
                path = self.repo / name
                old = path.read_text(encoding='utf-8') if path.exists() else ''
                self.put(name, old + '\n# changed\n')
                self.pip.reset_mock()
                self.assertFalse(env.prepare(self.repo))
                self.assertEqual(self.pip.call_count, 5)

    def test_source_changes_stay_editable_without_reinstall(self):
        env.prepare(self.repo)
        self.put('ais_bench/benchmark/cli/main.py', 'new_source = True\n')
        self.pip.reset_mock()
        self.assertTrue(env.prepare(self.repo))
        self.pip.assert_not_called()

    def test_distribution_version_or_removal_invalidates(self):
        env.prepare(self.repo)
        for remaining in ([self.dist('ais_bench_benchmark'), self.dist('example', '2')],
                          [self.dist('ais_bench_benchmark')]):
            self.distributions[:] = remaining
            self.pip.reset_mock()
            self.assertFalse(env.prepare(self.repo))
            self.assertEqual(self.pip.call_count, 5)

    def test_missing_metadata_or_wrong_editable_never_stamps(self):
        for broken in ([], [self.dist('example')],
                       [SimpleNamespace(metadata={}, read_text=lambda _: None)],
                       [self.dist('ais_bench_benchmark')]):
            with self.subTest(broken=broken):
                env.prepare(self.repo)
                if broken and broken[0].metadata.get('Name') == 'ais_bench_benchmark':
                    broken[0].read_text = lambda key: 'metadata' if key == 'METADATA' else None
                self.distributions[:] = broken
                self.pip.reset_mock()
                self.assertFalse(env.prepare(self.repo))
                self.assertEqual(self.pip.call_count, 5)
                self.assertFalse(self.stamp.exists())

    def test_editable_proof_rejects_wrong_root_noneditable_and_malformed_url(self):
        for direct in ({'url': (self.repo.parent / 'other').as_uri(),
                        'dir_info': {'editable': True}},
                       {'url': self.repo.as_uri(), 'dir_info': {'editable': False}},
                       {'url': self.repo.as_uri()}, 'malformed'):
            with self.subTest(direct=direct):
                root = self.dist('ais_bench_benchmark')
                original = root.read_text
                root.read_text = lambda key: (json.dumps(direct) if key == 'direct_url.json'
                                              else original(key))
                self.distributions[:] = [self.source_dist(), root]
                self.assertIsNone(env.installed_state(self.repo))
                self.assertFalse(env.prepare(self.repo))
                self.assertFalse(self.stamp.exists())

    def test_real_distribution_metadata_change_invalidates(self):
        env.prepare(self.repo)
        example = self.distributions[1]
        original = example.read_text
        example.read_text = lambda key: (original(key) + '\nRequires-Dist: new-dependency\n'
                                         if key == 'METADATA' else original(key))
        self.pip.reset_mock()
        self.assertFalse(env.prepare(self.repo))
        self.assertEqual(self.pip.call_count, 5)

    def test_each_failed_stage_removes_old_stamp_and_fails_fast(self):
        for failed in range(5):
            with self.subTest(stage=failed):
                env.prepare(self.repo)
                self.pip.reset_mock()
                self.pip.side_effect = [None] * failed + [subprocess.CalledProcessError(1, 'pip')]
                with self.assertRaises(subprocess.CalledProcessError):
                    env.prepare(self.repo, force=True)
                self.assertEqual(self.pip.call_count, failed + 1)
                self.assertFalse(self.stamp.exists())
                self.pip.side_effect = None

    def test_interpreter_and_repo_cache_scoping(self):
        original = env.environment_identity()
        path = env.cache_path(self.repo, original)
        self.assertEqual(path, env.cache_path(self.repo / '.', original))
        for key in ('executable', 'prefix', 'base_prefix', 'version'):
            changed = dict(original, **{key: original[key] + '-different'})
            self.assertNotEqual(path, env.cache_path(self.repo, changed))
        self.assertNotEqual(path, env.cache_path(self.repo.parent / 'other', original))
        env.prepare(self.repo)
        self.pip.reset_mock()
        with mock.patch.object(env, 'environment_identity', return_value=dict(original, prefix='other')):
            self.assertFalse(env.prepare(self.repo))
        self.assertEqual(self.pip.call_count, 5)
        # Identical tracked files in another checkout still need their own stamp.
        other = self.repo.parent / 'other'
        shutil.copytree(self.repo, other)
        self.distributions[:] = [self.dist('ais_bench_benchmark', repo=other), self.dist('example')]
        self.pip.reset_mock()
        self.assertFalse(env.prepare(other))
        self.assertEqual(self.pip.call_count, 5)
        self.pip.reset_mock()
        self.assertTrue(env.prepare(other))
        self.pip.assert_not_called()
        self.assertTrue(path.is_file())

    def test_persistent_default_platform_fallback_and_override(self):
        self.assertEqual(env.cache_root(self.repo),
                         self.repo.parent / 'cache/aisbench/smoke/dependencies')
        self.assertFalse(self.stamp.is_relative_to(self.repo))
        with mock.patch.dict(env.os.environ):
            env.os.environ.pop('XDG_CACHE_HOME')
            if os.name == 'nt':
                self.assertEqual(env.cache_root(),
                                 self.repo.parent / 'localappdata/aisbench/smoke/dependencies')
                env.os.environ.pop('LOCALAPPDATA')
                base = self.home / 'AppData/Local'
            elif sys.platform == 'darwin':
                base = self.home / 'Library/Caches'
            else:
                base = self.home / '.cache'
            self.assertEqual(env.cache_root(), base / 'aisbench/smoke/dependencies')
            if os.name != 'nt':
                with mock.patch.object(sys, 'platform', 'darwin'):
                    self.assertEqual(env.cache_root(),
                                     self.home / 'Library/Caches/aisbench/smoke/dependencies')
        with mock.patch.dict(env.os.environ, {'AISBENCH_SMOKE_CACHE_DIR': '~/custom-cache'}):
            self.assertEqual(env.cache_root(self.repo), self.home / 'custom-cache')
            self.assertFalse(env.prepare(self.repo))
            self.assertTrue(self.stamp.is_file())
            self.pip.reset_mock()
            self.assertTrue(env.prepare(self.repo))
            self.pip.assert_not_called()

    def test_cache_inside_repo_or_empty_override_is_rejected_before_pip(self):
        for root in (self.repo, self.repo / 'cache', self.repo / '../repo/nested', ''):
            with self.subTest(root=root), mock.patch.dict(
                    env.os.environ, {'AISBENCH_SMOKE_CACHE_DIR': str(root)}):
                with self.assertRaisesRegex(ValueError, 'outside the repository|must not be empty'):
                    env.prepare(self.repo)
                self.pip.assert_not_called()
                self.lock.assert_not_called()
        with mock.patch.dict(env.os.environ, {'XDG_CACHE_HOME': str(self.repo / 'xdg-cache')}):
            with self.assertRaisesRegex(ValueError, 'outside the repository'):
                env.prepare(self.repo)
        self.pip.assert_not_called()

    def test_cache_override_symlink_into_repo_is_rejected(self):
        alias = self.repo.parent / 'cache-alias'
        try:
            alias.symlink_to(self.repo, target_is_directory=True)
        except (OSError, NotImplementedError) as error:
            self.skipTest('directory symlinks unavailable: ' + str(error))
        self.assertEqual(env.cache_path(alias, env.environment_identity()), self.stamp)
        with mock.patch.dict(env.os.environ, {'AISBENCH_SMOKE_CACHE_DIR': str(alias / 'cache')}):
            with self.assertRaisesRegex(ValueError, 'outside the repository'):
                env.prepare(self.repo)
        self.pip.assert_not_called()

    @unittest.skipUnless(shutil.which('git'), 'git is required for checkout-clean regression')
    def test_real_git_clean_preserves_stamp_and_zero_pip_warm_hit(self):
        # Track every fingerprint input, including this helper, exactly as a
        # checkout does. No commit, user git config, or actual pip is required.
        helper = self.repo / 'smoke_tests/scripts/prepare_env.py'
        helper.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SCRIPT, helper)
        for args in (['init', '--quiet'], ['add', '--all']):
            subprocess.run(['git', *args], cwd=self.repo, check=True, capture_output=True)
        site = self.repo.parent / 'installed-site'
        installed = site / 'ais_bench_benchmark-1.dist-info'
        installed.mkdir(parents=True)
        (installed / 'METADATA').write_text('Name: ais_bench_benchmark\nVersion: 1\n', encoding='utf-8')
        (installed / 'direct_url.json').write_text(json.dumps(
            {'url': self.repo.as_uri(), 'dir_info': {'editable': True}}), encoding='utf-8')
        source = self.repo / 'ais_bench_benchmark.egg-info'

        def fake_pip(command, **kwargs):
            if '-e' in command:
                source.mkdir(exist_ok=True)
                (source / 'PKG-INFO').write_text(
                    'Name: ais_bench_benchmark\nVersion: 1\n', encoding='utf-8')

        def distributions():
            paths = [installed, *([source] if source.exists() else [])]
            return [env.metadata.PathDistribution(path) for path in paths]

        self.pip.side_effect = fake_pip
        with mock.patch.object(env, '__file__', str(helper)), mock.patch.object(
                env.metadata, 'distributions', side_effect=distributions):
            self.assertFalse(env.prepare(self.repo))
            self.assertEqual(self.pip.call_count, 5)
            stamp = self.stamp
            self.assertTrue(stamp.is_file())
            self.assertTrue(source.is_dir())
            state = env.installed_state(self.repo)
            self.put('smoke_tests/.cache/dependencies/old.json', 'old in-repo cache')
            subprocess.run(['git', 'clean', '-ffdx'], cwd=self.repo, check=True, capture_output=True)
            self.assertFalse(source.exists())
            self.assertFalse((self.repo / 'smoke_tests/.cache').exists())
            self.assertTrue(helper.is_file())
            self.assertTrue(stamp.is_file())
            self.assertEqual(state, env.installed_state(self.repo))
            self.pip.reset_mock()
            self.assertTrue(env.prepare(self.repo))
            self.pip.assert_not_called()

    def test_cache_path_reasons_and_total_time_are_logged(self):
        self.assertFalse(env.prepare(self.repo))
        self.assertTrue(env.prepare(self.repo))
        self.put('requirements/extra.txt', 'changed==2\n')
        self.assertFalse(env.prepare(self.repo))
        self.distributions.append(self.dist('other'))
        self.assertFalse(env.prepare(self.repo))
        self.assertFalse(env.prepare(self.repo, force=True))
        output = self.output.getvalue()
        for expected in (str(self.stamp), 'cache miss: no readable successful stamp',
                         'cache hit: setup inputs and installed metadata match',
                         'cache miss: setup inputs changed',
                         'cache miss: installed distributions changed',
                         'cache miss: forced reinstall'):
            self.assertIn(expected, output)
        self.assertEqual(output.count('total setup time:'), 5)

    def test_atomic_write_failure_no_stamp_or_temporary_file(self):
        with mock.patch.object(env.os, 'replace', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):
                env.prepare(self.repo)
        self.assertFalse(self.stamp.exists())
        self.assertEqual(list(self.stamp.parent.iterdir()), [])

    def test_remote_include_installs_but_disables_cache_without_logging_credentials(self):
        self.put('smoke_tests/requirements.txt',
                 '-r https://user:private-token@example.invalid/requirements.txt\n')
        env.prepare(self.repo)
        self.assertEqual(self.pip.call_count, 5)
        self.assertFalse(self.stamp.exists())
        self.assertIn('non-local/unresolved requirements include', self.output.getvalue())
        self.assertNotIn('private-token', self.output.getvalue())

    def test_helper_change_invalidates_and_missing_input_removes_stamp(self):
        helper = self.put('smoke_tests/scripts/prepare_env.py', '# helper version 1\n')
        with mock.patch.object(env, '__file__', str(helper)):
            env.prepare(self.repo)
            helper.write_text('# helper version 2\n', encoding='utf-8')
            self.pip.reset_mock()
            self.assertFalse(env.prepare(self.repo))
            self.assertEqual(self.pip.call_count, 5)
            (self.repo / env.REQUIREMENTS[0]).unlink()
            self.pip.reset_mock()
            with self.assertRaises(OSError):
                env.prepare(self.repo)
            self.assertFalse(self.stamp.exists())
            self.pip.assert_not_called()

    def test_pip_config_constraints_bypass_warm_cache(self):
        env.prepare(self.repo)
        config = Path(os.environ['XDG_CONFIG_HOME']) / 'pip/pip.conf'
        config.parent.mkdir(parents=True)
        config.write_text('[global]\nconstraint = external.txt\n', encoding='utf-8')
        self.pip.reset_mock()
        self.assertFalse(env.prepare(self.repo))
        self.assertEqual(self.pip.call_count, 5)
        self.assertFalse(self.stamp.exists())

    def test_external_pip_constraints_bypass_warm_cache(self):
        env.prepare(self.repo)
        for name in ('PIP_CONSTRAINT', 'PIP_BUILD_CONSTRAINT', 'PIP_REQUIREMENT', 'PIP_CONFIG_FILE'):
            with self.subTest(name=name), mock.patch.dict(env.os.environ, {name: 'external.txt'}):
                self.pip.reset_mock()
                self.assertFalse(env.prepare(self.repo))
                self.assertEqual(self.pip.call_count, 5)
                self.assertFalse(self.stamp.exists())

    def test_platform_lock_is_held_and_released_on_failure(self):
        backend = __import__('msvcrt' if env.os.name == 'nt' else 'fcntl')
        method = 'locking' if env.os.name == 'nt' else 'flock'
        with mock.patch.object(backend, method) as operation:
            with mock.patch.object(env.tempfile, 'gettempdir', return_value=str(self.repo.parent)):
                with self.assertRaisesRegex(RuntimeError, 'failed'):
                    with REAL_LOCK(env.environment_identity()):
                        self.assertEqual(operation.call_count, 1)
                        raise RuntimeError('failed')
        self.assertEqual(operation.call_count, 2)

    def test_lock_timeout_must_be_positive_and_finite(self):
        for timeout in ('0', '-1', 'nan', 'inf', '1e999', 'invalid'):
            with self.subTest(timeout=timeout), mock.patch.dict(
                    env.os.environ, {'AISBENCH_SMOKE_SETUP_LOCK_TIMEOUT': timeout}):
                with self.assertRaises(ValueError):
                    with REAL_LOCK(env.environment_identity()):
                        self.fail('invalid timeout acquired lock')

    def test_lock_uses_default_finite_timeout_and_preserves_stamp(self):
        env.prepare(self.repo)
        self.pip.reset_mock()
        backend = __import__('msvcrt' if env.os.name == 'nt' else 'fcntl')
        method = 'locking' if env.os.name == 'nt' else 'flock'
        with mock.patch.object(backend, method, side_effect=OSError(errno.EACCES, 'busy')):
            with mock.patch.object(env.tempfile, 'gettempdir', return_value=str(self.repo.parent)):
                with mock.patch.object(env.time, 'monotonic', side_effect=[0, 120]):
                    with self.assertRaisesRegex(TimeoutError, 'timed out after 120.00s'):
                        with REAL_LOCK(env.environment_identity()):
                            self.fail('contended lock acquired')
        self.assertTrue(self.stamp.exists())
        self.pip.assert_not_called()

    @unittest.skipIf(os.name == 'nt', 'fcntl contention is POSIX-only')
    def test_real_posix_lock_contention_times_out_then_releases(self):
        identity = env.environment_identity()
        lock_path = self.repo.parent / ('aisbench-smoke-' + env.digest(identity) + '.lock')
        child = subprocess.Popen([
            sys.executable, '-c',
            'import fcntl,sys; lock=open(sys.argv[1], "a+b"); '
            'fcntl.flock(lock.fileno(), fcntl.LOCK_EX); '
            'print("locked", flush=True); sys.stdin.read()', str(lock_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'locked')
            with mock.patch.object(env, 'environment_lock', REAL_LOCK), mock.patch.object(
                    env.tempfile, 'gettempdir', return_value=str(self.repo.parent)):
                with mock.patch.dict(env.os.environ, {'AISBENCH_SMOKE_SETUP_LOCK_TIMEOUT': '0.05'}):
                    started = time.monotonic()
                    with self.assertRaisesRegex(TimeoutError, 'timed out after 0.05s'):
                        env.prepare(self.repo)
                    self.assertGreaterEqual(time.monotonic() - started, 0.05)
                    self.assertLess(time.monotonic() - started, 2)
                    self.pip.assert_not_called()
            self.assertTrue(lock_path.exists())
            child.communicate(timeout=5)
            self.assertEqual(child.returncode, 0)
            with mock.patch.object(env.tempfile, 'gettempdir', return_value=str(self.repo.parent)):
                with REAL_LOCK(identity):
                    self.assertTrue(lock_path.exists())
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)

    def test_force_environment_flag(self):
        with mock.patch.object(env, 'prepare') as prepare, mock.patch.object(sys, 'argv', [str(SCRIPT)]):
            with mock.patch.dict(env.os.environ, {'AISBENCH_SMOKE_FORCE_INSTALL': '1'}):
                self.assertEqual(env.main(), 0)
        self.assertTrue(prepare.call_args.args[1])


if __name__ == '__main__':
    unittest.main()
