"""Dependency-free tests: no pip installation or AISBench imports."""

from contextlib import ExitStack, nullcontext, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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
        stack.enter_context(redirect_stdout(io.StringIO()))
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

    def dist(self, name, version='1'):
        texts = {'METADATA': 'Name: {}\nVersion: {}\n'.format(name, version)}
        if name == 'ais_bench_benchmark':
            texts['direct_url.json'] = json.dumps(
                {'url': self.repo.as_uri(), 'dir_info': {'editable': True}})
        return SimpleNamespace(metadata={'Name': name, 'Version': version},
                               read_text=texts.get, locate_file=lambda _: self.repo / 'site')

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

    def test_source_egg_info_duplicate_does_not_hide_editable_install(self):
        source = self.dist('ais_bench_benchmark')
        source.read_text = lambda key: 'Name: ais_bench_benchmark\nVersion: 1' if key == 'PKG-INFO' else None
        self.distributions.append(source)
        env.prepare(self.repo)
        self.pip.reset_mock()
        self.assertTrue(env.prepare(self.repo))
        self.pip.assert_not_called()

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

    def test_atomic_write_failure_no_stamp_or_temporary_file(self):
        with mock.patch.object(env.os, 'replace', side_effect=OSError('write failed')):
            with self.assertRaises(OSError):
                env.prepare(self.repo)
        self.assertFalse(self.stamp.exists())
        self.assertEqual(list(self.stamp.parent.iterdir()), [])

    def test_remote_include_installs_but_disables_cache(self):
        self.put('smoke_tests/requirements.txt', '-r https://example.invalid/requirements.txt\n')
        env.prepare(self.repo)
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

    def test_force_environment_flag(self):
        with mock.patch.object(env, 'prepare') as prepare, mock.patch.object(sys, 'argv', [str(SCRIPT)]):
            with mock.patch.dict(env.os.environ, {'AISBENCH_SMOKE_FORCE_INSTALL': '1'}):
                self.assertEqual(env.main(), 0)
        self.assertTrue(prepare.call_args.args[1])


if __name__ == '__main__':
    unittest.main()
