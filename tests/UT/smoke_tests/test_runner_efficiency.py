"""Regression tests for smoke orchestration overhead (no model dependencies)."""
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import Future
from unittest.mock import Mock, patch


FRAMEWORK = Path(__file__).resolve().parents[3] / 'smoke_tests' / 'framework'
with patch.dict(sys.modules), patch.object(sys, 'path', [str(FRAMEWORK)] + sys.path):
    spec = importlib.util.spec_from_file_location('smoke_runner_efficiency', FRAMEWORK / 'run.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)


class TestRunnerEfficiency(unittest.TestCase):
    def test_parallel_limit_preserves_default_and_validates_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runner.get_max_parallel(), 50)
        for value in ('1', '7'):
            with patch.dict(os.environ, {'SMOKE_MAX_PARALLEL': value}):
                self.assertEqual(runner.get_max_parallel(), int(value))
        for value in ('0', '-1', 'abc'):
            with patch.dict(os.environ, {'SMOKE_MAX_PARALLEL': value}):
                with self.assertRaisesRegex(ValueError, 'positive integer'):
                    runner.get_max_parallel()

    def test_case_log_is_streamed_in_bounded_chunks_without_truncation(self):
        content = 'smoke log 测试\n' * 20000
        with tempfile.TemporaryDirectory() as root:
            log_path = Path(root) / 'case.log'
            log_path.write_text(content)
            with patch.object(runner, 'log') as log:
                runner.append_case_log(log_path)
        chunks = [call.args[0] for call in log.call_args_list[:-1]]
        self.assertEqual(''.join(chunks), content)
        self.assertTrue(all(len(chunk) <= 64 * 1024 for chunk in chunks))
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(call.kwargs['end'] == '' for call in log.call_args_list[:-1]))

    def test_case_log_preserves_non_utf8_locale_text(self):
        content = 'case 测试 café\n'
        def locale_open(*args, **kwargs):
            return io.TextIOWrapper(io.BytesIO(content.encode('cp936')),
                                    encoding=kwargs.get('encoding', 'cp936'),
                                    errors=kwargs.get('errors', 'strict'))
        with patch('builtins.open', side_effect=locale_open), patch.object(runner, 'log') as log:
            runner.append_case_log('case.log')
        self.assertEqual(log.call_args_list[0].args[0], content)

    def run_suite(self, interrupted=False):
        config = dict(case_name='one', case_type='benchmark', rank_size=3,
                      case_tags={'short'}, yml_path='/test-case/group/one/case.yml')
        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor._processes = {123: Mock(pid=123)}
        future = Future()
        if interrupted:
            future.set_exception(KeyboardInterrupt())
        else:
            future.set_result(('benchmark', 'one', dict(success=True, metric_key=['one'])))
        executor.submit.return_value = future
        parent, child = Mock(), Mock()
        parent.children.return_value = [child]
        results = {}
        with patch.dict(os.environ, {'RESOURCE_PATH': '/resource', 'WORKSPACE_PATH': '/workspace',
                                     'SMOKE_MAX_PARALLEL': '7'}), \
                patch.object(runner, 'ProcessPoolExecutor', return_value=executor) as pool, \
                patch.object(runner, 'TaskScheduler'), patch.object(runner, 'log'), \
                patch.object(runner, 'tqdm'), patch.object(runner.sys, 'stderr', io.StringIO()), \
                patch.object(runner.time, 'perf_counter', side_effect=[0, 0, 10, 10]), \
                patch.object(runner.time, 'sleep') as sleep, \
                patch.object(runner.threading, 'Thread') as thread, \
                patch.object(runner.psutil, 'Process', return_value=parent), \
                patch.object(runner.psutil, 'wait_procs', return_value=([], [child])) as wait:
            record, was_interrupted = runner.execute_test_cases([config], results)
        return record, was_interrupted, pool, sleep, thread, parent, child, wait

    def test_normal_suite_has_no_cooldown_spinner_or_startup_pid_polling(self):
        record, interrupted, pool, sleep, thread, parent, _, _ = self.run_suite()
        self.assertFalse(interrupted)
        self.assertEqual(record['finished_cases']['success'], ['one'])
        pool.assert_called_once_with(max_workers=1)
        sleep.assert_not_called()
        thread.return_value.start.assert_not_called()
        parent.terminate.assert_not_called()  # no stale PID cleanup after normal pool shutdown

    def test_cancellation_waits_for_descendants_collected_before_reparenting(self):
        _, interrupted, _, _, _, parent, child, wait = self.run_suite(interrupted=True)
        self.assertTrue(interrupted)
        parent.children.assert_called_once_with(recursive=True)
        wait.assert_called_once_with([parent, child], timeout=1.0)
        child.terminate.assert_called_once()
        child.kill.assert_called_once()


if __name__ == '__main__':
    unittest.main()
