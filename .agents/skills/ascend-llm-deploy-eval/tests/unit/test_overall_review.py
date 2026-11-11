"""Regression coverage for the overall review against the current entrypoint."""
import base64
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

from test_compact_entry import inputs
import run as entry
import watch_events
import watch_logs
from bundle_distribution import PROGRAM
from container_baseline import docker_run_args
from task_model import normalize, validate
from validate_case import validate as validate_result


class OverallReviewTests(unittest.TestCase):
    def test_privileged_requires_explicit_choice(self):
        spec, bindings = inputs()
        for value in (True, False):
            spec['deployments'][0]['runtime']['container']['privileged'] = value
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)/'job'
                entry.generate(spec, bindings, root)
                text = next((root/'scripts').glob('*/*-container.sh')).read_text()
                self.assertIn('--privileged='+str(value).lower(), text)
        del spec['deployments'][0]['runtime']['container']['privileged']
        self.assertTrue(any('privileged' in error for error in validate(normalize(spec))))
        with self.assertRaisesRegex(ValueError, 'Explicit privileged'):
            docker_run_args({})

    def test_final_shell_preserves_boolean_json_and_argument_boundaries(self):
        spec, bindings = inputs()
        dep = spec['deployments'][0]
        inst = next(i for i in dep['instances'] if i['role'] != 'proxy')
        inst['launch']['args'].update({'enable-prefix-caching': False,
            'test-structured': {'name': "space ' quote", 'nested': [True, False]},
            'test-string': "a b; $(touch NEVER)"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'job'
            entry.generate(spec, bindings, root)
            script = root/'scripts'/dep['id']/('node_a-'+inst['id']+'.sh')
            line = next(x for x in script.read_text().splitlines() if x.startswith('exec vllm '))
            argv = shlex.split(line)[1:]
            self.assertIn('--no-enable-prefix-caching', argv)
            self.assertEqual(json.loads(argv[argv.index('--test-structured')+1]),
                             inst['launch']['args']['test-structured'])
            self.assertEqual(argv[argv.index('--test-string')+1], "a b; $(touch NEVER)")
            config = next((root/'scripts/tests').glob('*.config.py')).read_text()
            self.assertNotIn('work_dir =', config)
            test_script = next((root/'scripts/tests').glob('*.sh')).read_text()
            self.assertIn('$OUT/outputs', test_script)
            self.assertIn('TRY=${1:-try01}', test_script)

    def test_current_logs_drain_backlog_and_handle_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'results/case-try01/aisbench.log'
            path.parent.mkdir(parents=True)
            payload = 'x'*200000+'END\n'
            path.write_text(payload)
            cursors = {}
            with redirect_stdout(io.StringIO()) as output:
                watch_logs.poll(root, 'test', cursors)
            self.assertTrue(output.getvalue().replace('\r\n', '\n').endswith(payload))
            with redirect_stdout(io.StringIO()) as output:
                watch_logs.poll(root, 'test', cursors)
            self.assertEqual(output.getvalue(), '')
            path.write_text('rotated')
            with redirect_stdout(io.StringIO()) as output:
                watch_logs.poll(root, 'test', cursors)
            self.assertIn('rotated', output.getvalue())
            service = root/'logs/dep-run01/node-model.log'
            service.parent.mkdir(parents=True)
            service.write_text('model')
            self.assertIn(service, watch_logs.discover(root, 'service'))

    def test_timeline_cursor_and_local_timezone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = {'time': 1700000000, 'kind': 'case', 'id': 'a', 'status': 'PASSED'}
            (root/'status.json').write_text(json.dumps({'timeline': [record]}))
            cursor, events = watch_events.poll(root)
            self.assertEqual(events, [record])
            self.assertEqual(watch_events.poll(root, cursor), (1, []))
            self.assertIn('case/a PASSED', watch_events.format_event(record))
            from datetime import datetime
            self.assertIn(datetime.fromtimestamp(1700000000).astimezone().strftime('%z'), watch_events.format_event(record))
            (root/'status.json').write_text('{"timeline":[]}')
            with self.assertRaisesRegex(ValueError, 'truncated'):
                watch_events.poll(root, cursor)
            (root/'status.json').write_text('{"status":"RUNNING"}')
            from task_events import emit
            emit(root, 'legacy', 'done')
            legacy_cursor, legacy_events = watch_events.poll(root)
            self.assertEqual(len(legacy_events), 1)
            self.assertEqual(watch_events.poll(root, legacy_cursor)[1], [])

    def test_zero_output_and_aggregate_reset_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            details = root/'outputs/time/performances/model/data_details.jsonl'
            details.parent.mkdir(parents=True)
            row = dict(id=0, success=True, input_tokens=1, output_tokens=0)
            details.write_text(json.dumps(row))
            log = root/'outputs/time/logs/worker.log'
            log.parent.mkdir(); log.write_text('done')
            (root/'execution.json').write_text('{"returncode":0}')
            case = dict(id='c', formal_requests=1, output_tokens=16, reset_cache=True)
            # Misleading raw ordering must never override aggregate failure.
            (root/'reset-1-9.json').write_text('{"success":true}')
            (root/'cache-reset.json').write_text('{"status":"FAILED","instances":[]}')
            with self.assertRaisesRegex(ValueError, 'reset'):
                validate_result(root, case)
            (root/'cache-reset.json').write_text('{"status":"PASSED","instances":[{"status":"PASSED"}]}')
            self.assertEqual(validate_result(root, case)['formal']['zero_output_requests'], 1)
            row['success'] = False; details.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'failed'):
                validate_result(root, case)
            row['success'] = True; row['output_tokens'] = -1; details.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'token'):
                validate_result(root, case)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux receiver publication')
    def test_distribution_failures_and_competing_publishers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'job'
            def payload(data):
                return json.dumps({'root': str(root), 'files': {'a': base64.b64encode(data).decode()},
                                   'hashes': {'a': hashlib.sha256(data).hexdigest()}})
            def execute(program, data):
                return subprocess.run([sys.executable, '-c', program], input=data,
                                      capture_output=True, text=True, timeout=15)
            corrupted = json.loads(payload(b'valid')); corrupted['files']['a'] = base64.b64encode(b'wrong').decode()
            self.assertNotEqual(execute(PROGRAM, json.dumps(corrupted)).returncode, 0)
            self.assertFalse(root.exists())
            interrupted = PROGRAM.replace('            staged.rename(root)', '            __import__("os")._exit(9)')
            self.assertEqual(execute(interrupted, payload(b'valid')).returncode, 9)
            self.assertFalse(root.exists())
            processes = [subprocess.Popen([sys.executable, '-c', PROGRAM], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
            for process, data in zip(processes, (b'one', b'two')):
                process.stdin.write(payload(data)); process.stdin.close(); process.stdin = None
            for process in processes: process.communicate(timeout=15)
            self.assertEqual(sorted(p.returncode for p in processes), [0, 1])
            winner = (root/'a').read_bytes()
            self.assertIn(winner, (b'one', b'two'))
            self.assertEqual(execute(PROGRAM, payload(winner)).returncode, 0)
