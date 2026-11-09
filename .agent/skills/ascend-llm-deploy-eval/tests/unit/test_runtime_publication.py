import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import publish_runtime


class RuntimePublicationTests(unittest.TestCase):
    def test_minimal_package_has_import_closure_and_verified_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'proxy.py'
            source.write_text('print("explicit input")')
            manifest, payload = publish_runtime.build('root@example', '/work/user', 'case1', [source])
            self.assertEqual(manifest['runtime_root'], '/work/user/case1.runtime')
            self.assertFalse(any('tests/' in k or 'stage_runtime' in k for k in payload['files']))
            self.assertEqual(manifest['input_paths'][str(source.resolve())], '/work/user/case1.runtime/inputs/proxy.py')
            root = Path(directory) / 'runtime'
            for name, value in payload['files'].items():
                content = base64.b64decode(value)
                self.assertEqual(hashlib.sha256(content).hexdigest(), payload['hashes'][name])
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            for module in ('run', 'review_job', 'generate_gsm8k', 'watch_events', 'watch_logs'):
                result = subprocess.run([sys.executable, '-B', str(root / 'scripts' / (module + '.py')), '--help'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_escape_host_and_duplicate_inputs_before_transport(self):
        for host, root, identity in [('root@host;bad', '/work', 'x'), ('host', '/work/../bad', 'x'), ('host', '/', 'x'), ('host', '/work', '../x')]:
            with self.assertRaises(ValueError):
                publish_runtime.build(host, root, identity)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'proxy.py'
            source.write_text('pass')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                publish_runtime.build('host', '/work', 'x', [source, source])

    def test_failed_publish_is_not_retried_or_executed(self):
        calls = []
        def runner(argv, **kwargs):
            calls.append((argv, json.loads(kwargs['input'])))
            return subprocess.CompletedProcess(argv, 1, '', 'existing content differs')
        with patch.object(publish_runtime, 'transport_argv', side_effect=lambda host, argv: ['ssh', host, *argv]):
            with self.assertRaisesRegex(RuntimeError, 'existing content differs'):
                publish_runtime.publish('host', '/work', 'case1', runner=runner)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]['root'], '/work/case1.runtime')

    def test_plan_only_never_connects_and_receipt_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / 'receipt.json'
            argv = ['publish_runtime.py', '--host', 'host', '--work-root', '/work',
                    '--task-id', 'case1', '--receipt', str(receipt), '--plan-only']
            with patch.object(sys, 'argv', argv), patch.object(publish_runtime, 'publish') as remote:
                publish_runtime.main()
                remote.assert_not_called()
                self.assertEqual(json.loads(receipt.read_text())['status'], 'PLANNED')
                before = receipt.read_bytes()
                with self.assertRaises(FileExistsError):
                    publish_runtime.main()
                self.assertEqual(receipt.read_bytes(), before)

    def test_failed_publication_leaves_local_failure_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / 'receipt.json'
            argv = ['publish_runtime.py', '--host', 'host', '--work-root', '/work',
                    '--task-id', 'case1', '--receipt', str(receipt)]
            with patch.object(sys, 'argv', argv), patch.object(publish_runtime, 'publish', side_effect=RuntimeError('interrupted transfer')):
                with self.assertRaisesRegex(RuntimeError, 'interrupted transfer'):
                    publish_runtime.main()
            record = json.loads(receipt.read_text())
            self.assertEqual(record['status'], 'FAILED')
            self.assertEqual(record['task_id'], 'case1')

    def test_interruption_keeps_its_identity_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / 'receipt.json'
            argv = ['publish_runtime.py', '--host', 'host', '--work-root', '/work',
                    '--task-id', 'case1', '--receipt', str(receipt)]
            with patch.object(sys, 'argv', argv), patch.object(publish_runtime, 'publish', side_effect=KeyboardInterrupt('user stop')) as remote:
                with self.assertRaises(KeyboardInterrupt):
                    publish_runtime.main()
                remote.assert_called_once()
            record = json.loads(receipt.read_text())
            self.assertEqual(record['status'], 'INTERRUPTED')
            self.assertEqual(record['error'], 'user stop')

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux receiver uses flock')
    def test_receiver_idempotence_conflict_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / 'work'
            manifest, payload = publish_runtime.build('host', str(work), 'case1')
            def receive(value):
                return subprocess.run([sys.executable, '-c', publish_runtime.RECEIVER], input=json.dumps(value), capture_output=True, text=True)
            self.assertEqual(receive(payload).returncode, 0)
            self.assertEqual(receive(payload).returncode, 0)
            (Path(manifest['runtime_root']) / 'requirements.txt').write_text('changed')
            self.assertNotEqual(receive(payload).returncode, 0)
            outside = Path(directory) / 'outside'
            outside.mkdir()
            (work / 'case2.runtime').symlink_to(outside, target_is_directory=True)
            _, escape = publish_runtime.build('host', str(work), 'case2')
            self.assertNotEqual(receive(escape).returncode, 0)
            self.assertEqual(list(outside.iterdir()), [])
