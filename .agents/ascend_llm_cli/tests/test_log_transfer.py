import base64
import json
from pathlib import Path
import tempfile
import unittest
from deploy_eval_v2.log_transfer import serve,collect
from deploy_eval_v2.task_log import diagnostic
from deploy_eval_v2.event_journal import Journal
from deploy_eval_v2.protocol import ProtocolError


class LogTransferTests(unittest.TestCase):
    def test_collect_ack_and_replay_with_one_central_log(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);remote=root/'remote';local=root/'local'
            remote.mkdir();local.mkdir()
            diagnostic(remote/'worker','日志'*3000)
            original=(remote/'task.jsonl').read_bytes()
            def exchange(action,**kw):
                return serve(remote,action,offset=kw.get('log_offset',0),sha256=kw.get('log_sha256'))
            report=collect(local,'remote:/task',exchange)
            self.assertEqual(report['bytes'],len(original))
            self.assertEqual((remote/'task.jsonl').stat().st_size,0)
            records=[json.loads(line) for line in (local/'task.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(b''.join(base64.b64decode(r['remote_data']) for r in records),original)
            self.assertEqual(Journal(local,'t').events,[])

    def test_disconnect_preserves_source_and_retry_deduplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);remote=root/'remote';local=root/'local'
            remote.mkdir();local.mkdir();diagnostic(remote/'worker','x'*4000)
            original=(remote/'task.jsonl').read_bytes();fail=True
            def exchange(action,**kw):
                if fail and kw.get('log_offset',0)>=1024: raise OSError('disconnect')
                return serve(remote,action,offset=kw.get('log_offset',0),sha256=kw.get('log_sha256'))
            with self.assertRaises(OSError):collect(local,'remote',exchange)
            self.assertEqual((remote/'task.jsonl').read_bytes(),original)
            fail=False;collect(local,'remote',exchange)
            records=[json.loads(line) for line in (local/'task.jsonl').read_text().splitlines()]
            ids=[r['transfer_id'] for r in records]
            self.assertEqual(len(ids),len(set(ids)))

    def test_concurrent_source_append_prevents_discard(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);remote=root/'remote';local=root/'local'
            remote.mkdir();local.mkdir();diagnostic(remote/'worker','before')
            def exchange(action,**kw):
                if action=='log-ack':diagnostic(remote/'worker','after')
                return serve(remote,action,offset=kw.get('log_offset',0),sha256=kw.get('log_sha256'))
            with self.assertRaisesRegex(ProtocolError,'spool changed'):
                collect(local,'remote',exchange)
            self.assertIn('after',(remote/'task.jsonl').read_text())
