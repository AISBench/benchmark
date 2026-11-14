import io
import json
import unittest
from unittest.mock import MagicMock,patch
from deploy_eval_v2.remote_client import relay,main
from deploy_eval_v2.protocol import ProtocolError,envelope,encode
from test_cli import Output


class RemoteClientTests(unittest.TestCase):
    def test_stopped_task_artifacts_remain_readable_with_identity(self):
        import tempfile
        from pathlib import Path
        from deploy_eval_v2.durable import atomic_json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            atomic_json(root/'.runtime/identity.json',dict(task_id='t',task_root=str(root)))
            (root/'task.jsonl').write_bytes(b'saved evidence\n')
            request=dict(protocol_version='1.0',task_id='t',kind='read-result',artifact_ref='task.jsonl')
            factory=MagicMock(side_effect=ConnectionRefusedError())
            response=relay(root,request,socket_factory=factory)
            self.assertEqual(response['content'],'saved evidence\n')
            request['task_id']='foreign'
            with self.assertRaises(ProtocolError):
                relay(root,request,socket_factory=factory)

    def setUp(self):
        # Transport mocks exercise framing on Windows without claiming a real
        # Unix socket connection. Linux integration remains a separate gate.
        setting=patch('deploy_eval_v2.remote_client.socket.AF_UNIX',1,create=True)
        setting.start()
        self.addCleanup(setting.stop)
    def factory(self,data):
        factory=MagicMock()
        client=factory.return_value.__enter__.return_value
        client.makefile.return_value=io.BytesIO(data)
        return factory,client

    def test_query_budget_and_identity(self):
        request=dict(protocol_version='1.0',task_id='t',kind='wait-event',after_seq=7,wait_ms=5)
        factory,client=self.factory(encode(envelope('t',7,'WAIT_TIMEOUT'))+b'\n')
        result=relay('/work/t',request,socket_factory=factory)
        self.assertEqual(result['next_cursor'],7)
        self.assertEqual(json.loads(client.sendall.call_args.args[0]),request)
        for raw in (b'x'*8193+b'\n',encode(envelope('other',7))+b'\n',b'{}'):
            factory,_=self.factory(raw)
            with self.assertRaisesRegex(ProtocolError,'TRANSPORT_PROTOCOL_ERROR'):
                relay('/work/t',request,socket_factory=factory)

    def test_socket_failure_is_unknown_not_operation_failure(self):
        factory,client=self.factory(b'')
        client.connect.side_effect=ConnectionRefusedError()
        with self.assertRaisesRegex(ProtocolError,'UNKNOWN_OUTCOME'):
            relay('/work/t',dict(protocol_version='1.0',task_id='t',kind='get-status'),socket_factory=factory)

    def test_main_returns_bounded_error_preserving_cursor_without_traceback(self):
        output=Output();input_=MagicMock()
        input_.buffer=io.BytesIO(encode(dict(protocol_version='1.0',task_id='t',kind='wait-event',after_seq=9))+b'\n')
        with patch('deploy_eval_v2.remote_client.sys.argv',['client','--root','/work/t']),patch('deploy_eval_v2.remote_client.sys.stdin',input_),patch('deploy_eval_v2.remote_client.sys.stdout',output),patch('deploy_eval_v2.remote_client.relay',side_effect=ProtocolError('UNKNOWN_OUTCOME')):
            code=main()
        result=json.loads(output.buffer.getvalue())
        self.assertEqual(code,3)
        self.assertEqual(result['next_cursor'],9)
        self.assertLessEqual(len(output.buffer.getvalue()),8193)
