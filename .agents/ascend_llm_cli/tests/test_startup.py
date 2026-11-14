from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from deploy_eval_v2.startup import start
from deploy_eval_v2.publication_cli import start_task
from deploy_eval_v2.protocol import ProtocolError,envelope


def request():
    return dict(protocol_version='1.0',task_id='t',runtime_sha256='a'*64,
                bundle_sha256='b'*64,nonce='c'*64,python='/venv/bin/python')


class StartupTests(unittest.TestCase):
    def test_receipt_prevents_second_start_after_unknown_reply(self):
        with tempfile.TemporaryDirectory() as folder,patch('deploy_eval_v2.startup.validate_identity'):
            starter=Mock(side_effect=[ProtocolError('UNKNOWN_OUTCOME'),dict(identity_verified=True,pid=123)])
            with self.assertRaises(ProtocolError):
                start(folder,request(),starter=starter)
            response=start(folder,request(),starter=starter)
            self.assertTrue(response['started'])
            self.assertFalse(response['stage_submitted'])
            self.assertTrue(starter.call_args_list[0].kwargs['allow_start'])
            self.assertFalse(starter.call_args_list[1].kwargs['allow_start'])

    def test_changed_nonce_cannot_replace_launch_identity(self):
        with tempfile.TemporaryDirectory() as folder,patch('deploy_eval_v2.startup.validate_identity'):
            starter=Mock(return_value=dict(identity_verified=True))
            start(folder,request(),starter=starter)
            with self.assertRaisesRegex(ProtocolError,'REQUEST_CONFLICT'):
                start(folder,dict(request(),nonce='d'*64),starter=starter)
            self.assertEqual(starter.call_count,1)

    def test_control_start_transmits_persisted_nonce_without_stage_request(self):
        transport=Mock(return_value=envelope('t',0))
        config={'tasks':{'t':dict(host='host.invalid',root='/work/t',allowed_root='/work')}}
        start_task(dict(task_ref='t',runtime_sha256='a'*64,bundle_sha256='b'*64,nonce='c'*64),config,transport=transport)
        sent=transport.call_args.args[1]
        self.assertEqual(sent['nonce'],'c'*64)
        self.assertNotIn('kind',sent)
        self.assertEqual(sent['python'],'/usr/bin/python3')
