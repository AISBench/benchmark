from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
from deploy_eval_v2.node_worker import dispatch
from deploy_eval_v2.ssh_node import SSHNode
from deploy_eval_v2.protocol import ProtocolError
from deploy_eval_v2.remote_guard import HostGuard


class NodeWorkerTests(unittest.TestCase):
    def test_claim_uses_verified_group_and_freezes_execution_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'task'
            group=dict(node='n',root=str(root),host='host',npu_ids=[0],ports=[18000])
            values={'plan.json':dict(task_id='t',intent='deploy'),
                    'compiled.json':dict(topology=dict(groups={'g':group}))}
            request=dict(protocol_version='1.0',task_id='t',node='n',action='claim',command_id='claim1',
                         runtime_sha256='a'*64,bundle_sha256='b'*64,token='c'*64,timeout_seconds=10,group='g')
            guardroot=Path(folder)/'guard'
            with patch('deploy_eval_v2.node_worker.verify_runtime') as runtime,patch('deploy_eval_v2.node_worker.verify_frozen',return_value=values) as frozen:
                self.assertTrue(dispatch(root,request,guard_root=guardroot,node_factory=Mock())['result']['claimed'])
                runtime.assert_called_with(root/'runtime','a'*64)
                frozen.assert_called_with(root/'inputs','b'*64,'a'*64)
                with self.assertRaisesRegex(ProtocolError,'identity changed'):
                    dispatch(root,dict(request,token='d'*64),guard_root=guardroot,node_factory=Mock())
            with self.assertRaisesRegex(ProtocolError,'RESOURCE_BUSY'):
                HostGuard(guardroot,'other','d'*64).claim(['npu:0'])

    def test_arbitrary_shell_payload_is_rejected_before_package_reads(self):
        with patch('deploy_eval_v2.node_worker.verify_runtime') as verify:
            with self.assertRaisesRegex(ProtocolError,'fields'):
                dispatch('/unused',dict(action='shell',command='rm -rf /'))
            verify.assert_not_called()

    def test_ssh_adapter_only_sends_frozen_group_identity(self):
        group=dict(node='n',root='/task/t',host='user@host',npu_ids=[0],ports=[18000],commands=[])
        transport=Mock(return_value=dict(status='EVENTS',command_id='create1',result=dict(container_id='d'*64)))
        node=SSHNode('t','n',dict(topology=dict(groups={'g':group})),'a'*64,'b'*64,'c'*64,transport=transport)
        import threading,time
        result=node.create('create1',group,time.monotonic()+10,threading.Event())
        self.assertEqual(result['container_id'],'d'*64)
        sent=transport.call_args.args[1]
        self.assertEqual(sent['group'],'g')
        self.assertNotIn('argv',sent)
        self.assertNotIn('image',sent)
        self.assertIn('-T',transport.call_args.args[0])
