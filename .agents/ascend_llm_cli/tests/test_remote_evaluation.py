from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch
from deploy_eval_v2.remote_evaluation import RemoteEvaluation,RemoteCleanupUnknown
from deploy_eval_v2.process_guard import Cancelled
from deploy_eval_v2.protocol import ProtocolError,envelope
from deploy_eval_v2.hashes import digest
from deploy_eval_v2.durable import atomic_json
from deploy_eval_v2.tester_worker import dispatch,execute,operation_paths


class RemoteEvaluationTests(unittest.TestCase):
    def adapter(self,transport):
        tester=dict(host='tester',python='/python',work_root='/tests',machine_id='a'*32)
        return RemoteEvaluation('t',tester,'b'*64,'c'*64,'d'*64,transport=transport)

    def invoke(self,adapter,folder,flag=None):
        plan=dict(test_binding='test',cases=[dict(case_id='c',timeout_seconds=60)])
        return adapter(plan,{'testers':{'test':adapter.tester}},'c',Path(folder)/'op'/'benchmark',
                       flag or threading.Event(),verified_local_host='tester')

    def test_result_is_bound_to_operation_and_digest(self):
        result=dict(execution_status='SUCCEEDED',evidence_status='PARTIAL',performance_validity='UNVERIFIED')
        transport=Mock(return_value=dict(status='EVENTS',operation_id='op',result=result,
            result_sha256=digest(result),evidence=dict(host='tester',path='/tests/t/results/tester/op')))
        with tempfile.TemporaryDirectory() as folder:
            value=self.invoke(self.adapter(transport),folder)
            self.assertEqual(value['remote_evidence']['host'],'tester')
            self.assertTrue((Path(folder)/'op/benchmark/remote-result.json').exists())
        self.assertEqual(transport.call_count,1)

    def test_lost_response_stops_exact_operation_never_resubmits(self):
        transport=Mock(side_effect=[ProtocolError('UNKNOWN_OUTCOME'),dict(status='EVENTS',cleanup_status='VERIFIED')])
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ProtocolError):
                self.invoke(self.adapter(transport),folder)
        requests=[call.args[1] for call in transport.call_args_list]
        self.assertEqual([r['action'] for r in requests],['run','cancel'])
        self.assertEqual({r['operation_id'] for r in requests},{'op'})

    def test_unknown_stop_does_not_claim_client_cleanup(self):
        transport=Mock(side_effect=[ProtocolError('UNKNOWN_OUTCOME'),ProtocolError('SSH_UNAVAILABLE')])
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RemoteCleanupUnknown):
                self.invoke(self.adapter(transport),folder)

    def test_cancel_is_reported_only_after_verified_stop(self):
        flag=threading.Event();flag.set()
        transport=Mock(side_effect=[ProtocolError('UNKNOWN_OUTCOME'),dict(status='EVENTS',cleanup_status='VERIFIED')])
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(Cancelled):
                self.invoke(self.adapter(transport),folder,flag)
        self.assertNotIn('cancelled',transport.call_args.kwargs)


class TesterWorkerTests(unittest.TestCase):
    def request(self,action='run'):
        return dict(protocol_version='1.0',task_id='t',operation_id='op',case_id='c',
                    runtime_sha256='a'*64,bundle_sha256='b'*64,token='c'*64,action=action)

    def inputs(self):
        return {},{},dict(python='/python',host='tester'),dict(timeout_seconds=60)

    def test_response_loss_never_launches_second_load(self):
        with tempfile.TemporaryDirectory() as folder,\
             patch('deploy_eval_v2.tester_worker.verified_inputs',return_value=self.inputs()):
            root=Path(folder);logged=Mock(side_effect=ProtocolError('UNKNOWN_OUTCOME'))
            self.addCleanup(patch.stopall)
            patch('deploy_eval_v2.tester_worker.spawn_owned',logged).start()
            with self.assertRaises(ProtocolError):
                dispatch(root,self.request(),logged=logged)
            with self.assertRaisesRegex(ProtocolError,'already accepted'):
                dispatch(root,self.request(),logged=logged)
            self.assertEqual(logged.call_count,1)
            argv=logged.call_args.args[0]
            self.assertEqual(argv[0],'/python')
            self.assertIn('--execute',argv)

    def test_cancel_before_start_leaves_tombstone_and_blocks_worker(self):
        with tempfile.TemporaryDirectory() as folder,\
             patch('deploy_eval_v2.tester_worker.verified_inputs',return_value=self.inputs()):
            root=Path(folder);runner=Mock(return_value=(5,b'',b''));logged=Mock()
            response=dispatch(root,self.request('cancel'),runner=runner,logged=logged)
            self.assertEqual(response['cleanup_status'],'VERIFIED')
            with self.assertRaisesRegex(ProtocolError,'cancelled'):
                dispatch(root,self.request(),runner=runner,logged=logged)
            evaluator=Mock()
            with self.assertRaisesRegex(ProtocolError,'cancelled'):
                execute(root,self.request(),evaluator=evaluator)
            evaluator.assert_not_called();logged.assert_not_called()

    def test_completed_response_replayed_without_process_start(self):
        with tempfile.TemporaryDirectory() as folder,\
             patch('deploy_eval_v2.tester_worker.verified_inputs',return_value=self.inputs()):
            root=Path(folder);request=self.request();path,identity,unit=operation_paths(root,request)
            atomic_json(path/'request.json',request)
            response=envelope('t',0);response['operation_id']='op'
            atomic_json(path/'response.json',response)
            logged=Mock()
            self.assertEqual(dispatch(root,request,logged=logged),response)
            logged.assert_not_called()
