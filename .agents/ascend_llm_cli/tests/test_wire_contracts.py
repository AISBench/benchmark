from pathlib import Path
import tempfile
import unittest
from deploy_eval_v2.event_journal import Journal
from deploy_eval_v2.protocol import envelope,ProtocolError
from deploy_eval_v2.wire_contracts import validate_response


class WireContractTests(unittest.TestCase):
    def test_known_boundary_payloads_cannot_omit_evidence_or_claim_wrong_state(self):
        cases=[('deployment.ready','READY',dict(service_id='s',endpoint='http://example.invalid',model='m',evidence_ref='results/deploy.json')),
               ('evaluation.completed','EVALUATED',dict(case_id='c',result_ref='results/eval.json',execution_status='SUCCEEDED',performance_validity='UNVERIFIED')),
               ('operation.failed','FAILED',dict(phase='health',error_code='FAILED',evidence_ref='logs/error.log',cleanup_status='VERIFIED')),
               ('operation.cancelled','CANCELLED',dict(evidence_ref='results/cancel.json',cleanup_status='VERIFIED')),
               ('release.completed','RELEASED',dict(service_id='s',resource_results_ref='results/release.json')),
               ('release.unverified','RECOVERY_REQUIRED',dict(service_id='s',resource_results_ref='results/release.json',reason='unknown'))]
        for kind,state,data in cases:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder:
                journal=Journal(folder,'t')
                event=journal.append(kind,'op',1,state,data)
                validate_response(journal.watch(0,0))
                with self.assertRaises(ProtocolError):
                    journal.append(kind,'op',2,'WRONG',data)
                for key in data:
                    missing={k:v for k,v in data.items() if k!=key}
                    with self.subTest(missing=key),self.assertRaises(ProtocolError):
                        journal.append(kind,'op',2,state,missing)
                self.assertEqual(len(journal.events),1)

    def test_command_identity_and_evidence_traversal_rejected_before_persistence(self):
        with tempfile.TemporaryDirectory() as folder:
            journal=Journal(folder,'t')
            command=dict(request_id='r',request_hash='a'*64,operation_id='other',kind='deploy',after_seq=0)
            with self.assertRaisesRegex(ProtocolError,'identity'):
                journal.append('command.accepted','op',1,'RUNNING',command)
            command.update(operation_id='op',after_seq=9)
            with self.assertRaisesRegex(ProtocolError,'identity'):
                journal.append('command.accepted','op',1,'RUNNING',command)
            with self.assertRaisesRegex(ProtocolError,'evidence path'):
                journal.append('operation.cancelled','op',1,'CANCELLED',dict(cleanup_status='VERIFIED',evidence_ref='results/../outside'))
            self.assertFalse(journal.path.exists())

    def test_invalid_event_never_reaches_journal(self):
        with tempfile.TemporaryDirectory() as folder:
            journal=Journal(folder,'t')
            with self.assertRaisesRegex(ProtocolError,'event schema'):
                journal.append('deployment.ready','../outside',1,'READY',{})
            self.assertFalse((Path(folder)/'task.jsonl').exists())
            self.assertEqual(journal.events,[])

    def test_page_identity_cursor_and_error_structure_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            journal=Journal(folder,'t')
            event=journal.append('future.extension','op',1,'RUNNING',{'unknown':True})
            response=journal.watch(0,0)
            validate_response(response)
            for changed in (dict(response,next_cursor=0),dict(response,task_id='other'),
                            dict(response,events=[event,event]),dict(response,status='WAIT_TIMEOUT')):
                with self.assertRaises(ProtocolError):
                    validate_response(changed)
        with self.assertRaises(ProtocolError):
            validate_response(envelope('t',0,'ERROR'))

    def test_summary_budget_checked_in_serialized_bytes(self):
        response=envelope('t',0,'ERROR')
        response['error']=dict(code='INVALID_INPUT',message='中'*500,retryable=False)
        with self.assertRaisesRegex(ProtocolError,'budget'):
            validate_response(response)
