import unittest
from deploy_eval_v2.result_contract import evaluation_result
from deploy_eval_v2.protocol import ProtocolError


class ResultContractTests(unittest.TestCase):
    def bind(self,value):
        return evaluation_result(value,task_id='t',operation_id='op',service_id='s',case_id='c')

    def test_coordinator_binds_identity_without_overwriting_conflicts(self):
        result=dict(execution_status='SUCCEEDED',evidence_status='PARTIAL',performance_validity='UNVERIFIED')
        self.assertEqual(self.bind(result)['operation_id'],'op')
        self.assertNotIn('operation_id',result)
        for key in ('task_id','operation_id','service_id','case_id'):
            with self.subTest(key=key),self.assertRaisesRegex(ProtocolError,'identity mismatch'):
                self.bind(dict(result,**{key:'foreign'}))

    def test_invalid_states_and_unsupported_performance_claims_rejected(self):
        for value in [dict(execution_status='OK',evidence_status='COMPLETE',performance_validity='VERIFIED'),
                      dict(execution_status='SUCCEEDED',evidence_status='PARTIAL',performance_validity='VERIFIED'),
                      dict(execution_status='SUCCEEDED')]:
            with self.subTest(value=value),self.assertRaises(ProtocolError):
                self.bind(value)
