from copy import deepcopy
import unittest
from deploy_eval_v2.service_contract import validate_service
from deploy_eval_v2.protocol import ProtocolError


class ServiceContractTests(unittest.TestCase):
    def fixture(self):
        return dict(schema_version='1.0',task_id='t',service_id='s',ownership='owned',
                    endpoint='http://localhost:8000',model='m',bundle_sha256='a'*64,
                    runtime_sha256='b'*64,health_time='2026-11-12T00:00:00Z',
                    health_evidence_ref='results/op/stage-result.json',
                    policy=dict(idle_timeout_seconds=900,on_idle='keep_service',on_failure='release_owned'),
                    deployment_summary=dict(container_groups=1),
                    containers={'node/model':dict(container_id='c'*64)})

    def test_complete_handoff(self):
        validate_service(self.fixture())

    def test_invalid_endpoint_and_evidence_paths(self):
        cases={
            'endpoint':['file:///tmp/x','http://user:secret@localhost', 'http://localhost:0',
                        'http://localhost:99999','http://localhost/?token=x','http://localhost/#x'],
            'health_evidence_ref':['results/../x','results//x','results/./x','results/x:stream',
                                   'results/x\x00','results/','results/\\x'],
            'health_time':['yesterday','2026-11-12T00:00:00+08:00'],
        }
        for key,values in cases.items():
            for value in values:
                with self.subTest(key=key,value=value):
                    service=self.fixture();service[key]=value
                    with self.assertRaises(ProtocolError):
                        validate_service(service)

    def test_duplicate_or_incomplete_container_inventory(self):
        service=self.fixture()
        service['containers']['other/model']=deepcopy(service['containers']['node/model'])
        with self.assertRaises(ProtocolError):
            validate_service(service)
        service['deployment_summary']['container_groups']=2
        with self.assertRaises(ProtocolError):
            validate_service(service)
        service['containers']['other/model']['container_id']='d'*64
        validate_service(service)
