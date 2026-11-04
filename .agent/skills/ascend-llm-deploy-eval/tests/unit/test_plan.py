import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import plan_run as planner
from test_serial_task import fixture
ROOT=planner.PACKAGE

class PlanTests(unittest.TestCase):
    def setUp(self): self.spec=planner.normalize(fixture()); self.spec['cases'][0]['formal_requests']=64
    def test_defaults_and_explicit_request_count(self):
        self.spec.pop('execution')
        normalized=planner.normalize(self.spec)
        self.assertEqual(planner.validate(normalized),[])
        self.assertEqual(normalized['execution']['max_retries'],3)
        self.assertEqual(normalized['execution']['smoke'],'default')
        self.assertEqual(normalized['cases'][0]['formal_requests'],64)
    def test_reject_conflicts(self):
        mutations=[lambda s:s.update(schema_version='2.0'),
            lambda s:s['cases'][0].update(id='../bad'),lambda s:s.update(target={'container':'other'}),
            lambda s:s['execution'].update(max_retries=4),lambda s:s['cases'][0].update(formal_requests=0),
            lambda s:s['deployments'][0].update(recipe='../../elsewhere'),lambda s:s.update(unresolved=['missing input'])]
        for mutate in mutations:
            s=copy.deepcopy(self.spec); mutate(s)
            self.assertTrue(planner.validate(s))
    def test_idempotent_and_conflicting_plan(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            planner.plan(self.spec,root); before=(root/'common/plan.json').read_bytes()
            planner.plan(self.spec,root); self.assertEqual(before,(root/'common/plan.json').read_bytes())
            self.spec['cases'][0]['formal_requests']=65
            with self.assertRaises(ValueError): planner.plan(self.spec,root)
            self.assertEqual(before,(root/'common/plan.json').read_bytes())
    def test_deployment_not_bound_to_image_or_model(self):
        self.spec['deployments'][0]['runtime']['image']='candidate:test'
        self.spec['deployments'][0]['model']['served_name']='alternate'
        self.assertEqual(planner.validate(self.spec),[])
    def test_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json'; p.write_text('{"a":1,"a":2}')
            with self.assertRaises(ValueError):planner.load_json(p)
