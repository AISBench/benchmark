import copy
import unittest
from deploy_eval_v2.plan_model import normalize_external_plan, validate_eval_bindings
from deploy_eval_v2.protocol import ProtocolError
from deploy_eval_v2.compiler import review_inputs, verify_review


def external_plan():
    return dict(schema_version='1.0', task_id='t', intent='external_eval',
                service=dict(service_id='s', endpoint='https://example.invalid:443', model='m',
                             ownership='external', deployment_summary=None,
                             permissions=dict(reset_cache=False, restart=False, release=False)),
                test_binding='tester', cases=[dict(case_id='c', concurrency=8, input_tokens=128,
                    output_tokens=16, steady_state=False, num_warmups=1, reset_cache=False,
                    dataset_id='synthetic', timeout_seconds=60)],
                policy=dict(idle_timeout_seconds=900, on_idle='keep_external_service',
                            on_failure='stop_owned_clients'), unresolved=[])


def eval_bindings():
    return dict(schema_version='1.0', task_id='t', testers={'tester': dict(
        host='example.invalid', work_root='/work/tasks', python='/venv/bin/python',
        aisbench='/venv/bin/ais_bench', model_path='/models/tokenizer',
        datasets={'synthetic': {'type': 'aisbench_synthetic'}})})


class PlanTests(unittest.TestCase):
    def test_freeze_requires_reviewed_defaults_and_resolved_plan(self):
        plan = external_plan()
        bindings, runtime = eval_bindings(), 'a'*64
        review = dict(schema_version='1.0', status='APPROVED',
                      inputs=review_inputs(plan, bindings, runtime), unresolved=[],
                      evidence_refs=['review/input.json'])
        with self.assertRaisesRegex(ProtocolError, 'normalize and resolve'):
            verify_review(plan, bindings, review, runtime)
        plan, _ = normalize_external_plan(plan)
        review['inputs'] = review_inputs(plan, bindings, runtime)
        verify_review(plan, bindings, review, runtime)
        plan['unresolved'] = ['missing benchmark version']
        review['inputs'] = review_inputs(plan, bindings, runtime)
        with self.assertRaisesRegex(ProtocolError, 'normalize and resolve'):
            verify_review(plan, bindings, review, runtime)

    def test_defaults_explicit_values_and_no_mutation(self):
        plan = external_plan()
        original = copy.deepcopy(plan)
        normalized, provenance = normalize_external_plan(plan)
        self.assertEqual(plan, original)
        self.assertEqual(normalized['cases'][0]['formal_requests'], 32)
        self.assertEqual(provenance[0]['source'], 'concurrency*4')
        plan['cases'][0]['steady_state'] = True
        self.assertEqual(normalize_external_plan(plan)[0]['cases'][0]['formal_requests'], 80)
        plan['cases'][0]['formal_requests'] = 13
        normalized, provenance = normalize_external_plan(plan)
        self.assertEqual(normalized['cases'][0]['formal_requests'], 13)
        self.assertEqual(provenance, [])

    def test_permissions_and_duplicate_ids(self):
        plan = external_plan()
        plan['cases'][0]['reset_cache'] = True
        with self.assertRaisesRegex(ProtocolError, 'not authorized'):
            normalize_external_plan(plan)
        plan = external_plan()
        plan['service']['permissions']['release'] = True
        with self.assertRaises(ProtocolError):
            normalize_external_plan(plan)
        plan = external_plan()
        plan['cases'].append(copy.deepcopy(plan['cases'][0]))
        with self.assertRaisesRegex(ProtocolError, 'duplicate'):
            normalize_external_plan(plan)


class BindingTests(unittest.TestCase):
    def test_valid_bindings_do_not_modify_inputs(self):
        plan, bindings = external_plan(), eval_bindings()
        original = copy.deepcopy(bindings)
        validate_eval_bindings(plan, bindings)
        self.assertEqual(bindings, original)

    def test_reference_mismatches(self):
        for mutation in ('task', 'tester', 'dataset'):
            plan, bindings = external_plan(), eval_bindings()
            if mutation == 'task':
                bindings['task_id'] = 'other'
            elif mutation == 'tester':
                plan['test_binding'] = 'missing'
            else:
                plan['cases'][0]['dataset_id'] = 'missing'
            with self.assertRaises(ProtocolError):
                validate_eval_bindings(plan, bindings)

    def test_linux_paths_and_host(self):
        for key, value in [('host', '-oProxyCommand=bad'), ('work_root', '/'),
                           ('model_path', '/models/../secret'), ('python', 'C:\\python.exe'),
                           ('aisbench', '/venv/bin/ais\nbench')]:
            bindings = eval_bindings()
            bindings['testers']['tester'][key] = value
            with self.assertRaises(ProtocolError):
                validate_eval_bindings(external_plan(), bindings)

    def test_file_prefix_evidence_and_format(self):
        plan, bindings = external_plan(), eval_bindings()
        dataset = dict(type='jsonl', path='/data/input.jsonl', input_column='question', output_column='answer')
        bindings['testers']['tester']['datasets']['synthetic'] = dataset
        validate_eval_bindings(plan, bindings)
        plan['cases'][0]['shared_prefix_ratio'] = .5
        with self.assertRaisesRegex(ProtocolError, 'generation evidence'):
            validate_eval_bindings(plan, bindings)
        dataset['generation_manifest'] = '/data/manifest.json'
        validate_eval_bindings(plan, bindings)
        dataset['path'] = '/data/input.csv'
        with self.assertRaisesRegex(ProtocolError, 'format'):
            validate_eval_bindings(plan, bindings)

    def test_invalid_endpoint_and_unknown_field(self):
        for endpoint in ('https://user:secret@example.invalid', 'file:///tmp/model', 'http://h:0', 'https://h/?secret=x'):
            plan = external_plan()
            plan['service']['endpoint'] = endpoint
            with self.assertRaises(ProtocolError):
                normalize_external_plan(plan)
        plan = external_plan()
        plan['cases'][0]['concurency'] = 8
        with self.assertRaises(ProtocolError):
            normalize_external_plan(plan)
