import copy
from pathlib import Path
import tempfile
import unittest
from deploy_eval_v2.compiler import review_inputs, freeze_inputs, verify_frozen, verify_review, digest,local_command,canonical
from deploy_eval_v2.json_io import load_json
from deploy_eval_v2.protocol import ProtocolError


class CompilerTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'schema_version': '1.0', 'explicit': {'kv': {'unknown_option': True}}, 'cases': []}
        self.bindings = {'node': {'host': 'example.invalid'}}
        self.runtime = 'a'*64
        self.review = {'schema_version': '1.0', 'status': 'APPROVED',
                       'inputs': review_inputs(self.plan, self.bindings, self.runtime),
                       'unresolved': [], 'evidence_refs': ['review/reference-inventory.json']}

    def test_freeze_preserves_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'frozen'
            reference = freeze_inputs(target, self.plan, self.bindings, self.review, self.runtime)
            result = verify_frozen(target, reference['sha256'], self.runtime)
            self.assertEqual(result['plan.json'], self.plan)
            with self.assertRaises(ProtocolError):
                freeze_inputs(target, self.plan, self.bindings, self.review, self.runtime)
            (target/'bindings.json').write_text('{}')
            with self.assertRaisesRegex(ProtocolError, 'changed'):
                verify_frozen(target, reference['sha256'], self.runtime)

    def test_stale_review_and_runtime(self):
        changed = copy.deepcopy(self.plan)
        changed['explicit']['kv']['unknown_option'] = False
        for plan, runtime in ((changed, self.runtime), (self.plan, 'b'*64)):
            with self.assertRaisesRegex(ProtocolError, 'stale review'):
                verify_review(plan, self.bindings, self.review, runtime)

    def test_digest_independent_of_object_key_order(self):
        self.assertEqual(digest({'a': 1, 'b': {'x': 2, 'y': 3}}), digest({'b': {'y': 3, 'x': 2}, 'a': 1}))

    def test_rehashed_summary_cannot_change_reviewed_meaning(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'frozen'
            reference=freeze_inputs(target,self.plan,self.bindings,self.review,self.runtime)
            summary=load_json(target/'review-summary.json')
            summary['scope']='unreviewed replacement'
            (target/'review-summary.json').write_bytes(canonical(summary))
            manifest=load_json(target/'manifest.json')
            manifest['files']['review-summary.json']=digest(summary)
            (target/'manifest.json').write_bytes(canonical(manifest))
            with self.assertRaisesRegex(ProtocolError,'summary differs'):
                verify_frozen(target,digest(manifest),self.runtime)

    def test_draft_review_is_not_approval_and_existing_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'plan.json').write_bytes(canonical(self.plan))
            (root/'bindings.json').write_bytes(canonical(self.bindings))
            request=dict(plan=str(root/'plan.json'),bindings=str(root/'bindings.json'),
                         destination=str(root/'review'),runtime_sha256=self.runtime)
            result=local_command('draft-review',request)
            self.assertFalse(result['approved'])
            review=load_json(root/'review/review.json')
            self.assertEqual(review['status'],'DRAFT')
            self.assertEqual(review['inputs']['summary_sha256'],result['summary_sha256'])
            with self.assertRaisesRegex(ProtocolError,'not approved'):
                verify_review(self.plan,self.bindings,review,self.runtime)
            with self.assertRaises(FileExistsError):
                local_command('draft-review',request)
