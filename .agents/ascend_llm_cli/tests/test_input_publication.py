from pathlib import Path
import tempfile
import unittest
import zipfile
from deploy_eval_v2.input_publication import build_input_archive,install_input_archive
from deploy_eval_v2.compiler import freeze_inputs,review_inputs,verify_frozen
from deploy_eval_v2.protocol import ProtocolError
from deploy_eval_v2.plan_model import normalize_external_plan
from test_plan_model import external_plan,eval_bindings


class InputPublicationTests(unittest.TestCase):
    def fixture(self,folder):
        plan,_=normalize_external_plan(external_plan())
        bindings=eval_bindings()
        runtime='a'*64
        review=dict(schema_version='1.0',status='APPROVED',unresolved=[],evidence_refs=['review.json'],
                    inputs=review_inputs(plan,bindings,runtime))
        root=Path(folder)/'source'
        ref=freeze_inputs(root,plan,bindings,review,runtime)
        archive=Path(folder)/'inputs.zip'
        build_input_archive(root,archive,ref['sha256'],runtime)
        return archive,ref['sha256'],runtime

    def test_install_reuse_and_refuse_changed_existing_input(self):
        with tempfile.TemporaryDirectory() as folder:
            archive,bundle,runtime=self.fixture(folder)
            target=Path(folder)/'remote'/'inputs'
            self.assertFalse(install_input_archive(archive,target,bundle,runtime)['reused'])
            verify_frozen(target,bundle,runtime)
            self.assertTrue(install_input_archive(archive,target,bundle,runtime)['reused'])
            (target/'bindings.json').write_text('{}')
            with self.assertRaises(ProtocolError):
                install_input_archive(archive,target,bundle,runtime)
            self.assertEqual((target/'bindings.json').read_text(),'{}')

    def test_traversal_and_duplicate_entries_never_publish(self):
        for bad in ('../escape.json','plan.json'):
            with tempfile.TemporaryDirectory() as folder:
                archive,bundle,runtime=self.fixture(folder)
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',UserWarning)
                    with zipfile.ZipFile(archive,'a') as handle:
                        handle.writestr(bad,'{}')
                target=Path(folder)/'remote'
                with self.assertRaises(ProtocolError):
                    install_input_archive(archive,target,bundle,runtime)
                self.assertFalse(target.exists())
                self.assertFalse((Path(folder)/'escape.json').exists())

    def test_task_identity_checked_before_install(self):
        with tempfile.TemporaryDirectory() as folder:
            archive,bundle,runtime=self.fixture(folder)
            target=Path(folder)/'remote'
            with self.assertRaisesRegex(ProtocolError,'task identity'):
                install_input_archive(archive,target,bundle,runtime,expected_task_id='other')
            self.assertFalse(target.exists())
