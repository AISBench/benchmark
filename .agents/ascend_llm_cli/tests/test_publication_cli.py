import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch,Mock
from deploy_eval_v2.cli import main
from deploy_eval_v2.publication_cli import publish,publication_targets
from deploy_eval_v2.compiler import freeze_inputs,review_inputs
from deploy_eval_v2.plan_model import normalize_external_plan
from deploy_eval_v2.protocol import ProtocolError
from test_cli import Output
from test_plan_model import external_plan,eval_bindings


class PublicationCLITests(unittest.TestCase):
    def test_remote_tester_must_have_matching_publication_binding(self):
        values={'plan.json':dict(task_id='t',intent='deploy',evaluation={'test_binding':'test'}),
                'bindings.json':{'testers':{'test':dict(host='tester',work_root='/tests')}},
                'compiled.json':{'topology':{'groups':{'a/model':dict(node='a',host='host-a',root='/work/t')}}}}
        binding=dict(host='host-a',root='/work/t',allowed_root='/work')
        with self.assertRaisesRegex(ProtocolError,'tester publication'):
            publication_targets(values,binding)
        binding['tester']=dict(host='tester',root='/tests/t',allowed_root='/tests')
        self.assertEqual([name for name,_ in publication_targets(values,binding)],['runtime','tester'])
        binding['tester']['root']='/wrong/t'
        with self.assertRaises(ProtocolError):
            publication_targets(values,binding)

    def test_all_node_targets_must_match_frozen_hosts_and_allowed_roots(self):
        values={'plan.json':{'intent':'deploy'},'compiled.json':{'topology':{'groups':{
            'a/model':dict(node='a',host='host-a',root='/work/t'),
            'b/model':dict(node='b',host='host-b',root='/tasks/t')}}}}
        main=dict(host='host-a',root='/work/t',allowed_root='/work')
        with self.assertRaisesRegex(ProtocolError,'target differs'):
            publication_targets(values,main)
        main['nodes']={'b':dict(host='host-b',root='/tasks/t',allowed_root='/tasks')}
        self.assertEqual([name for name,_ in publication_targets(values,main)],['runtime','b'])
        main['nodes']['b']['allowed_root']='/other'
        with self.assertRaisesRegex(ProtocolError,'task root'):
            publication_targets(values,main)

    def fixture(self,folder):
        folder=Path(folder)
        request=folder/'package.json'
        archive=folder/'runtime.zip'
        request.write_text(json.dumps(dict(destination=str(archive))))
        output=Output()
        with patch('deploy_eval_v2.cli.sys.stdout',output):
            code=main(['package-runtime','--request',str(request)])
        self.assertEqual(code,0)
        result=json.loads(output.buffer.getvalue())['runtime_package']
        plan,_=normalize_external_plan(external_plan())
        bindings=eval_bindings()
        review=dict(schema_version='1.0',status='APPROVED',unresolved=[],evidence_refs=['review.json'],
                    inputs=review_inputs(plan,bindings,result['runtime_sha256']))
        frozen=folder/'inputs'
        ref=freeze_inputs(frozen,plan,bindings,review,result['runtime_sha256'])
        return dict(task_ref='t',runtime_archive=str(archive),input_directory=str(frozen),
                    runtime_sha256=result['runtime_sha256'],bundle_sha256=ref['sha256'])

    def test_package_cli_then_validate_payload_before_publish(self):
        with tempfile.TemporaryDirectory() as folder:
            request=self.fixture(folder)
            publisher=Mock(return_value=dict(status='EVENTS'))
            config={'tasks':{'t':dict(host='host.invalid',root='/work/t',allowed_root='/work')}}
            result=publish(request,config,publisher=publisher)
            self.assertFalse(result['started'])
            args=publisher.call_args.args
            self.assertEqual(args[:3],('host.invalid','/usr/bin/python3','/work/t'))
            self.assertEqual(args[-2:],(request['runtime_sha256'],request['bundle_sha256']))
            # Temporary archive lifetime ends only after the synchronous publisher returns.
            self.assertFalse(Path(args[4]).exists())

    def test_invalid_digest_or_root_does_not_contact_host(self):
        with tempfile.TemporaryDirectory() as folder:
            request=self.fixture(folder)
            publisher=Mock()
            config={'tasks':{'t':dict(host='host.invalid',root='/other/t',allowed_root='/work')}}
            with self.assertRaises(ProtocolError):
                publish(request,config,publisher=publisher)
            config['tasks']['t']['root']='/work/t'
            request['bundle_sha256']='f'*64
            with self.assertRaises(ProtocolError):
                publish(request,config,publisher=publisher)
            publisher.assert_not_called()
