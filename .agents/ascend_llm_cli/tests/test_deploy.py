import tempfile
import threading
from pathlib import Path
import unittest
from unittest.mock import Mock
from deploy_eval_v2.deploy import execute_deployment
from deploy_eval_v2.protocol import ProtocolError
from deploy_eval_v2.remote_guard import HostGuard


class DeployOrderingTests(unittest.TestCase):
    def run_deploy(self, folder, fail=None,retries=0,smoke_fail_once=False):
        calls=[]
        nodes={}
        groups={}
        scripts={}
        for index,name in enumerate(('a','b')):
            node=Mock()
            node.guard=HostGuard(Path(folder)/name,name,'a'*64)
            node.runner=Mock(side_effect=lambda argv, **kw: (1 if argv[0]=='fuser' else 0,b'',b''))
            for method in ('install_scripts','start_container','check_dependency','start_service'):
                def action(*args, tag=name+'.'+method):
                    calls.append(tag)
                    if tag==fail:
                        raise ProtocolError('UNKNOWN_OUTCOME')
                    return {}
                getattr(node,method).side_effect=action
            def create(*args, tag=name+'.create',cid=str(index+1)*64):
                calls.append(tag)
                if tag==fail:
                    raise ProtocolError('UNKNOWN_OUTCOME')
                return dict(container_id=cid)
            node.create.side_effect=create
            nodes[name]=node
            groups[name]=dict(node=name,npu_ids=[0],ports=[18000],
                commands=[dict(name='model',role='mixed',startup_health_urls=[]),
                          dict(name='proxy',role='proxy',startup_health_urls=['model-health'])],health_urls=['ready'])
            scripts[name]=[dict(instance=instance,purpose=purpose) for instance in ('model','proxy')
                           for purpose in ('service','dependency')]
        compiled=dict(topology=dict(groups=groups,endpoint='http://endpoint'),scripts=scripts)
        execution=dict(startup_timeout_seconds=30,dependency_timeout_seconds=10,
                       smoke_timeout_seconds=5,proxy_start_after_models_healthy=True,max_retries=retries)
        def health(urls,*args):
            calls.append('health:'+','.join(urls))
        def smoke(*args):
            calls.append('smoke')
            if smoke_fail_once and calls.count('smoke')==1:
                raise ProtocolError('INVALID_INPUT: smoke unsuccessful')
            return {'status':'VERIFIED'}
        report=execute_deployment(compiled,nodes,execution,Path(folder)/'result.json',threading.Event(),health=health,smoke=smoke)
        return report,calls,nodes

    def test_known_failed_smoke_retries_after_cleanup_with_fresh_command_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            report,calls,nodes=self.run_deploy(folder,retries=1,smoke_fail_once=True)
            self.assertEqual(report['execution_status'],'SUCCEEDED')
            self.assertEqual([x['execution_status'] for x in report['attempts']],['FAILED','SUCCEEDED'])
            self.assertEqual(report['attempts'][0]['cleanup_status'],'VERIFIED')
            ids=[call.args[0] for call in nodes['a'].create.call_args_list]
            self.assertEqual(len(set(ids)),2)
            self.assertTrue((Path(folder)/report['attempts'][0]['evidence_ref']).exists())

    def test_unknown_outcome_never_retried_even_when_known_resources_cleaned(self):
        with tempfile.TemporaryDirectory() as folder:
            report,_,nodes=self.run_deploy(folder,'b.check_dependency',retries=3)
            self.assertEqual(report['cleanup_status'],'VERIFIED')
            self.assertEqual(report['error_code'],'UNKNOWN_OUTCOME')
            self.assertEqual(len(report['attempts']),1)
            self.assertEqual(nodes['a'].create.call_count,1)
            self.assertTrue((Path(folder)/report['error_ref']).exists())

    def test_all_container_and_dependency_barriers_and_proxy_gates(self):
        with tempfile.TemporaryDirectory() as folder:
            report,calls,_=self.run_deploy(folder)
            self.assertEqual(report['execution_status'],'SUCCEEDED')
            self.assertLess(calls.index('b.create'),calls.index('a.start_container'))
            self.assertLess(max(i for i,c in enumerate(calls) if 'check_dependency' in c),calls.index('a.start_service'))
            self.assertLess(calls.index('b.start_service'),calls.index('health:model-health'))
            self.assertEqual(calls[-1],'smoke')

    def test_dependency_failure_never_starts_model_and_cleans_all_hosts(self):
        with tempfile.TemporaryDirectory() as folder:
            report,calls,_=self.run_deploy(folder,'b.check_dependency')
            self.assertEqual(report['execution_status'],'FAILED')
            self.assertEqual(report['cleanup_status'],'VERIFIED')
            self.assertFalse(any('start_service' in c for c in calls))

    def test_unknown_create_keeps_host_claim_and_cleans_other_host(self):
        with tempfile.TemporaryDirectory() as folder:
            report,calls,nodes=self.run_deploy(folder,'b.create')
            self.assertEqual(report['cleanup']['a']['cleanup_status'],'VERIFIED')
            self.assertEqual(report['cleanup']['b']['cleanup_status'],'UNVERIFIED')
            with self.assertRaisesRegex(ProtocolError,'RESOURCE_BUSY'):
                HostGuard(Path(folder)/'b','other','b'*64).claim(['npu:0'])
            self.assertFalse(any('start_container' in c for c in calls))
