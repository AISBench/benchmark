"""Topology task planning and serial coordinator checks."""
import copy
import io
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import plan_run
import task_preflight as preflight
import task_runtime as task
import stage_runtime as runtime
import cluster_runtime as cluster
from test_topology import deployment


def fixture():
    dep=deployment(False)
    dep['runtime']={'image':'test-image','container':{'shm_size':'1g','privileged':False}}
    dep['id']='a'
    second=copy.deepcopy(dep);second['id']='b'
    return {'schema_version':'3.0','job_id':'20261101-101','state':'ready',
      'deployments':[dep,second], 'cases':[{'id':'c1','input_tokens':32,'output_tokens':8,'concurrency':1},
      {'id':'c2','input_tokens':32,'output_tokens':8,'concurrency':2}],
      'sub_tasks':[{'id':'t1','deployment_id':'a','case_id':'c1'}, {'id':'t2','deployment_id':'a','case_id':'c2'},
                   {'id':'t3','deployment_id':'b','case_id':'c1'}], 'unresolved':[]}


class SerialTests(unittest.TestCase):
    def test_plan_defaults_and_reusable_case_order(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);result=plan_run.plan(fixture(),root)
            self.assertEqual([c['formal_requests'] for c in result['job_spec']['cases']],[4,8])
            self.assertEqual([t['case_id'] for t in result['sequence']],['c1','c2','c1'])
            self.assertNotIn(b'\r',(root/'run.sh').read_bytes())
            self.assertEqual(task.read_task(root),result['job_spec'])

    def test_duplicate_or_missing_reference_rejected(self):
        spec=plan_run.normalize(fixture());spec['sub_tasks'][0]['case_id']='missing'
        self.assertTrue(plan_run.validate(spec))
        spec=plan_run.normalize(fixture());spec['sub_tasks'][1]['id']='t1'
        self.assertTrue(plan_run.validate(spec))

    def test_multiple_networks_require_a_specific_choice(self):
        interfaces=[{'ifname':n,'operstate':'UP','addr_info':[{'family':'inet','scope':'global','local':ip}]} for n,ip in [('a','10.1.0.1'),('b','10.2.0.1')]]
        with patch.object(preflight,'command',return_value=json.dumps(interfaces)),patch.object(preflight,'interface_speed',return_value=1000):
            with self.assertRaises(ValueError):preflight.discover_network()
            self.assertEqual(preflight.discover_network('b')['ip'],'10.2.0.1')

    def test_shell_binding_quotes_and_missing_value(self):
        self.assertEqual(preflight.expand('source {{paths.env}}',{'paths':{'env':'/path with spaces/$(bad)'}},shell=True),"source '/path with spaces/$(bad)'")
        with self.assertRaises(ValueError):preflight.expand('{{ports.missing}}',{'ports':{}})

    def test_loop_switches_and_always_finalizes(self):
        spec=plan_run.normalize(fixture())
        resolved={'deployments':{key:{'compiled':{'endpoint_group':'g'}} for key in ['a','b']}}
        client=MagicMock();client.call.return_value={'status':'PASSED','identity':{'test':1}}
        with tempfile.TemporaryDirectory() as d,patch.object(cluster,'deploy') as deploy, \
             patch.object(cluster,'clients',return_value={'g':client}), \
             patch.object(cluster,'release_all',return_value={'g':{'status':'PASSED'}}) as release, \
             patch.object(cluster,'finalize',return_value={'status':'PASSED'}) as finalize:
            result=cluster.run(Path(d),spec,resolved)
            self.assertEqual(result['status'],'PASSED')
            self.assertEqual(deploy.call_count,3)
            self.assertEqual([c.args[1] for c in release.call_args_list],['keep_service','stop_service','keep_service'])
            self.assertEqual([c.kwargs['case']['id'] for c in client.call.call_args_list if c.args[0]=='evaluate'],['t1','t2','t3'])
            finalize.assert_called_once()

    def test_completed_entry_only_repeats_cleanup(self):
        with tempfile.TemporaryDirectory() as d,patch.object(cluster,'deploy') as deploy,patch.object(cluster,'finalize',return_value={'status':'PASSED'}) as finalize:
            root=Path(d);runtime.save(root/'summary.json',{'status':'PASSED','sub_tasks':{}})
            self.assertEqual(cluster.run(root,{}, {})['status'],'PASSED')
            deploy.assert_not_called();finalize.assert_called_once()

    def test_transport_failure_stops_and_still_finalizes(self):
        spec=plan_run.normalize(fixture())
        resolved={'deployments':{'a':{'compiled':{'endpoint_group':'g'}},'b':{}}}
        with tempfile.TemporaryDirectory() as d,patch.object(cluster,'deploy',side_effect=RuntimeError('connection lost')), patch.object(cluster,'release_all',return_value={'g':{'status':'FAILED'}}), \
             patch.object(cluster,'finalize',return_value={'status':'FAILED'}) as finalize:
            result=cluster.run(Path(d),spec,resolved)
            self.assertEqual(result['status'],'FAILED')
            self.assertEqual(result['sub_tasks']['t2']['status'],'NOT_RUN')
            finalize.assert_called_once()

    def test_release_failure_does_not_skip_other_nodes(self):
        one,two=MagicMock(),MagicMock()
        one.call.side_effect=RuntimeError('node unavailable');two.call.return_value={'status':'PASSED'}
        with patch.object(cluster,'clients',return_value={'one':one,'two':two}):
            result=cluster.release_all({},'stop_container','job')
        self.assertEqual(result['one']['status'],'FAILED');two.call.assert_called_once()

class CaseIsolationTests(unittest.TestCase):
    def test_exhausted_case_does_not_consume_next_case_budget(self):
        import evaluate
        from types import SimpleNamespace
        task_spec=plan_run.normalize(fixture())
        spec={'execution':task_spec['execution'],'benchmark':{'tool_dir':'unused'}}
        first=copy.deepcopy(task_spec['cases'][0]); first['id']='first'
        second=copy.deepcopy(first); second['id']='second'
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['scripts','logs','results']:(root/name).mkdir()
            runtime.save(root/'deployment-artifacts.json', {'id':'test-01','path':str(root/'deployments/test-01')})
            runtime.update(root,service='READY',container_id='test')
            calls=[]
            def attempt(argv,**kwargs):
                folder=Path(argv[-1]); case_id=folder.parent.name; calls.append((case_id,folder.name))
                success=case_id=='second'
                runtime.save(folder/'cleanup.json',{'released':True})
                runtime.save(folder/'summary.json',{'status':'PASSED' if success else 'FAILED','error':'test failure' if not success else None})
                return SimpleNamespace(returncode=0 if success else 2)
            with patch.object(evaluate,'read_plan',return_value=spec),patch.object(evaluate,'ready',return_value=True), \
                 patch.object(evaluate,'service_identity',return_value={'instance':1}), \
                 patch.object(evaluate.subprocess,'run',side_effect=attempt),patch.object(evaluate.time,'sleep'),redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):evaluate.evaluate(root,first)
                self.assertEqual(evaluate.evaluate(root,second)['status'],'PASSED')
            self.assertEqual(calls,[('first',f'attempt-{i:02}') for i in range(1,5)]+[('second','attempt-01')])
            self.assertEqual(runtime.case_state(root,'first')['status'],'FAILED')
            self.assertEqual(runtime.case_state(root,'second')['status'],'PASSED')

class RecordOnlyTests(unittest.TestCase):
    def test_failed_case_continues_and_overall_remains_failed(self):
        spec=plan_run.normalize(fixture())
        dep={'compiled':{'endpoint_group':'g'}}
        client=MagicMock()
        def call(action,**kwargs):
            if action=='evaluate' and kwargs['case']['id']=='t1':raise RuntimeError('case failed')
            return {'status':'PASSED','identity':{'test':1}}
        client.call.side_effect=call
        with tempfile.TemporaryDirectory() as d,patch.object(cluster,'deploy'),patch.object(cluster,'clients',return_value={'g':client}),patch.object(cluster,'reconcile_failure',return_value={'continue':True}),patch.object(cluster,'release_all',return_value={'g':{'status':'PASSED'}}),patch.object(cluster,'finalize',return_value={'status':'PASSED'}):
            result=cluster.run(Path(d),spec,{'deployments':{'a':dep,'b':dep}})
            self.assertEqual(result['status'],'FAILED')
            self.assertEqual([result['sub_tasks'][x]['status'] for x in ['t1','t2','t3']],['FAILED','PASSED','PASSED'])
            self.assertEqual(result['sub_tasks']['t1']['on_failure'],'record_only')

    def test_reconcile_health_death_and_unknown(self):
        for status,release_ok,expected in [('READY',True,'RELEASED'),('DEAD',True,'RELEASED'),('DEAD',False,'UNKNOWN'),('UNKNOWN',True,'RELEASED')]:
            with self.subTest(status=status,release_ok=release_ok),tempfile.TemporaryDirectory() as d:
                node=MagicMock();node.call.return_value={'service':status,'identity':{'pid':1}}
                with patch.object(cluster,'clients',return_value={'g':node}),patch.object(cluster,'release_all',return_value={'g':{'status':'PASSED' if release_ok else 'FAILED'}}) as release:
                    result=cluster.reconcile_failure(Path(d),{},'t1')
                self.assertEqual(json.loads((Path(d)/'service-state.json').read_text())['status'],expected)
                self.assertEqual(result['continue'],expected!='UNKNOWN')
                self.assertEqual(release.call_count,1)

    def test_reconcile_disconnect_blocks_unverified_release(self):
        node=MagicMock();node.call.side_effect=TimeoutError('offline')
        with tempfile.TemporaryDirectory() as d,patch.object(cluster,'clients',return_value={'g':node}),patch.object(cluster,'release_all',return_value={'g':{'status':'FAILED'}}) as release:
            self.assertFalse(cluster.reconcile_failure(Path(d),{},'t1')['continue'])
            release.assert_called_once()

    def test_process_probe_distinguishes_exit_from_pid_reuse(self):
        import service_process
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);runtime.save(root/'service-processes.json',[{'pid':12,'start_ticks':'100'}])
            output=io.StringIO()
            with patch.object(service_process,'ticks',return_value=None),redirect_stdout(output):
                service_process.main('probe',root)
            self.assertFalse(json.loads(output.getvalue())['alive'])
            with patch.object(service_process,'ticks',return_value='200'):
                with self.assertRaises(ValueError):service_process.main('probe',root)
