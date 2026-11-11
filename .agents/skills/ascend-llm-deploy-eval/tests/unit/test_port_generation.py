import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import cluster_runtime as cluster
from test_topology import deployment
from stage_runtime import save

class PortGeneration(unittest.TestCase):
    def test_recompile_all_nodes_and_preserve_other_deployment(self):
        dep=deployment(True)
        dep['runtime']={'image':'image','container':{}}
        spec={'job_id':'20261103-999','deployments':[dep], 'sub_tasks':[{'deployment_id':dep['id']}], 'execution':{},'cleanup':{}}
        bindings={'nodes':{n:{'host':'host-'+n,'work_dir':'/work','package':'/runtime','model_paths':{'m':'/model'}} for n in ('a','b')}}
        requests=[]
        def prepare(binding, request):
            requests.append(copy.deepcopy(request))
            if request['action']=='discover':
                bases={name:20000+i*100 for i,name in enumerate(request['port_requests'])}
                return {'network':{'ip':'192.0.2.'+('1' if binding['host']=='host-a' else '2'),'interface':'nic'},'port_bases':bases,'package':'/runtime','tools':{}}
            return {'run_dir':request['run_dir']+'/rendered','job_id':'20261103-999','service_fingerprint':'new','scripts_sha256':{}}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            save(root/'resolved-cluster.json',{'deployments':{'other':{'untouched':True}}})
            with patch.object(cluster,'prepare_call',side_effect=prepare):
                cluster.preflight(root,spec,bindings,deployment_id=dep['id'],generation='2')
            result=json.loads((root/'resolved-cluster.json').read_text())
            self.assertEqual(result['deployments']['other'],{'untouched':True})
            new=result['deployments'][dep['id']]
            self.assertEqual(len(new['groups']),2)
            self.assertIn(':20200',new['compiled']['endpoint'])
            self.assertTrue(all('/port-generations/2' in r['run_dir'].replace(chr(92),'/') for r in requests))
            renders=[r for r in requests if r['action']=='render']
            self.assertEqual(len(renders),2)
            text=json.dumps([r['deployment'] for r in renders])
            self.assertNotIn(':10001',text)
            self.assertNotIn(':11001',text)
            self.assertIn('192.0.2.2',text)

    def test_conflict_releases_before_reallocation_and_retries(self):
        from unittest.mock import Mock
        from remote_node import PortConflict
        with tempfile.TemporaryDirectory() as d:
            task=Path(d); (task/'common/inputs').mkdir(parents=True)
            save(task/'common/inputs/bindings.json', {'nodes':{}})
            old={'compiled':{'endpoint_group':'g','endpoint':'http://old:1'},'groups':{'g':{'old':True}}}
            new={'compiled':{'endpoint_group':'g','endpoint':'http://new:2'},'groups':{'g':{'new':True}}}
            order=[]
            first=Mock()
            def fail(*a,**k):
                raise PortConflict('occupied')
            first.call.side_effect=fail
            second=Mock(); second.call.return_value={'healthy':True,'identity':{'pid':2}}
            def release(*a):
                order.append('release'); return {'g':{'status':'PASSED'}}
            def rebind(*a,**k):
                order.append('reallocate'); save(task/'resolved-cluster.json',{'deployments':{'dep':new}})
            def curl(argv,**kw):
                output=Path(argv[argv.index('-o')+1])
                output.write_text(json.dumps({'choices':[{'message':{'content':'OK'}}]}))
                self.assertEqual(argv[-1],'http://new:2/v1/chat/completions')
            spec={'execution':{'max_retries':1,'startup_timeout_seconds':5,'smoke_timeout_seconds':5},
                  '_served_name':'m','_event_root':str(task),'_event_context':{'deployment_id':'dep'}}
            root=task/'services/dep'; root.mkdir(parents=True)
            with patch.object(cluster,'clients',side_effect=[{'g':first},{'g':second}]), patch.object(cluster,'release_all',side_effect=release), patch.object(cluster,'preflight',side_effect=rebind), patch.object(cluster,'command',side_effect=curl), patch.object(cluster,'emit'):
                cluster.deploy(root,spec,old)
            self.assertEqual(order,['release','reallocate'])
            self.assertEqual(old,new)
            self.assertEqual(json.loads((root/'service-state.json').read_text())['status'],'READY')
