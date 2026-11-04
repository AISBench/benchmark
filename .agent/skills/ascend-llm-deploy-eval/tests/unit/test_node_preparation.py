"""Render real per-node files using mocked discovery, without Docker or SSH."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import cluster_runtime as cluster
import node_prepare
import plan_run
from container_baseline import baseline_mounts
from test_serial_task import fixture


class NodePreparationTests(unittest.TestCase):
    def test_two_nodes_render_cross_host_routes_and_detect_changed_request(self):
        spec=plan_run.normalize(fixture());spec['deployments']=spec['deployments'][:1]
        dep=spec['deployments'][0]
        dep['instances'][1]['placement']['node']='b'
        calls=[]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model=root/'model';model.mkdir()
            bindings={'nodes':{key:{'host':key+'.example','work_dir':str(root/key),'package':'/skill',
                'model_paths':{'m':str(model)},'paths':{'comm':'/etc/hixlep'}} for key in ['a','b']}}
            def rpc(binding,request):
                calls.append(request)
                host=binding['host']
                with patch.object(node_prepare,'discover_network',return_value={'interface':'nic','ip':'192.0.2.1' if host.startswith('a') else '192.0.2.2'}), \
                     patch.object(node_prepare,'command',side_effect=lambda argv:json.dumps([{'Id':'image-sha'}]) if argv[0]=='docker' else 'idle'), \
                     patch.object(node_prepare,'inspect_image_tools',return_value={'aisbench_work_path':'/ais'}), \
                     patch('preflight.check',return_value={'status':'PASSED'}), patch('container_baseline.baseline_mounts', side_effect=lambda mounts: baseline_mounts(mounts, exists=lambda path: True)):
                    return node_prepare.prepare(request)
            with patch.object(cluster,'prepare_call',side_effect=rpc):
                cluster.preflight(root,spec,bindings)
            resolved=json.loads((root/'resolved-cluster.json').read_text())['deployments']['a']
            self.assertEqual(len(resolved['groups']),2)
            group=Path(resolved['groups']['a/model']['run_dir'])
            proxy=(group/'scripts/start-proxy.sh').read_text()
            self.assertIn('192.0.2.2',proxy)
            self.assertNotIn('{{',proxy)
            service=json.loads((group/'scripts/service.json').read_text())
            self.assertIn('192.0.2.2',service['reset_urls'][1])
            changed=next(r for r in calls if r['action']=='render')
            changed['deployment']['commands'][0]['env']['CHANGED']='1'
            with self.assertRaises(ValueError):node_prepare.prepare(changed)

    def test_unregistered_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'unrelated.txt').write_text('preserve')
            with self.assertRaises(ValueError):node_prepare.prepare({'run_dir':str(root),'job_id':'20261101-123','action':'discover'})
            self.assertEqual((root/'unrelated.txt').read_text(),'preserve')

    def test_pool_unions_models_and_devices_and_separates_images(self):
        spec=plan_run.normalize(fixture())
        spec['deployments'][1]['model']['id']='other'
        binding={'nodes':{'a':{'host':'host','work_dir':'/work','model_paths':{'m':'/m','other':'/other'}}}}
        pools=cluster.container_pools(spec,binding)
        one=pools[('a','a')];two=pools[('b','a')]
        self.assertIs(one,two)
        self.assertEqual(one['name'],spec['job_id']+'-node-a')
        self.assertEqual(one['devices'],list(range(8)))
        self.assertEqual(one['model_paths'],['/m','/other'])
        spec['deployments'][1]['runtime']['image']='different'
        pools=cluster.container_pools(spec,binding)
        self.assertNotEqual(pools[('a','a')]['name'],pools[('b','a')]['name'])

    def test_rendered_deployments_share_container_but_not_service_identity(self):
        spec=plan_run.normalize(fixture())
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model=root/'model';model.mkdir()
            bindings={'nodes':{'a':{'host':'host.example','work_dir':str(root/'node'),'package':'/skill','model_paths':{'m':str(model)},'paths':{'comm':'/etc/hixlep'}}}}
            def rpc(binding,request):
                with patch.object(node_prepare,'discover_network',return_value={'interface':'nic','ip':'192.0.2.1'}),patch.object(node_prepare,'command',side_effect=lambda argv:json.dumps([{'Id':'image-sha'}]) if argv[0]=='docker' else 'idle'),patch.object(node_prepare,'inspect_image_tools',return_value={'aisbench_work_path':'/ais'}),patch('preflight.check',return_value={'status':'PASSED'}), patch('container_baseline.baseline_mounts', side_effect=lambda mounts: baseline_mounts(mounts, exists=lambda path: True)):
                    return node_prepare.prepare(request)
            with patch.object(cluster,'prepare_call',side_effect=rpc):cluster.preflight(root,spec,bindings)
            resolved=json.loads((root/'resolved-cluster.json').read_text())
            executions=[json.loads((Path(v['groups']['a/model']['run_dir'])/'resolved.json').read_text()) for v in resolved['deployments'].values()]
            self.assertEqual({v['target']['container'] for v in executions},{spec['job_id']+'-node-a'})
            self.assertEqual(len({v['container_fingerprint'] for v in executions}),1)
            self.assertEqual(len({v['container_registry'] for v in executions}),1)
            self.assertEqual(len({v['service_fingerprint'] for v in executions}),2)
