import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from topology import compile_deployment, validate_deployment


def deployment(two_nodes=True):
    instances = []
    for name, role, node, devices, dp, tp in [
        ('p', 'prefill', 'a', [0,1,2,3], 2, 2),
        ('d', 'decode', 'b' if two_nodes else 'a', [0,1,2,3] if two_nodes else [4,5,6,7], 4, 1),
        ('proxy', 'proxy', 'a', [], 1, 1)]:
        instances.append({'id': name, 'role': role,
            'placement': {'node': node, 'devices': devices},
            'launch': {'args': {'data-parallel-size': dp, 'tensor-parallel-size': tp, 'max-model-len': 4096}},
            'ports': {'http':1, **({'kv':1} if role != 'proxy' else {})},
            'checks': {'health':'/health', 'prefix_cache_reset':'/reset_prefix_cache','metrics':'/metrics'}})
    return {'id':'test-pd','topology':'pd','model':{'id':'m','served_name':'test'},
        'instances':instances, 'connections':{'kv_transfer':[{'producer':'p','consumer':'d','connector':'TestConnector'}],
        'routing':[{'proxy':'proxy','prefillers':['p'],'decoders':['d']}]},'endpoint':{'instance':'proxy','port':'http'}}


def bindings():
    return {node: {'network':{'ip':ip,'interface':'data-test'},
        'ports':{'p':{'http':10001,'kv':10002},'d':{'http':11001,'kv':11002},'proxy':{'http':12001}}}
        for node,ip in [('a','192.0.2.1'),('b','192.0.2.2')]}


class TopologyTests(unittest.TestCase):
    def test_reference_preserves_device_env_and_engine_identifier(self):
        dep=deployment()
        for item in dep['instances']:
            if item['role']=='proxy': continue
            item['launch'].update(preserve_reference=True,engine_id='0')
        result=compile_deployment(dep,bindings())
        command=result['groups']['a/model']['commands'][0]
        self.assertNotIn('--device-ids',command['argv'])
        self.assertEqual(command['env']['ASCEND_RT_VISIBLE_DEVICES'],'0,1,2,3')
        self.assertNotIn('VLLM_SERVER_DEV_MODE',command['env'])
        kv=command['argv'][command['argv'].index('--kv-transfer-config')+1]['json']
        self.assertEqual(kv['engine_id'],'0')

    def test_mooncake_v2_does_not_force_legacy_endpoint_files(self):
        dep=deployment();dep['connections']['kv_transfer'][0]['connector']='MooncakeConnectorV2'
        result=compile_deployment(dep,bindings())
        argv=result['groups']['a/model']['commands'][0]['argv']
        kv=argv[argv.index('--kv-transfer-config')+1]['json']
        self.assertNotIn('ascend_local_comm_res_path',kv['kv_connector_extra_config'])

    def test_three_prefillers_dp_and_pcp(self):
        for pcp in (1,4):
            dep=deployment(); nodes=bindings()
            p=dep['instances'][0]
            p['launch']['args'].update({'data-parallel-size':4//pcp,'tensor-parallel-size':1,'prefill-context-parallel-size':pcp})
            for name,node,devices in [('p2','a',[4,5,6,7]),('p3','b',[4,5,6,7])]:
                peer=copy.deepcopy(p);peer['id']=name;peer['placement']={'node':node,'devices':devices}
                dep['instances'].append(peer)
                dep['connections']['kv_transfer'].append({'producer':name,'consumer':'d','connector':'TestConnector'})
                dep['connections']['routing'][0]['prefillers'].append(name)
                nodes[node]['ports'][name]={'http':13001,'kv':13002}
            result=compile_deployment(dep,nodes)
            self.assertEqual(sum(len(g['model_health_urls']) for g in result['groups'].values()),4)
            self.assertEqual(len(result['metrics_urls']),4)
            broken=copy.deepcopy(dep);broken['connections']['kv_transfer'].pop()
            self.assertTrue(validate_deployment(broken))
            broken=copy.deepcopy(dep);broken['instances'][-1]['launch']['args']['tensor-parallel-size']=2
            self.assertTrue(validate_deployment(broken))
            broken=copy.deepcopy(dep);broken['connections']['routing'][0]['prefillers'].pop()
            self.assertTrue(validate_deployment(broken))

    def test_single_node_pd_shares_container_and_partitions_devices(self):
        result=compile_deployment(deployment(False),bindings())
        self.assertEqual(list(result['groups']),['a/model'])
        self.assertEqual(result['groups']['a/model']['npu_ids'],list(range(8)))
        self.assertEqual(len(result['reset_urls']),2)

    def test_two_node_pd_resolves_peer_and_kv_parallelism(self):
        result=compile_deployment(deployment(),bindings())
        self.assertEqual(set(result['groups']),{'a/model','b/model'})
        proxy=result['groups']['a/model']['commands'][1]['argv']
        self.assertIn('192.0.2.2',proxy)
        args=result['groups']['a/model']['commands'][0]['argv']
        kv=args[args.index('--kv-transfer-config')+1]['json']
        self.assertEqual(kv['kv_connector_extra_config']['decode'],{'dp_size':4,'tp_size':1})

    def test_device_conflict_rejected_on_same_host_not_across_hosts(self):
        self.assertEqual(validate_deployment(deployment()),[])
        dep=deployment();dep['instances'][1]['placement']['node']='a'
        self.assertTrue(validate_deployment(dep))

    def test_duplicate_runtime_parameter_rejected(self):
        dep=deployment();dep['instances'][0]['launch']['args']['kv-transfer-config']={}
        self.assertTrue(validate_deployment(dep))

    def test_mixed_has_direct_endpoint_and_no_proxy(self):
        dep=deployment();dep['topology']='colocated';dep['instances']=dep['instances'][:1]
        dep['instances'][0]['role']='mixed';dep['connections']={};dep['endpoint']['instance']='p'
        result=compile_deployment(dep,bindings())
        self.assertEqual(result['endpoint'],'http://192.0.2.1:10001')
        self.assertEqual(len(result['groups']['a/model']['commands']),1)
        self.assertNotIn('--kv-transfer-config',result['groups']['a/model']['commands'][0]['argv'])
