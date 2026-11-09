import copy
import unittest
from test_topology import deployment, bindings
from topology import compile_deployment, validate_deployment

class ProducerIdentityTests(unittest.TestCase):
    def dep(self):
        dep=deployment();p=dep['instances'][0]
        p['launch']['engine_id']='producer'
        other=copy.deepcopy(p);other['id']='p2';other['placement']['node']='b';other['placement']['devices']=[4,5,6,7]
        dep['instances'].append(other)
        dep['connections']['kv_transfer'][0]['connector']='MooncakeConnectorV2'
        dep['connections']['kv_transfer'].append({'producer':'p2','consumer':'d','connector':'MooncakeConnectorV2'})
        dep['connections']['routing'][0]['prefillers'].append('p2')
        return dep
    def test_duplicate_across_nodes_blocks_generation(self):
        dep=self.dep()
        with self.assertRaisesRegex(ValueError,'producer engine_id collision'):
            compile_deployment(dep,bindings())
    def test_unique_ids_pass_and_preserve_values(self):
        dep=self.dep();dep['instances'][-1]['launch']['engine_id']='other'
        self.assertEqual(validate_deployment(dep),[])
        nodes=bindings();nodes['b']['ports']['p2']={'http':13001,'kv':13002}
        compiled=compile_deployment(dep,nodes)
        cmd=compiled['groups']['b/model']['commands'][-1]['argv']
        self.assertEqual(cmd[cmd.index('--kv-transfer-config')+1]['json']['engine_id'],'other')
    def test_full_kv_override_is_checked(self):
        dep=self.dep();dep['instances'][-1]['launch']['engine_id']='other'
        dep['instances'][-1]['launch']['kv_transfer_config']={'engine_id':'producer'}
        self.assertTrue(any('collision' in e for e in validate_deployment(dep)))
    def test_no_unsupported_global_uniqueness_rule(self):
        dep=deployment();dep['connections']['kv_transfer'][0]['connector']='MooncakeConnectorV2'
        for item in dep['instances'][:2]:item['launch']['engine_id']='same'
        self.assertEqual(validate_deployment(dep),[])
        dep=self.dep()
        for link in dep['connections']['kv_transfer']:link['connector']='UnreviewedConnector'
        self.assertEqual(validate_deployment(dep),[])
if __name__=='__main__':unittest.main()
