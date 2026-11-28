import copy
import json
from pathlib import Path
import unittest
from deploy_eval_v2.deployment_compiler import compile_bound_deployment
from deploy_eval_v2.json_io import load_json
from deploy_eval_v2.protocol import ProtocolError
from test_deployment_model import deploy_plan


def bindings():
    value = load_json(Path(__file__).parent/'fixtures/deploy-bindings.json')
    value.update(schema_version='1.0', task_id='t')
    return value


class DeploymentCompilerTests(unittest.TestCase):
    def test_complete_reference_compiles_without_mutation(self):
        plan, bound = deploy_plan(), bindings()
        original = copy.deepcopy((plan, bound))
        result = compile_bound_deployment(plan, bound)
        self.assertEqual((plan, bound), original)
        self.assertEqual(result['endpoint'], 'http://192.0.2.1:18056')
        commands = result['groups']['node_a/model']['commands']
        model = commands[0]
        self.assertIn('/path/to/model-weights', model['argv'])
        self.assertIn('unset HCCL_INTRA_ROCE_ENABLE', model['bootstrap'])
        kv = json.loads(model['argv'][model['argv'].index('--kv-transfer-config')+1])
        self.assertEqual(kv['kv_connector'], 'MooncakeHybridConnector')
        self.assertEqual(len(commands[-1]['startup_health_urls']), 2)

    def test_conflicting_ports_devices_and_environment(self):
        for kind in ('port', 'device', 'env', 'missing'):
            plan, bound = deploy_plan(), bindings()
            if kind == 'port':
                bound['nodes']['node_a']['deployments']['pd-a']['ports']['decode']['http'] = 18000
            elif kind == 'device':
                plan['deployment']['instances'][1]['placement']['devices'][0] = 0
            elif kind == 'env':
                plan['deployment']['instances'][0]['launch']['env']['HCCL_IF_IP'] = '192.0.2.99'
            else:
                del bound['nodes']['node_a']['paths']['proxy_program']
            with self.assertRaises(ProtocolError):
                compile_bound_deployment(plan, bound)
