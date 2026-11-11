import copy
import unittest
from unittest.mock import patch
from test_compact_entry import inputs
from review_job import assess

class ReviewAssessmentTests(unittest.TestCase):
    def test_compile_once_per_deployment_and_network_error_once_per_binding(self):
        spec,bindings=inputs()
        dep=spec['deployments'][0]
        duplicate=copy.deepcopy(dep['instances'][0]);duplicate['id']='another'
        dep['instances'].append(duplicate)
        other=copy.deepcopy(dep);other['id']='second'
        spec['deployments'].append(other)
        for node in bindings['nodes'].values():
            node['deployments']['second']=copy.deepcopy(node['deployments'][dep['id']])
        evidence={slot:{'errors':['probe failure'],'network':[], 'models':{}} for slot in bindings['nodes']}
        # An unrelated evidence node must not multiply deployment compilation.
        evidence['extra']={'errors':[], 'network':[], 'models':{}}
        with patch('review_job.compile_deployment') as compile_dep:
            result=assess(spec,bindings,evidence)
        self.assertEqual(compile_dep.call_count,2)
        expected={d['id']+'/'+i['placement']['node'] for d in spec['deployments'] for i in d['instances']}
        network_errors=[e for e in result['errors'] if 'IP/interface' in e]
        self.assertEqual(len(network_errors),len(expected))
        self.assertEqual({e.split(':')[0] for e in network_errors},expected)
        self.assertEqual(len([e for e in result['errors'] if 'probe failure' in e]),len(bindings['nodes']))
        self.assertEqual(result['status'],'BLOCKED')
