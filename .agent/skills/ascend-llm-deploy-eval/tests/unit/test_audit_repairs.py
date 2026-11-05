import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from test_compact_entry import inputs, PACKAGE
import run as entry
from test_evidence import native_config_check
from aisbench_cli import validate_native_result
from review_job import assess


class AuditRepairs(unittest.TestCase):
    def test_binding_and_status_contracts_are_enforced(self):
        from jsonschema import ValidationError
        spec,bindings=inputs();bindings['nodes']['node_a']['host']='-oProxyCommand=unexpected'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'job'
            with self.assertRaises(ValidationError):entry.generate(spec,bindings,root)
            self.assertFalse(root.exists())
            with self.assertRaises(ValidationError):entry.save(Path(tmp)/'status.json',{'status':'SUCCESS'})
            self.assertFalse((Path(tmp)/'status.json').exists())

    def test_reset_configuration_conflicts_fail_before_generation(self):
        spec,bindings=inputs();spec['execution']['enable_cache_reset_api']=False;spec['cases'][0]['reset_cache']=True
        for item in spec['deployments'][0]['instances']:item['launch'].setdefault('env',{}).pop('VLLM_SERVER_DEV_MODE',None)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'reset_cache requires'):entry.generate(spec,bindings,Path(tmp)/'job')

    def test_generation_failure_is_transactional(self):
        spec,bindings=inputs()
        bindings['nodes']['node_a']['paths'].clear()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'job'
            with self.assertRaises(ValueError):entry.generate(spec,bindings,root)
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_reserved_and_colliding_scripts_rejected(self):
        spec,bindings=inputs();dep=spec['deployments'][0]
        old=dep['id'];dep['id']='tests';spec['sub_tasks'][0]['deployment_id']='tests'
        bindings['nodes']['node_a']['deployments']['tests']=bindings['nodes']['node_a']['deployments'].pop(old)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'reserved'):entry.generate(spec,bindings,Path(tmp)/'job')
            file=Path(tmp)/'same.sh';entry.script(file,'true')
            with self.assertRaisesRegex(ValueError,'collision'):entry.script(file,'false')
            self.assertIn('true',file.read_text())

    def test_confirmed_network_conflict_is_not_repaired(self):
        spec,bindings=inputs()
        spec['deployments'][0]['instances'][0]['launch'].setdefault('env',{})['HCCL_IF_IP']='192.0.2.99'
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'conflicts with binding'):entry.generate(spec,bindings,Path(tmp)/'job')

    def test_proxy_extra_args_survive(self):
        spec,bindings=inputs()
        proxy=next(i for i in spec['deployments'][0]['instances'] if i['role']=='proxy')
        proxy['launch']['args']={'workers':4,'max-retries':9}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'job';entry.generate(spec,bindings,root)
            text=(root/'scripts'/spec['deployments'][0]['id']/('node_a-'+proxy['id']+'.sh')).read_text()
            self.assertIn('--workers 4',text);self.assertIn('--max-retries 9',text)

    def test_explicit_kv_extra_config_survives_and_conflict_fails(self):
        spec,bindings=inputs();dep=spec['deployments'][0];item=dep['instances'][0]
        item['launch']['kv_transfer_config']={'kv_connector':dep['connections']['kv_transfer'][0]['connector'],
            'kv_role':'kv_producer','kv_port':str(bindings['nodes']['node_a']['deployments'][dep['id']]['ports'][item['id']]['kv']),
            'engine_id':'0','kv_connector_extra_config':{'custom_option':{'nested':17}}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'job';entry.generate(spec,bindings,root)
            text=(root/'scripts'/dep['id']/('node_a-'+item['id']+'.sh')).read_text()
            self.assertIn('custom_option',text);self.assertNotIn('ascend_local_comm_res_path',text)
            item['launch']['kv_transfer_config']['kv_port']='1'
            with self.assertRaisesRegex(ValueError,'conflicts with topology'):entry.generate(spec,bindings,Path(tmp)/'bad')

    def test_remote_claim_shared_alias_and_second_coordinator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'.runtime').mkdir()
            plan={'deployments':{'A':{'groups':{'a':{'host':'a','root':tmp},'b':{'host':'b','root':tmp}}}}}
            with patch.object(entry,'transport_argv',side_effect=lambda host,argv:[sys.executable,*argv[1:]]):
                entry.claim_execution(plan,'a'*32)
                entry.claim_execution(plan,'a'*32)
                with self.assertRaisesRegex(RuntimeError,'already claimed'):entry.claim_execution(plan,'b'*32)
            self.assertEqual((root/'.runtime/execution.claim').read_text(),'a'*32)

    def test_collection_missing_artifacts_fails_even_shared(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'.runtime').mkdir()
            entry.save(root/'status.json',{'status':'FAILED','cases':{},'release':{},'expected_artifacts':['results/x/validated.json']})
            plan={'deployments':{'A':{'groups':{'a':{'host':'a','root':tmp}}}}}
            def remote(argv,**kwargs):return Mock(returncode=0,stdout=Path(argv[1]).read_text())
            with patch.object(entry,'transport_argv',side_effect=lambda h,a:a),patch.object(entry.subprocess,'run',side_effect=remote):
                with self.assertRaisesRegex(RuntimeError,'Missing required'):entry.collect(root,plan)

    def test_native_configuration_and_missing_result_group(self):
        case={'concurrency':1,'output_tokens':8,'formal_requests':1}
        config={'models':[{'abbr':'m','model':'served','path':'/weights','host_ip':'192.0.2.1','host_port':80,
                           'batch_size':1,'max_out_len':8,'request_rate':0}], 'datasets':[{'abbr':'a'},{'abbr':'b'}]}
        expected=native_config_check(config,case,{'served_name':'served'},'http://192.0.2.1:80','/weights')
        changed=copy.deepcopy(config);changed['models'][0]['host_ip']='wrong'
        with self.assertRaisesRegex(ValueError,'host_ip'):native_config_check(changed,case,{'served_name':'served'},'http://192.0.2.1:80','/weights')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'execution.json').write_text('{"returncode":0}')
            (root/'native-expectations.json').write_text(json.dumps(expected))
            file=root/'outputs/time/performances/m/a_details.jsonl';file.parent.mkdir(parents=True)
            file.write_text('{"id":0,"success":true}\n')
            with self.assertRaisesRegex(ValueError,'result groups'):validate_native_result(root,case)

    def test_generated_reset_script_has_before_and_after(self):
        spec,bindings=inputs();spec['cases'][0]['reset_cache']=True
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'job';entry.generate(spec,bindings,root)
            text=next((root/'scripts/tests').glob('*.sh')).read_text()
            self.assertIn('"$OUT" before',text);self.assertIn('"$OUT" after "$RC"',text)

    def test_review_uses_weights_and_machine_without_claiming_approval(self):
        spec,bindings=inputs()
        evidence={'node_a':{'errors':[],'network':[{'ifname':'eth0','addr_info':[{'local':'192.0.2.1'}]}],
            'models':{'dsv4':{'config':{'architectures':['Example'],'max_position_embeddings':8192}}}}}
        report=assess(spec,bindings,evidence)
        self.assertEqual(report['status'],'REQUIRES_AGENT_REVIEW')
        self.assertEqual(report['review'][0]['model_config']['max_position_embeddings'],8192)
        evidence['node_a']['network']=[]
        self.assertEqual(assess(spec,bindings,evidence)['status'],'BLOCKED')

    def test_test_evidence_resets_every_instance_and_validates_outputs(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        from test_evidence import check
        requests=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_GET(self):
                requests.append(('GET',self.path));self.send_response(200);self.end_headers()
                self.wfile.write(b'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nvllm:prefix_cache_queries_total 1\nvllm:prefix_cache_hits_total 0\n')
            def do_POST(self):
                requests.append(('POST',self.path));self.send_response(200);self.end_headers();self.wfile.write(b'{"success":true}')
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);folder=root/'result';folder.mkdir();(root/'.runtime').mkdir()
                spec,bindings=inputs();case=spec['cases'][0]
                case.update(concurrency=1,formal_requests=1,input_tokens=1,output_tokens=8,reset_cache=True)
                spec=entry.normalize(spec);sub=spec['sub_tasks'][0]
                dataset=root/'dataset.jsonl';dataset.write_text('{"question":"hello","answer":"yes"}\n')
                bindings['nodes']['node_a']['benchmark']['datasets'][case['dataset_id']]['path']=str(dataset)
                base='http://127.0.0.1:'+str(server.server_port)
                service={'reset_urls':[base+'/p/reset_prefix_cache',base+'/d/reset_prefix_cache'],
                    'metric_instances':[{'id':'p','role':'prefill','url':base+'/p/metrics'},{'id':'d','role':'decode','url':base+'/d/metrics'}]}
                entry.save(root/'job_spec.json',spec);entry.save(root/'bindings.json',bindings)
                entry.save(root/'.runtime/dispatch.json',{'deployments':{sub['deployment_id']:service}})
                check(str(root),sub['id'],'node_a',str(folder),'before')
                self.assertEqual([path for method,path in requests if method=='POST'],['/p/reset_prefix_cache','/d/reset_prefix_cache'])
                output=folder/'outputs/time/performances/m/data_details.jsonl';output.parent.mkdir(parents=True)
                output.write_text('{"id":0,"success":true,"input_tokens":1,"output_tokens":1}\n')
                log=folder/'outputs/time/logs/worker.log';log.parent.mkdir(parents=True);log.write_text('done')
                check(str(root),sub['id'],'node_a',str(folder),'after','0')
                self.assertEqual(json.loads((folder/'cache-reset.json').read_text())['status'],'PASSED')
                self.assertEqual(json.loads((folder/'prefix-evidence.json').read_text())['status'],'VERIFIED')
                self.assertEqual(json.loads((folder/'validated.json').read_text())['status'],'PASSED')
        finally:
            server.shutdown();server.server_close();worker.join(timeout=2)


if __name__=='__main__':unittest.main()
