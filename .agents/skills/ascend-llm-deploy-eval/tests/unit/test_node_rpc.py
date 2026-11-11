import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import node_rpc
import remote_node
from stage_runtime import digest


class NodeRpcTests(unittest.TestCase):
    def test_changed_script_or_registration_rejected_before_operation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'scripts').mkdir();script=root/'scripts/start.sh';script.write_text('original')
            spec={'schema_version':'resolved-1','job_id':'20261101-123','service_fingerprint':'fp','target':{'work_dir':str(root)}}
            (root/'plan.json').write_text(json.dumps({'execution_spec':spec,'execution_sha256':digest(spec)}))
            request={'run_dir':str(root),'job_id':spec['job_id'],'fingerprint':'fp','action':'start',
                'scripts_sha256':{'start.sh':hashlib.sha256(script.read_bytes()).hexdigest()}}
            script.write_text('changed')
            with self.assertRaisesRegex(ValueError,'script changed'):node_rpc.execute(request)
            request['fingerprint']='other'
            with self.assertRaisesRegex(ValueError,'identity differs'):node_rpc.execute(request)

    def test_transport_sends_json_data_and_preserves_failure(self):
        client=remote_node.RemoteNode('user@node.example','/task package/node_rpc.py','/task/run','20261101-123','fp')
        with patch.object(remote_node.subprocess,'run',return_value=SimpleNamespace(returncode=2,stdout='{"status":"FAILED","error":"busy"}\n',stderr='')) as execute:
            with self.assertRaisesRegex(RuntimeError,'busy'):client.call('check')
            self.assertEqual(json.loads(execute.call_args.kwargs['input'])['action'],'check')
            self.assertIn("'/task package/node_rpc.py'",execute.call_args.args[0][-1])
        with self.assertRaises(ValueError):remote_node.RemoteNode('-oProxyCommand=bad','/script','/run','id','fp')
