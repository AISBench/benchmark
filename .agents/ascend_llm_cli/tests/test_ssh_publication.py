import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from deploy_eval_v2.blob_receiver import receive
from deploy_eval_v2.ssh_transport import exchange
from deploy_eval_v2 import blob_receiver
from deploy_eval_v2 import publication_receiver
from deploy_eval_v2.publication import build_archive,verify_runtime
from deploy_eval_v2.compiler import freeze_inputs,review_inputs
from deploy_eval_v2.input_publication import build_input_archive
from deploy_eval_v2.plan_model import normalize_external_plan
from test_plan_model import external_plan,eval_bindings


class UploadTests(unittest.TestCase):
    def test_real_child_installs_runtime_and_reviewed_inputs_then_reuses(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            runtime_archive=folder/'runtime.zip'
            published=build_archive(Path(blob_receiver.__file__).parent,runtime_archive)
            plan,_=normalize_external_plan(external_plan())
            bindings=eval_bindings()
            review=dict(schema_version='1.0',status='APPROVED',unresolved=[],evidence_refs=['review.json'],
                        inputs=review_inputs(plan,bindings,published['runtime_sha256']))
            ref=freeze_inputs(folder/'source',plan,bindings,review,published['runtime_sha256'])
            input_archive=folder/'inputs.zip'
            inputs=build_input_archive(folder/'source',input_archive,ref['sha256'],published['runtime_sha256'])
            target=folder/'remote'
            for archive in (runtime_archive,input_archive):
                checksum=hashlib.sha256(archive.read_bytes()).hexdigest()
                exchange([sys.executable,'-B',blob_receiver.__file__,str(target)],
                    dict(protocol_version='1.0',task_id='t',sha256=checksum,size=archive.stat().st_size),input_file=archive,timeout=10)
            request=dict(protocol_version='1.0',task_id='t',runtime_archive_sha256=published['archive_sha256'],
                         inputs_archive_sha256=inputs['archive_sha256'],runtime_sha256=published['runtime_sha256'],bundle_sha256=ref['sha256'])
            argv=[sys.executable,'-B',publication_receiver.__file__,str(target)]
            first=exchange(argv,request,timeout=15)
            self.assertEqual(first['status'],'EVENTS',first)
            self.assertFalse(first['inputs']['reused'])
            second=exchange(argv,request,timeout=15)
            self.assertTrue(second['runtime']['reused'])
            self.assertTrue(second['inputs']['reused'])
            verify_runtime(target/'runtime',published['runtime_sha256'])

    def packet(self,data,**overrides):
        header=dict(protocol_version='1.0',task_id='t',sha256=hashlib.sha256(data).hexdigest(),size=len(data))
        header.update(overrides)
        return io.BytesIO(json.dumps(header).encode()+b'\n'+data)

    def test_stream_real_child_upload_and_idempotent_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'archive.zip'
            with source.open('wb') as output:
                for _ in range(32):
                    output.write(b'x'*65536)
            checksum=hashlib.sha256(source.read_bytes()).hexdigest()
            request=dict(protocol_version='1.0',task_id='t',sha256=checksum,size=source.stat().st_size)
            target=Path(folder)/'remote'
            argv=[sys.executable,'-B',blob_receiver.__file__,str(target)]
            first=exchange(argv,request,input_file=source,timeout=5)
            second=exchange(argv,request,input_file=source,timeout=5)
            self.assertFalse(first['reused'])
            self.assertTrue(second['reused'])
            self.assertEqual((target/'.incoming'/(checksum+'.zip')).stat().st_size,2*1024*1024)

    def test_bad_digest_and_incomplete_stream_never_publish(self):
        for options in ({'sha256':'a'*64},{'size':4}):
            with tempfile.TemporaryDirectory() as folder:
                target=Path(folder)/'task'
                with self.assertRaises(ValueError):
                    receive(target,self.packet(b'abc',**options))
                self.assertEqual(list((target/'.incoming').iterdir()),[])

    def test_existing_different_content_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'task'
            receive(target,self.packet(b'abc'))
            artifact=target/'.incoming'/(hashlib.sha256(b'abc').hexdigest()+'.zip')
            artifact.write_bytes(b'xyz')
            with self.assertRaises(ValueError):
                receive(target,self.packet(b'abc'))
            self.assertEqual(artifact.read_bytes(),b'xyz')
