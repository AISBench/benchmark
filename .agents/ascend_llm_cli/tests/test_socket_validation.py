import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from deploy_eval_v2.socket_validation import start_socket, start_installed
from deploy_eval_v2.protocol import ProtocolError
from deploy_eval_v2.ssh_publication import publish_task
import json
import subprocess
import sys


class SocketValidationTests(unittest.TestCase):
    def test_receiver_error_keeps_task_identity(self):
        from deploy_eval_v2 import socket_validation
        receiver=Path(socket_validation.__file__).with_name('socket_receiver.py')
        result=subprocess.run([sys.executable,str(receiver),'/'],input=json.dumps({'task_id':'t'}),text=True,capture_output=True)
        response=json.loads(result.stdout)
        self.assertEqual(response['task_id'],'t')
        self.assertEqual(response['status'],'ERROR')

    def test_invalid_nonce_precedes_upload(self):
        with patch('deploy_eval_v2.socket_validation.upload_archive') as upload:
            with self.assertRaises(ProtocolError):
                start_socket(dict(task_ref='t', nonce='bad'), {})
            upload.assert_not_called()

    def test_production_directory_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder)/'inputs').mkdir()
            with self.assertRaises(ProtocolError):
                start_installed(folder, dict(task_id='t', runtime_sha256='a'*64))

    def test_publication_preserves_task_identity(self):
        responses=[]
        def transport(argv, request, **kwargs):
            responses.append(request)
            return dict(status='EVENTS', runtime=dict(runtime_sha256='r'), inputs=dict(bundle_sha256='b'))
        with patch('deploy_eval_v2.ssh_publication.upload_archive', return_value={'sha256':'a'}):
            publish_task('host','/python','/tasks/t','r.zip','i.zip','task-identity','r','b',transport=transport)
        self.assertEqual(responses[0]['task_id'], 'task-identity')
