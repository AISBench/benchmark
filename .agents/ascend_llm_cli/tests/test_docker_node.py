import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock
from deploy_eval_v2.docker_node import DockerNode
from deploy_eval_v2.remote_guard import HostGuard
from deploy_eval_v2.protocol import ProtocolError


class DockerNodeTests(unittest.TestCase):
    def test_create_pins_image_and_records_identity_before_start(self):
        with tempfile.TemporaryDirectory() as folder:
            guard = HostGuard(Path(folder)/'guard', 't', 'b'*64)
            cid, image = 'c'*64, 'sha256:'+'a'*64
            inspect = json.dumps(dict(Id=cid, Config=dict(Labels={'bench.task':'t', 'bench.execution':'b'*64}))).encode()
            runner = Mock(side_effect=[(1,b'',b''), (0,image.encode(),b''), (0,cid.encode(),b''), (0,inspect,b'')])
            node = DockerNode(guard, runner)
            group = dict(npu_ids=[0], ports=[], image='registry/model:tag',
                         container=dict(privileged=False, devices=[]), mounts=[])
            deadline, cancelled = time.monotonic()+30, threading.Event()
            result = node.create('create1', group, deadline, cancelled)
            self.assertEqual(result['container_id'], cid)
            argv = runner.call_args_list[2].args[0]
            self.assertEqual(argv[-3:], [image, 'sleep', 'infinity'])
            self.assertIn('bench.execution='+'b'*64, argv)
            self.assertNotIn('start', argv)
            guard.release_claims(['npu:0'], resources_verified_absent=True)
            HostGuard(Path(folder)/'guard', 'next', 'd'*64).claim(['npu:0'])
            self.assertEqual(node.create('create1', group, deadline, cancelled), result)
            self.assertEqual(runner.call_count, 4)

    def test_lost_create_response_never_repeats_docker_create(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = Mock(side_effect=[(1,b'',b''), (0,('sha256:'+'a'*64).encode(),b''),
                                       ProtocolError('UNKNOWN_OUTCOME')])
            node = DockerNode(HostGuard(folder,'t','b'*64), runner)
            group = dict(npu_ids=[0], ports=[], image='x',container=dict(privileged=False,devices=[]), mounts=[])
            for _ in range(2):
                with self.assertRaises(ProtocolError):
                    node.create('create1',group,time.monotonic()+30,threading.Event())
            self.assertEqual(runner.call_count,3)

    def test_busy_npu_blocks_image_lookup_and_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            runner = Mock(return_value=(0,b'1234',b''))
            node = DockerNode(HostGuard(folder,'t','b'*64),runner)
            group = dict(npu_ids=[0],ports=[],image='x',container={},mounts=[])
            with self.assertRaisesRegex(ProtocolError,'RESOURCE_BUSY'):
                node.create('create1',group,time.monotonic()+30,threading.Event())
            self.assertEqual(runner.call_count,1)
