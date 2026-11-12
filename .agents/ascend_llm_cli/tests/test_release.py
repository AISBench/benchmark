import json
import unittest
import tempfile
import time
import socket
from pathlib import Path
from unittest.mock import Mock
from deploy_eval_v2.release import release_container, release_host_resources, verify_resources_absent
from deploy_eval_v2.remote_guard import HostGuard
from deploy_eval_v2.json_io import load_json
from deploy_eval_v2.protocol import ProtocolError


class ReleaseTests(unittest.TestCase):
    def test_resource_probe_checks_real_port_and_fuser_errors(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen()
            resource='tcp:'+str(listener.getsockname()[1])
            self.assertEqual(verify_resources_absent([resource], Mock(), time.monotonic()+5)[0]['status'], 'UNVERIFIED')
        self.assertEqual(verify_resources_absent([resource], Mock(), time.monotonic()+5)[0]['status'], 'ABSENT_VERIFIED')
        for response in ((0,b'123',b''),(1,b'',b'missing device'),(127,b'',b'not found')):
            self.assertEqual(verify_resources_absent(['npu:0'],Mock(return_value=response),time.monotonic()+5)[0]['status'],'UNVERIFIED')

    def test_absent_container_with_busy_npu_retains_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            guard=HostGuard(Path(folder)/'guard','t','a'*64)
            guard.claim(['npu:0'])
            runner=Mock(side_effect=[(0,b'',b''),(0,b'123',b'')])
            target=Path(folder)/'release.json'
            result=release_host_resources(guard,'release1',['a'*64],['npu:0'],target,runner=runner)
            self.assertEqual(result['cleanup_status'],'UNVERIFIED')
            self.assertFalse(result['claims_released'])
            self.assertEqual(load_json(target)['resource_results'][0]['status'],'UNVERIFIED')
            with self.assertRaisesRegex(ProtocolError,'RESOURCE_BUSY'):
                HostGuard(Path(folder)/'guard','next','b'*64).claim(['npu:0'])

    def test_owned_release_and_absence_verification(self):
        cid = 'a'*64
        evidence = json.dumps({'Id': cid, 'Config': {'Labels': {'bench.task': 't', 'bench.execution': 'token'}}}).encode()
        runner = Mock(side_effect=[(0, cid.encode()+b'\n', b''), (0, evidence, b''), (0, b'', b''), (0, b'', b'')])
        self.assertEqual(release_container(cid, 't', 'token', runner)['status'], 'RELEASED_VERIFIED')
        self.assertEqual(runner.call_args_list[2].args[0], ['docker', 'rm', '-f', cid])

    def test_foreign_container_is_not_removed(self):
        cid = 'a'*64
        evidence = json.dumps({'Id': cid, 'Config': {'Labels': {'bench.task': 'other'}}}).encode()
        runner = Mock(side_effect=[(0, cid.encode(), b''), (0, evidence, b'')])
        with self.assertRaises(ProtocolError):
            release_container(cid, 't', 'token', runner)
        self.assertEqual(runner.call_count, 2)

    def test_deadline_prevents_new_docker_calls(self):
        runner = Mock()
        with self.assertRaisesRegex(ProtocolError, 'deadline'):
            release_container('a'*64, 't', 'token', runner, deadline=time.monotonic()-1)
        runner.assert_not_called()

    def test_malformed_inspection_never_removes(self):
        for raw in (b'[]', b'{"Config":null}', b'{"Config":{"Labels":[]}}'):
            runner = Mock(side_effect=[(0, ('a'*64).encode(), b''), (0, raw, b'')])
            with self.assertRaises(ProtocolError):
                release_container('a'*64, 't', 'token', runner)
            self.assertEqual(runner.call_count, 2)

    def test_host_release_replay_and_verified_claim_transfer(self):
        with tempfile.TemporaryDirectory() as folder:
            guard = HostGuard(Path(folder)/'guard', 't', 'a'*64)
            guard.claim(['npu:0'])
            runner = Mock(side_effect=lambda argv, **kw: (1 if argv[0]=='fuser' else 0,b'',b''))
            args = (guard, 'release1', ['a'*64], ['npu:0'], Path(folder)/'result.json')
            first = release_host_resources(*args, runner=runner)
            self.assertEqual(first, release_host_resources(*args, runner=runner))
            self.assertEqual(runner.call_count, 2)
            self.assertTrue(first['claims_released'])
            self.assertTrue(load_json(args[-1])['claims_released'])
            HostGuard(Path(folder)/'guard', 'next', 'b'*64).claim(['npu:0'])

    def test_partial_release_keeps_claims_and_attempts_other_containers(self):
        with tempfile.TemporaryDirectory() as folder:
            guard = HostGuard(Path(folder)/'guard', 't', 'a'*64)
            guard.claim(['npu:0', 'npu:1'])
            runner = Mock(side_effect=[(1, b'', b'error'), (0, b'', b'')])
            target = Path(folder)/'result.json'
            result = release_host_resources(guard, 'release1', ['a'*64, 'b'*64],
                                             ['npu:0', 'npu:1'], target, runner=runner)
            self.assertEqual(result['cleanup_status'], 'UNVERIFIED')
            self.assertFalse(result['claims_released'])
            self.assertEqual(runner.call_count, 2)
            self.assertEqual(load_json(target)['container_results'][1]['status'], 'ABSENT_VERIFIED')
            with self.assertRaisesRegex(ProtocolError, 'RESOURCE_BUSY'):
                HostGuard(Path(folder)/'guard', 'next', 'b'*64).claim(['npu:1'])
