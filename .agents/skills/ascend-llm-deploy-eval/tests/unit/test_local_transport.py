import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from remote_node import transport_argv


class LocalTransportTests(unittest.TestCase):
    def test_local_requires_matching_user_and_ip(self):
        result=SimpleNamespace(returncode=0,stdout=json.dumps([{'addr_info':[{'local':'192.0.2.10'}]}]))
        with patch('remote_node.sys.platform','linux'),patch('remote_node.getpass.getuser',return_value='root'),patch('remote_node.subprocess.run',return_value=result):
            self.assertEqual(transport_argv('root@192.0.2.10',['python3','a b.py']),['python3','a b.py'])
            self.assertEqual(transport_argv('other@192.0.2.10',['true'])[0],'ssh')
            self.assertEqual(transport_argv('root@192.0.2.11',['true'])[0],'ssh')
            self.assertEqual(transport_argv('root@unresolved.example',['true'])[0],'ssh')
