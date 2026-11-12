import socket
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from preflight import probe_port, PortUnavailable

class PortRecovery(unittest.TestCase):
    def test_live_listener_is_not_treated_as_free(self):
        with socket.socket() as listener:
            listener.bind(('0.0.0.0',0))
            listener.listen()
            port=listener.getsockname()[1]
            with self.assertRaises(PortUnavailable): probe_port(port,timeout=0)
        probe_port(port,timeout=0)

    @unittest.skipIf(sys.platform=='win32','Linux server socket semantics')
    def test_time_wait_does_not_block_redeployment(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',0)); server.listen()
            port=server.getsockname()[1]
            with socket.create_connection(('127.0.0.1',port)) as client:
                peer,_=server.accept()
                peer.close()
                client.recv(1)
        probe_port(port,timeout=0)
