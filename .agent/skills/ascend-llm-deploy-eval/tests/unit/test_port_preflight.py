"""Linux socket-level preflight checks; ephemeral loopback ports, no model service."""
import socket,subprocess,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from run import port_check_program

@unittest.skipUnless(sys.platform.startswith('linux'),'Linux TCP reuse semantics')
class PortCheckTests(unittest.TestCase):
    def probe(self,port):
        return subprocess.run([sys.executable,'-c',port_check_program([port])],capture_output=True,text=True,timeout=5)
    def test_time_wait_allows_restart_but_bare_bind_fails(self):
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',0));port=server.getsockname()[1];server.listen(1)
            with socket.create_connection(('127.0.0.1',port)) as client:
                peer,_=server.accept()
                peer.close()  # server is the active closer, leaving server-port TIME_WAIT
                self.assertEqual(client.recv(1),b'')
        with socket.socket() as bare:
            with self.assertRaises(OSError):bare.bind(('0.0.0.0',port))
        result=self.probe(port);self.assertEqual(result.returncode,0,result.stderr)
    def test_listener_remains_blocked_with_and_without_reuseaddr(self):
        for reuse in (0,1):
            with self.subTest(reuse=reuse),socket.socket() as server:
                server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,reuse)
                server.bind(('127.0.0.1',0));port=server.getsockname()[1];server.listen(1)
                result=self.probe(port)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('TCP port %s unavailable'%port,result.stderr)
                with socket.create_connection(('127.0.0.1',port)):
                    peer,_=server.accept();peer.close()
    def test_nonreusable_bound_socket_is_blocked(self):
        with socket.socket() as bound:
            bound.bind(('127.0.0.1',0))
            self.assertNotEqual(self.probe(bound.getsockname()[1]).returncode,0)

if __name__=='__main__':unittest.main()
