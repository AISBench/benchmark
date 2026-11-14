"""Run tester tools in an existing, identity-bound container, never manage it."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid

from .durable import atomic_json
from .process_guard import Cancelled, remaining, run_logged
from .protocol import ProtocolError
from .task_log import diagnostic


class ContainerTester:
    def __init__(self, tester, folder, deadline):
        self.tester, self.folder = tester, Path(folder)
        self.container = tester['container']['id']
        self.deadline = deadline
        self.root = self.folder.parent / ('container-tools-' + uuid.uuid4().hex)
        self.runtime = self.root / 'runtime'
        self.source_runtime = Path(__file__).parent.parent
        item = json.loads(self.command(['docker', 'inspect', self.container]))[0]
        if (item['Id'] != self.container or item['Image'] != tester['container']['image']
                or not item['State']['Running']):
            raise ProtocolError('INVALID_INPUT: tester container identity or running state')
        # An exclusive task directory, with no image install or shared mount requirement.
        self.command(['docker', 'exec', self.container, tester['python'], '-c',
                      'import pathlib,sys; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=False); (p/".runtime").mkdir()', str(self.root)])
        self.command(['docker', 'cp', str(self.source_runtime), self.container + ':' + str(self.runtime)])
        self.command(['docker', 'exec', self.container, tester['python'], '-c',
                      'import pathlib,sys; pathlib.Path(sys.argv[1]).mkdir(parents=True,exist_ok=False)', str(self.folder)])

    def command(self, argv, timeout=None):
        result = subprocess.run(argv, capture_output=True, timeout=timeout or min(30, remaining(self.deadline)))
        diagnostic(self.folder/'container.log', json.dumps(argv), event='container.command', returncode=result.returncode)
        if result.returncode:
            diagnostic(self.folder/'container.log', result.stderr.decode(errors='replace'))
            raise ProtocolError('CONTAINER_TESTER_COMMAND_FAILED')
        return result.stdout

    def run(self, argv, log, deadline, cancelled, env=None):
        self.command(['docker', 'cp', str(self.folder) + '/.', self.container + ':' + str(self.folder)])
        token = uuid.uuid4().hex
        request = self.folder / ('container-command-' + token + '.json')
        done = str(self.root / (token + '.done.json'))
        cancel = str(self.root / (token + '.cancel'))
        actual = [arg.replace(str(self.source_runtime), str(self.runtime)) for arg in argv]
        atomic_json(request, dict(argv=actual, log=str(log), timeout=remaining(deadline),
                                  environment=self.tester.get('environment', {}), done=done, cancel=cancel,
                                  root=str(self.root)))
        self.command(['docker', 'cp', str(request), self.container + ':' + str(request)])
        bootstrap = 'import sys; sys.path.insert(0, ' + repr(str(self.runtime)) + '); from deploy_eval_v2.container_tester import worker; worker(sys.argv[1])'
        stop = threading.Event()
        errors = []
        def watch():
            while not stop.wait(.1):
                if cancelled.is_set():
                    try:
                        self.command(['docker', 'exec', self.container, self.tester['python'], '-c',
                                      'import pathlib,sys; pathlib.Path(sys.argv[1]).touch()', cancel], timeout=10)
                    except Exception as error:
                        errors.append(error)
                    return
        monitor = threading.Thread(target=watch, daemon=True)
        monitor.start()
        observer_error = None
        try:
            run_logged(['docker', 'exec', self.container, self.tester['python'], '-B', '-c', bootstrap, str(request)],
                       self.folder/'container.log', deadline+20, threading.Event())
        except Exception as error:
            observer_error = error
        finally:
            stop.set()
            monitor.join(timeout=11)
        # Docker client termination alone never proves in-container client cleanup.
        try:
            outcome = json.loads(self.command(['docker', 'exec', self.container, self.tester['python'], '-c',
                                               'import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())', done], timeout=10))
        except Exception as error:
            from .remote_evaluation import RemoteCleanupUnknown
            raise RemoteCleanupUnknown('Container tester cleanup unverified') from error
        # The worker has finished; persist diagnostics before acknowledging removal.
        for path in (self.root/'task.jsonl', self.folder/'task.jsonl'):
            raw = self.command(['docker', 'exec', self.container, self.tester['python'], '-c',
                                'import pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.stdout.write(p.read_text() if p.exists() else "")', str(path)], timeout=10)
            if raw:
                diagnostic(self.folder/'container.log', raw.decode(), event='container.framework.collected')
                self.command(['docker', 'exec', self.container, self.tester['python'], '-c',
                              'import pathlib,sys; pathlib.Path(sys.argv[1]).unlink()', str(path)], timeout=10)
        self.command(['docker', 'cp', self.container + ':' + str(self.folder) + '/.', str(self.folder)], timeout=30)
        if outcome.get('cleanup_status') != 'VERIFIED' or errors:
            from .remote_evaluation import RemoteCleanupUnknown
            raise RemoteCleanupUnknown('Container tester cleanup unverified')
        if cancelled.is_set() or outcome.get('cancelled'):
            raise Cancelled('Container tester cancelled; owned process group stopped')
        if outcome.get('error'):
            raise ProtocolError('CONTAINER_TESTER_FAILED: ' + outcome['error'])
        if observer_error:
            raise observer_error
        return outcome['returncode']


def worker(request_path):
    request = json.loads(Path(request_path).read_text())
    class Flag:
        def is_set(self):
            return Path(request['cancel']).exists()
    environment = dict(os.environ)
    environment.update(request['environment'])
    outcome = dict(cleanup_status='UNVERIFIED')
    try:
        outcome['returncode'] = run_logged(request['argv'], request['log'],
                                           time.monotonic()+request['timeout'], Flag(), env=environment)
        outcome['cleanup_status'] = 'VERIFIED'
    except (Cancelled, TimeoutError) as error:
        outcome.update(cleanup_status='VERIFIED', error=type(error).__name__, cancelled=isinstance(error, Cancelled))
    except Exception as error:
        outcome['error'] = type(error).__name__
    atomic_json(Path(request['done']), outcome)
