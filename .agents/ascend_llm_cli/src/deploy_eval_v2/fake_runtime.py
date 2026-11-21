"""Test-only executor; does not claim to create or release real resources."""


class FakeExecutor:
    def __init__(self, delay=0):
        self.delay = delay
        self.calls = []

    def __call__(self, command, operation, cancelled):
        self.calls.append(command['kind'])
        if cancelled.wait(self.delay):
            return 'operation.cancelled', 'CANCELLED', {'cleanup_status': 'SIMULATED'}
        kind = command['kind']
        data = {'service_id': 'fake-service', 'evidence_ref': 'results/fake.json', 'simulated': True}
        if kind == 'deploy':
            data.update(endpoint='http://fake.invalid',model='simulated-model')
            return 'deployment.ready', 'READY', data
        if kind == 'eval':
            data.update(case_id=command['case_id'], result_ref='results/fake.json',execution_status='SUCCEEDED', performance_validity='UNVERIFIED')
            return 'evaluation.completed', 'EVALUATED', data
        data['resource_results_ref']='results/fake.json'
        return 'release.completed', 'RELEASED', data
