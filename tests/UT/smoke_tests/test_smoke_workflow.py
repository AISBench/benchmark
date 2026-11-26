"""Execute only local workflow result-check snippets, never a smoke install."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = yaml.safe_load((ROOT / '.github/workflows/run_smoke_test_pr_and_daily.yml').read_text(encoding='utf-8'))
STEPS = {step['name']: step for step in WORKFLOW['jobs']['run_smoke_test_cases']['steps']}
SOURCE = ROOT / 'smoke_tests/framework/smoke_timing.py'
spec = importlib.util.spec_from_file_location('workflow_timing_fixture', SOURCE)
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'requires POSIX Bash')
class SmokeWorkflowTest(unittest.TestCase):
    def test_final_check_requires_explicit_success_and_both_false_flags(self):
        values = ['steps.run_cases.outcome', 'steps.get_results.outcome',
                  'steps.run_cases.outputs.test_execution_failed', 'steps.get_results.outputs.test_case_failed']
        cases = [('success', 'success', 'false', 'false'),
                 ('cancelled', 'success', '', 'false'), ('success', 'skipped', 'false', ''),
                 ('success', 'success', '', ''), ('success', 'success', 'false', 'true')]
        for index, case in enumerate(cases):
            with self.subTest(case=case):
                script = STEPS['Final Check']['run']
                for name, value in zip(values, case):
                    script = script.replace('${{ ' + name + ' }}', value)
                result = subprocess.run(['bash', '-e', '-c', script], text=True, capture_output=True, timeout=5)
                self.assertEqual(result.returncode == 0, index == 0, result.stdout + result.stderr)

    def test_shell_errexit_does_not_lose_failed_execution_output(self):
        for command, expected in [('true', 'false'), ('false', 'true')]:
            with self.subTest(command=command), tempfile.TemporaryDirectory() as root:
                output = Path(root) / 'output'
                script = STEPS['Run Smoke Cases']['run'].replace(
                    'bash smoke_tests/scripts/run_steps.sh -t short', command)
                result = subprocess.run(['bash', '-e', '-c', script], text=True, capture_output=True,
                                        env=dict(os.environ, GITHUB_OUTPUT=str(output)), timeout=5)
                self.assertIn('test_execution_failed=' + expected, output.read_text())
                self.assertEqual(result.returncode == 0, command == 'true')

    def test_result_check_uses_this_run_and_requires_complete_timing(self):
        for status in ('missing', 'incomplete', 'success'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                target = root / 'smoke_tests/framework/smoke_timing.py'
                target.parent.mkdir(parents=True)
                shutil.copyfile(SOURCE, target)
                workspace = root / 'this-run'
                if status != 'missing':
                    records = timing.CaseTimings(workspace, 1)
                    records.record({'case_name': 'case'}, {'success': True, 'execution_time': 1})
                    if status == 'success':
                        records.finish(1)
                output = root / 'output'
                env = dict(os.environ, PROJECT_WORKSPACE_PATH=str(workspace), GITHUB_OUTPUT=str(output),
                           PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
                result = subprocess.run(['bash', '-e', '-c', STEPS['Get Result files']['run']],
                                        cwd=root, env=env, text=True, capture_output=True, timeout=10)
                expected = 'false' if status == 'success' else 'true'
                self.assertIn('test_case_failed=' + expected, output.read_text(), result.stderr)
                self.assertEqual(result.returncode == 0, status != 'missing', result.stderr)
                self.assertFalse((root / 'smoke_tests/RunWorkspace/empty_dir').exists())

    def test_unit_test_job_installs_declared_simulator_dependencies(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/run-ut-on-pr-py.yml').read_text(encoding='utf-8'))
        steps = workflow['jobs']['pr_run_test']['steps']
        run_step = next(step for step in steps if step['name'] == 'Run Test')
        self.assertIn('-r requirements/test.txt', run_step['run'])
        self.assertIn('Flask', (ROOT / 'requirements/test.txt').read_text(encoding='utf-8'))

    def test_scheduled_smoke_forces_full_workload_before_execution(self):
        step = STEPS['Force full workload for scheduled smoke']
        self.assertEqual(step['if'], "github.event_name == 'schedule'")
        self.assertLess(list(STEPS).index(step['name']), list(STEPS).index('Run Smoke Cases'))
        with tempfile.TemporaryDirectory() as root:
            environment_file = Path(root) / 'github-env'
            environment_file.write_text('AISBENCH_SMOKE_FAST=1\n', encoding='utf-8')
            result = subprocess.run(['bash', '-e', '-c', step['run']], text=True,
                                    capture_output=True, timeout=5,
                                    env=dict(os.environ, GITHUB_ENV=str(environment_file)))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(environment_file.read_text(encoding='utf-8').splitlines()[-1],
                             'AISBENCH_SMOKE_FAST=0')

    def test_diagnostics_are_attempt_scoped_and_uploaded_on_failure(self):
        job = WORKFLOW['jobs']['run_smoke_test_cases']
        workspace = job['env']['PROJECT_WORKSPACE_PATH']
        self.assertIn('github.run_id', workspace)
        self.assertIn('github.run_attempt', workspace)
        self.assertEqual(STEPS['Get Result files']['if'], 'always()')
        self.assertIn('case_timings.jsonl', STEPS['Upload Results']['with']['path'])
        self.assertEqual(STEPS['Upload Results']['if'], 'always()')
        self.assertIn("github.event_name == 'pull_request'", WORKFLOW['env']['AISBENCH_SMOKE_FAST'])


if __name__ == '__main__':
    unittest.main()
