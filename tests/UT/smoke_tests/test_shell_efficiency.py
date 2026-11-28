"""Exercise real Bash functions with deterministic local command doubles."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[3] / 'smoke_tests' / 'scripts'


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'requires POSIX Bash')
class TestShellEfficiency(unittest.TestCase):
    def bash(self, filename, body):
        result = subprocess.run(['bash', '-c', 'source "$1"; shift; ' + body,
                                 'test', str(SCRIPTS / filename)],
                                text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_ci_git_summary_does_not_launch_python_or_tput(self):
        output = self.bash('git_utils.sh', r'''
            git() { printf 'git %s\n' "$*"; }
            python3() { echo UNEXPECTED_PYTHON; return 99; }
            tput() { echo UNEXPECTED_TPUT; return 99; }
            get_git_info .
        ''')
        self.assertIn('Git repository:', output)
        self.assertIn('log --oneline --no-decorate --color=never -3', output)
        self.assertNotIn('UNEXPECTED_', output)

    def test_healthy_service_is_probed_immediately_with_bounded_curl(self):
        output = self.bash('utils_funcs.sh', r'''
            VIRTUAL_SERVICE_PID=123
            AISBENCH_SMOKE_SERVICE_IP=127.0.0.1
            AISBENCH_SMOKE_SERVICE_PORT=8080
            SMOKE_SERVICE_READY_TIMEOUT=5
            ps() { return 0; }
            curl() {
                [[ " $* " == *' --connect-timeout 1 '* ]] || return 98
                [[ " $* " == *' --max-time 2 '* ]] || return 98
                printf '{"active_connections": "123: 0"}'
            }
            sleep() { echo UNEXPECTED_SLEEP; return 99; }
            check_service_ready && [[ "$SERVICE_STATUS" == 0 ]]
        ''')
        self.assertNotIn('UNEXPECTED_SLEEP', output)

    def test_failed_http_cannot_pass_with_a_valid_body_and_wait_is_bounded(self):
        output = self.bash('utils_funcs.sh', r'''
            VIRTUAL_SERVICE_PID=123
            AISBENCH_SMOKE_SERVICE_IP=127.0.0.1
            AISBENCH_SMOKE_SERVICE_PORT=8080
            SMOKE_SERVICE_READY_TIMEOUT=1
            ps() { return 0; }
            curl() { printf '{"active_connections":"123: 0"}'; return 22; }
            sleep() { SECONDS=$((SECONDS + 1)); }
            if check_service_ready; then exit 99; fi
            [[ "$SERVICE_STATUS" == -1 ]]
        ''')
        self.assertIn('服务启动超时', output)

    def test_simulator_uses_selected_interpreter_not_gunicorn_shebang(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            python = root / 'selected-python'
            python.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\n')
            python.chmod(0o755)
            (root / 'config.sh').write_text('PROCESS_NUM=1\nIP=127.0.0.1\nPORT=8080\n')
            launcher = SCRIPTS.parents[1] / 'tools/infer_serve_simulator/launch_service.sh'
            result = subprocess.run(['bash', str(launcher)], cwd=root, text=True,
                                    capture_output=True, timeout=5,
                                    env=dict(os.environ, AISBENCH_SMOKE_PYTHON=str(python)))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(result.stdout.startswith('-m\ngunicorn\n'), result.stdout)

    def test_invalid_deadline_fails_before_probe(self):
        output = self.bash('utils_funcs.sh', r'''
            SMOKE_SERVICE_READY_TIMEOUT=invalid
            curl() { echo UNEXPECTED_CURL; }
            if check_service_ready; then exit 99; fi
        ''')
        self.assertIn('positive integer', output)
        self.assertNotIn('UNEXPECTED_CURL', output)


if __name__ == '__main__':
    unittest.main()
