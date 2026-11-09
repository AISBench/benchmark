# Harbor-related unit tests

Run from the benchmark repository root, in the complete AISBench/Harbor scheduler
Python environment. Install `pytest` and `pytest-cov` first. These tests use mocks
and local fixtures; they do not call a paid model API or require Docker tasks.

Use the existing runner instead of maintaining a separate shell wrapper:

```bash
mkdir -p test_reports/harbor
PYTEST_ADDOPTS="--cov-fail-under=0 --junitxml=$PWD/test_reports/harbor/junit.xml" \
python tests/run_tests.py tests/UT \
  --specific-tests \
  tests/UT/tasks/custom_tasks/test_harbor_agent_task.py \
  tests/UT/tasks/custom_tasks/test_harbor_task.py \
  tests/UT/summarizers/test_harbor.py \
  tests/UT/runners/test_harbor_runner.py \
  tests/UT/runners/test_harbor_monitor.py \
  tests/UT/test_run_tests.py \
  --output test_reports/harbor -v
```

`--specific-tests` selects only those files. Without it, the runner executes the
entire supplied test directory. The explicit `--cov-fail-under=0` applies only to
this targeted run: the repository-wide 80% coverage gate remains unchanged. Keep
the measured coverage reports, and do not claim whole-repository coverage from
these selected cases. Check the process exit code and JUnit failures, errors and
skips; retain the logs, coverage outputs and tested source commit.

Select a public release tag or a fixed commit and record it with the results.
Validation of a development commit does not certify a different release.
