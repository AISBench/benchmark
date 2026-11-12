"""Test-only coverage startup, inherited by real worker subprocesses."""
import os

if os.environ.get('COVERAGE_PROCESS_START'):
    import coverage
    coverage.process_startup()
