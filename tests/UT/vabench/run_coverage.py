"""Run scoped VABench UT coverage including child processes; no model downloads.

Usage from the source root: python3 tests/UT/vabench/run_coverage.py
Requires pytest and coverage==7.10.7 in the development environment.
"""
import os
import json
from pathlib import Path
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    output = root / 'test_reports/vabench'
    output.mkdir(parents=True, exist_ok=True)
    config = here / '.coveragerc'
    clean_env = dict(os.environ)
    clean_env.pop('COVERAGE_PROCESS_START', None)
    clean_env['COVERAGE_RCFILE'] = str(config)
    clean_env['COVERAGE_FILE'] = str(output / '.coverage')
    clean_env['PYTHONUTF8'] = '1'
    coverage_cmd = [sys.executable, '-m', 'coverage']
    subprocess.run(coverage_cmd + ['erase'], cwd=root, env=clean_env, check=True)
    collect_env = dict(clean_env, COVERAGE_PROCESS_START=str(config))
    collect_env['PYTHONPATH'] = os.pathsep.join([
        str(here / 'coverage_bootstrap'), str(root), clean_env.get('PYTHONPATH', '')])
    result = subprocess.run([sys.executable, '-m', 'pytest', str(here), '-q', '--tb=short'],
                            cwd=root, env=collect_env)
    subprocess.run(coverage_cmd + ['combine'], cwd=root, env=clean_env, check=True)
    for format_ in ('json', 'html', 'xml'):
        subprocess.run(coverage_cmd + [format_, '--fail-under=0'], cwd=root, env=clean_env, check=True)
    report = subprocess.run(coverage_cmd + ['report'], cwd=root, env=clean_env)
    data = json.loads((output / 'coverage.json').read_text(encoding='utf-8'))
    totals = data['totals']
    line_percent = 100 * totals['covered_lines'] / totals['num_statements']
    branch_percent = 100 * totals['covered_branches'] / totals['num_branches'] if totals['num_branches'] else 100
    low_files = [name for name, values in data['files'].items()
                 if values['summary']['percent_covered'] < 80]
    print(f'Statement coverage: {line_percent:.2f}%; branch coverage: {branch_percent:.2f}%')
    if low_files:
        print('Modules below 80%:', ', '.join(low_files))
    return result.returncode or report.returncode or int(line_percent < 80 or branch_percent < 80 or bool(low_files))


if __name__ == '__main__':
    raise SystemExit(main())
