"""Layering, packaging and legacy entry points are runtime contracts."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ALIASES = {
    'data': 'ais_bench.benchmark.utils.vabench.data',
    'engine': 'ais_bench.benchmark.tasks.vabench.engine',
    'worker': 'ais_bench.benchmark.tasks.vabench.worker',
    'resources': 'ais_bench.benchmark.tasks.vabench.resources',
    'results': 'ais_bench.benchmark.summarizers.vabench_results',
    'scoring': 'ais_bench.benchmark.summarizers.vabench_scoring',
    'cli': 'ais_bench.tools.vabench.cli',
    'environment': 'ais_bench.tools.vabench.environment',
}


@pytest.mark.parametrize('name,target', ALIASES.items())
def test_legacy_import_is_same_module(name, target):
    # Module identity matters: monkeypatches and shared state must not diverge.
    assert importlib.import_module('ais_bench.benchmark.vabench.' + name) is importlib.import_module(target)


@pytest.mark.parametrize('module', [
    'ais_bench.tools.vabench.cli', 'ais_bench.benchmark.vabench.cli',
    'ais_bench.tools.vabench.environment', 'ais_bench.benchmark.vabench.environment',
    'ais_bench.benchmark.tasks.vabench.worker', 'ais_bench.benchmark.vabench.worker',
])
def test_new_and_legacy_module_commands(module):
    result = subprocess.run([sys.executable, '-m', module, '--help'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout
    assert 'RuntimeWarning' not in result.stderr


def test_commands_import_without_framework_or_model_dependencies():
    code = '''
import importlib.abc
import sys
class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'mmengine', 'datasets', 'transformers'}:
            raise AssertionError('Unexpected heavy import: ' + fullname)
sys.meta_path.insert(0, BlockHeavyImports())
import ais_bench.tools.vabench.cli
import ais_bench.tools.vabench.environment
import ais_bench.benchmark.tasks.vabench.worker
import ais_bench.benchmark.vabench.cli
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_new_and_legacy_catalog_export_identical_data(tmp_path):
    source = Path(__file__).resolve().parents[4] / 'VABench'
    if not source.exists():
        pytest.skip('Local official checkout is not available')
    results = []
    for module in ('ais_bench.tools.vabench.cli', 'ais_bench.benchmark.vabench.cli'):
        output = tmp_path / (module + '.json')
        result = subprocess.run([sys.executable, '-m', module, 'catalog',
                                 '--source-root', str(source), '--subset', 'standard',
                                 '--output', str(output)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        results.append(json.loads(output.read_text(encoding='utf-8')))
    assert results[0] == results[1]
    assert results[0]['count'] == 778


def test_environment_plan_resolves_relocated_requirements_without_installing(tmp_path):
    from ais_bench.tools.vabench.environment import REQUIREMENTS
    from ais_bench.benchmark.utils.vabench.data import PACKAGE_ROOT, PROTOCOL_PATH
    assert PACKAGE_ROOT.name == 'ais_bench'
    assert PROTOCOL_PATH.is_file()
    assert (REQUIREMENTS / 'vabench.txt').is_file()
    plans = []
    for module in ('ais_bench.tools.vabench.environment', 'ais_bench.benchmark.vabench.environment'):
        result = subprocess.run([sys.executable, '-m', module, 'plan', '--source-root', 'unused',
                                 '--cache-dir', 'unused', '--work-dir', str(tmp_path / 'audit')],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        plans.append(json.loads(result.stdout))
    assert plans[0] == plans[1]
    assert not (tmp_path / 'audit').exists()
    assert any(str(REQUIREMENTS / 'vabench.txt') in step for step in plans[0]['commands'])


@pytest.mark.parametrize('file', [
    'benchmark/utils/vabench/data.py', 'benchmark/tasks/vabench/worker.py',
    'benchmark/summarizers/vabench_scoring.py', 'tools/vabench/environment.py',
    'benchmark/datasets/vabench.py', 'benchmark/tasks/vabench_eval.py',
])
def test_provenance_detects_changes_in_each_implementation_layer(tmp_path, monkeypatch, file):
    from ais_bench.benchmark.tasks.vabench import resources
    monkeypatch.setattr(resources, 'PACKAGE_ROOT', tmp_path)
    target = tmp_path / file
    target.parent.mkdir(parents=True)
    target.write_text('original', encoding='utf-8')
    before = resources.integration_signature()
    target.write_text('modified', encoding='utf-8')
    assert resources.integration_signature() != before


def test_protocol_is_in_source_distribution_manifest():
    from setuptools.command.egg_info import FileList
    files = FileList()
    files.findall()
    for line in Path('MANIFEST.in').read_text(encoding='utf-8').splitlines():
        files.process_template_line(line)
    paths = {Path(file).as_posix() for file in files.files}
    assert 'ais_bench/benchmark/utils/vabench/protocol.json' in paths
