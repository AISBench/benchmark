"""Layering, packaging and dependency-isolated entry points are runtime contracts."""
import importlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


IMPLEMENTATIONS = {
    'ais_bench.benchmark.utils.vabench.data': 'benchmark/utils/vabench/data.py',
    'ais_bench.benchmark.tasks.vabench.engine': 'benchmark/tasks/vabench/engine.py',
    'ais_bench.benchmark.tasks.vabench.worker': 'benchmark/tasks/vabench/worker.py',
    'ais_bench.benchmark.tasks.vabench.resources': 'benchmark/tasks/vabench/resources.py',
    'ais_bench.benchmark.summarizers.vabench.results': 'benchmark/summarizers/vabench/results.py',
    'ais_bench.benchmark.summarizers.vabench.scoring': 'benchmark/summarizers/vabench/scoring.py',
    'ais_bench.tools.vabench.cli': 'tools/vabench/cli.py',
    'ais_bench.tools.vabench.environment': 'tools/vabench/environment.py',
    'ais_bench.tools.vabench.official_compare': 'tools/vabench/official_compare.py',
    'ais_bench.tools.vabench.official_prepare': 'tools/vabench/official_prepare.py',
}


@pytest.mark.parametrize('target,relative', IMPLEMENTATIONS.items())
def test_runtime_modules_resolve_to_their_layers(target, relative):
    from ais_bench.benchmark.utils.vabench.data import PACKAGE_ROOT
    module = importlib.import_module(target)
    assert Path(module.__file__).resolve() == PACKAGE_ROOT / relative


def test_no_top_level_compatibility_package_is_shipped():
    from setuptools import find_packages
    from ais_bench.benchmark.utils.vabench.data import PACKAGE_ROOT
    assert not (PACKAGE_ROOT / 'benchmark' / 'vabench').exists()
    assert all(package.split('.') != ['ais_bench', 'benchmark', 'vabench']
               for package in find_packages())


def test_only_the_manual_workflow_config_is_shipped():
    from ais_bench.benchmark.utils.vabench.data import PACKAGE_ROOT
    names = {p.name for p in (PACKAGE_ROOT / 'configs/vabench_examples').glob('eval_vabench*.py')}
    assert names == {'eval_vabench_custom.py'}


@pytest.mark.parametrize('command', ['run', 'summarize', 'compare'])
def test_cli_rejects_retired_alternate_workflows(command):
    from ais_bench.tools.vabench.cli import main
    with pytest.raises(SystemExit) as exc:
        main([command])
    assert exc.value.code == 2


@pytest.mark.parametrize('module', [
    'ais_bench.tools.vabench.cli',
    'ais_bench.tools.vabench.environment',
    'ais_bench.benchmark.tasks.vabench.worker',
    'ais_bench.tools.vabench.official_prepare',
])
def test_layered_module_commands(module):
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
import ais_bench.tools.vabench.official_prepare
import ais_bench.tools.vabench.official_compare
import ais_bench.benchmark.summarizers.vabench.results
import ais_bench.benchmark.summarizers.vabench.scoring
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_catalog_command_exports_the_pinned_set(tmp_path):
    source = Path(__file__).resolve().parents[4] / 'VABench'
    if not source.exists():
        pytest.skip('Local official checkout is not available')
    from ais_bench.benchmark.utils.vabench.data import catalog
    output = tmp_path / 'catalog.json'
    result = subprocess.run([sys.executable, '-m', 'ais_bench.tools.vabench.cli', 'catalog',
                             '--source-root', str(source), '--subset', 'standard',
                             '--output', str(output)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    exported = json.loads(output.read_text(encoding='utf-8'))
    assert exported['count'] == 778
    assert exported['entries'] == catalog(source)


def test_environment_plan_resolves_relocated_requirements_without_installing(tmp_path):
    from ais_bench.tools.vabench.environment import REQUIREMENTS
    from ais_bench.benchmark.utils.vabench.data import PACKAGE_ROOT, PROTOCOL_PATH
    assert PACKAGE_ROOT.name == 'ais_bench'
    assert PROTOCOL_PATH.is_file()
    assert (REQUIREMENTS / 'vabench.txt').is_file()
    result = subprocess.run([sys.executable, '-m', 'ais_bench.tools.vabench.environment',
                             'plan', '--source-root', 'unused', '--cache-dir', 'unused',
                             '--work-dir', str(tmp_path / 'audit')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert not (tmp_path / 'audit').exists()
    assert any(str(REQUIREMENTS / 'vabench.txt') in step for step in plan['commands'])


@pytest.mark.parametrize('file', [
    'benchmark/utils/vabench/data.py', 'benchmark/tasks/vabench/worker.py',
    'benchmark/summarizers/vabench/scoring.py', 'tools/vabench/environment.py',
    'benchmark/datasets/vabench/dataset.py', 'benchmark/tasks/vabench/eval.py',
    'benchmark/summarizers/vabench/__init__.py', 'benchmark/datasets/vabench/__init__.py',
    'tools/vabench/official_compare.py', 'tools/vabench/official_prepare.py',
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
    assert 'ais_bench/configs/vabench_examples/eval_vabench_custom.py' in paths
    assert 'ais_bench/configs/vabench_examples/mini10pct.json' in paths
    assert 'requirements/datasets/vabench.txt' in paths


def test_framework_class_exports_and_lazy_summarizer_attributes():
    from ais_bench.benchmark.datasets.vabench import VABenchDataset
    from ais_bench.benchmark.datasets.vabench.dataset import VABenchDataset as concrete_dataset
    from ais_bench.benchmark.tasks import VABenchEvalTask
    from ais_bench.benchmark.tasks.vabench.eval import VABenchEvalTask as concrete_task
    from ais_bench.benchmark.summarizers import vabench
    from ais_bench.benchmark.summarizers.vabench.summarizer import VABenchSummarizer
    assert VABenchDataset is concrete_dataset
    assert VABenchEvalTask is concrete_task
    assert vabench.VABenchSummarizer is VABenchSummarizer
    with pytest.raises(AttributeError, match='unknown'):
        getattr(vabench, 'unknown')
