"""AISBench registry, task config, summary and packaging contracts."""
from pathlib import Path
import io

from mmengine.config import Config, ConfigDict

from ais_bench.benchmark.tasks.vabench_eval import VABenchEvalTask
from ais_bench.benchmark.summarizers.vabench import VABenchSummarizer
from ais_bench.benchmark.utils.vabench.data import write_json


def test_config_and_partitioning(tmp_path):
    from ais_bench.benchmark.partitioners import NaivePartitioner
    cfg = Config.fromfile('ais_bench/configs/vabench_examples/eval_vabench.py')
    cfg.work_dir = str(tmp_path)
    cfg.cli_args = {'mode': 'eval', 'debug': False}
    task_cfg = NaivePartitioner(keep_keys=['eval.runner.task.num_gpus'])(cfg)[0]
    task = VABenchEvalTask(task_cfg)
    assert task.num_gpus == 1
    assert len(task.dataset_cfgs) == 1
    command = task.get_command('config with spaces.py', '{task_cmd}')
    assert 'ais_bench.benchmark.tasks.vabench_eval' in command
    assert 'config with spaces.py' in command


def test_smoke_config_selects_only_clap():
    cfg = Config.fromfile('ais_bench/configs/vabench_examples/eval_vabench_smoke.py')
    assert cfg.datasets[0].eval_cfg.dimensions == ['second_clap']
    assert 'smoke' in cfg.models[0].abbr
    assert 'smoke' in cfg.datasets[0].abbr


def test_published_custom_and_smoke_templates_are_loadable():
    custom = Config.fromfile('ais_bench/configs/vabench_examples/eval_vabench_custom.py')
    smoke = Config.fromfile('ais_bench/configs/vabench_examples/eval_vabench_smoke_clap.py')
    assert custom.datasets[0].path == '/workspace/prepared_all_official_v2'
    assert custom.eval.runner.task.num_gpus == 2
    assert smoke.datasets[0].eval_cfg.dimensions == ['second_clap']


def test_task_preserves_incomplete_report(tmp_path, monkeypatch):
    import ais_bench.benchmark.tasks.vabench_eval as task_module
    report = dict(metrics={'second_clap': dict(scored=0, failed=1, missing=0)}, details={})
    monkeypatch.setattr(task_module, 'run', lambda *a, **kw: report)
    prepared = tmp_path / 'prepared'
    write_json(prepared / 'manifest.json', dict(records=[]))
    cfg = ConfigDict(models=[dict(abbr='test')], datasets=[[
        dict(abbr='vabench', path=str(prepared),
             eval_cfg=dict(source_root='source', cache_dir='cache'))]],
        work_dir=str(tmp_path), cli_args={})
    import pytest
    with pytest.raises(RuntimeError, match='incomplete'):
        VABenchEvalTask(cfg).run()
    assert (tmp_path / 'results/test/vabench.json').exists()


def test_native_summary_with_normalization_and_unavailable_total(tmp_path):
    cfg = ConfigDict(models=[dict(abbr='test')],
                     datasets=[dict(abbr='vabench', path='prepared')], work_dir=str(tmp_path))
    write_json(tmp_path / 'results/test/vabench.json', dict(
        metrics={'second_desync': dict(mean=0.2, expected=10, scored=10, missing=0,
                                       failed=0, scenes={'scene': {'mean': 0.2}})},
        directions={'second_desync': 'lower'}, details={}))
    summarizer = VABenchSummarizer(cfg)
    raw, parsed, metrics, modes = summarizer._pick_up_results()
    assert parsed['test']['vabench/second_desync']['mean'] == 0.2
    assert raw['test']['vabench/second_desync']['direction'] == 'lower'
    assert parsed['test']['vabench/second_desync']['normalized_score'] == 90
    assert parsed['test']['vabench/total']['normalized_score'] == '—'
    assert len(parsed['test']) == 16
    table = summarizer._format_table(parsed, metrics, modes,
                                     required_dataset_abbrs=summarizer.dataset_abbrs)
    assert table
    row = next(r for r in table if len(r) > 1 and r[1] == 'vabench/second_desync')
    assert row[2] == 'lower'
    # A failed dimension has an explicit N/A mean, never a fabricated zero.
    file = tmp_path / 'results/test/vabench.json'
    from ais_bench.benchmark.utils.vabench.data import read_json
    report = read_json(file)
    report['metrics']['second_desync'].update(mean=None, scored=0, failed=10)
    write_json(file, report)
    raw, parsed, metrics, modes = summarizer._pick_up_results()
    table = summarizer._format_table(parsed, metrics, modes)
    row = next(r for r in table if len(r) > 1 and r[1] == 'vabench/second_desync')
    assert row[3] == '—' and row[-1] == 'incomplete'
