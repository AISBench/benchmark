"""AISBench registry, task config, summary and packaging contracts."""
from pathlib import Path
import io
import pytest

from mmengine.config import Config, ConfigDict

from ais_bench.benchmark.tasks.vabench.eval import VABenchEvalTask
from ais_bench.benchmark.summarizers.vabench import VABenchSummarizer
from ais_bench.benchmark.utils.vabench.data import write_json


def test_config_and_partitioning(tmp_path):
    from ais_bench.benchmark.partitioners import NaivePartitioner
    cfg = Config.fromfile('ais_bench/configs/vabench_examples/eval_vabench_custom.py')
    cfg.work_dir = str(tmp_path)
    cfg.cli_args = {'mode': 'eval', 'debug': False}
    task_cfg = NaivePartitioner(keep_keys=['eval.runner.task.num_gpus'])(cfg)[0]
    task = VABenchEvalTask(task_cfg)
    assert task.num_gpus == 2
    assert len(task.dataset_cfgs) == 1
    command = task.get_command('config with spaces.py', '{task_cmd}')
    assert 'ais_bench.benchmark.tasks.vabench.eval' in command
    assert 'config with spaces.py' in command




def test_reader_config_and_dataset_use_the_same_selected_rows(tmp_path):
    from ais_bench.benchmark.datasets.vabench import VABenchDataset
    cfg = Config.fromfile('ais_bench/configs/vabench_examples/eval_vabench_custom.py')
    assert cfg.datasets[0].reader_cfg.input_columns == ['idx']
    assert cfg.datasets[0].eval_cfg.python is None
    records = [dict(idx='a'), dict(idx='b'), dict(idx='c')]
    write_json(tmp_path / 'manifest.json', dict(records=records))
    selected = VABenchDataset(tmp_path, reader_cfg={'test_range': '[1:2]'})
    assert selected.dataset['idx'] == ['b']


@pytest.mark.parametrize(('test_range', 'expected'), [
    ('[2:8]', ['2', '3', '4']),
    ('[::-1]', ['9', '8', '7']),
])
def test_size_partitioner_applies_partition_to_user_range(tmp_path, monkeypatch, test_range, expected):
    from ais_bench.benchmark.datasets.vabench import VABenchDataset
    from ais_bench.benchmark.partitioners.size import SizePartitioner
    from ais_bench.benchmark.utils.vabench.data import input_signature, select_prepared

    records = [dict(idx=str(i), category='scene', video_sha256='video', audio_sha256='audio')
               for i in range(10)]
    manifest = dict(records=records, evaluation_scope='custom', input_signature=input_signature(records))
    grouped = {'scene': [dict(video_list=[f'{i}.mp4']) for i in range(10)]}
    write_json(tmp_path / 'manifest.json', manifest)
    dataset_cfg = ConfigDict(abbr='vabench', reader_cfg={'test_range': test_range})
    partitioner = SizePartitioner(out_dir=str(tmp_path), max_task_size=3)
    size = len(VABenchDataset(tmp_path, reader_cfg=dataset_cfg.reader_cfg).dataset)
    monkeypatch.setattr(partitioner, 'get_cost', lambda *args, **kwargs: (size, 1))
    split = partitioner.split_dataset(dataset_cfg)[0]
    dataset = VABenchDataset(tmp_path, reader_cfg=split.reader_cfg)
    selected, metadata, _ = select_prepared(manifest, grouped, split.reader_cfg.test_range)
    assert dataset.dataset['idx'] == expected
    assert [record['idx'] for record in selected['records']] == expected
    assert [row['video_list'][0] for row in metadata['scene']] == [f'{idx}.mp4' for idx in expected]




def test_task_preserves_incomplete_report(tmp_path, monkeypatch):
    import ais_bench.benchmark.tasks.vabench.eval as task_module
    report = dict(metrics={'second_clap': dict(scored=0, failed=1, missing=0)}, details={})
    destination = tmp_path / 'results/test/vabench.result'
    monkeypatch.setattr(task_module, 'get_infer_output_path', lambda *a: str(destination))

    def run(prepared, source, cache, output, **kwargs):
        assert output == destination.with_name('vabench_artifacts')
        return report

    monkeypatch.setattr(task_module, 'run', run)
    prepared = tmp_path / 'prepared'
    write_json(prepared / 'manifest.json', dict(records=[]))
    cfg = ConfigDict(models=[dict(abbr='test')], datasets=[[
        dict(abbr='vabench', path=str(prepared),
             eval_cfg=dict(source_root='source', cache_dir='cache'))]],
        work_dir=str(tmp_path), cli_args={})
    import pytest
    with pytest.raises(RuntimeError, match='incomplete'):
        VABenchEvalTask(cfg).run()
    assert destination.exists()


@pytest.mark.parametrize('has_valid_metric', [False, True])
def test_task_lsa_omission_needs_other_valid_output(tmp_path, monkeypatch, has_valid_metric):
    import ais_bench.benchmark.tasks.vabench.eval as task_module
    report = dict(
        requested_dimensions=['second_lsa'],
        metrics={'second_lsa': dict(expected=1, scored=0, missing=1, failed=0,
                                    not_evaluated=0, backend_issue_count=0,
                                    status='incomplete')},
        details={})
    if has_valid_metric:
        report['requested_dimensions'].append('second_clap')
        report['metrics']['second_clap'] = dict(expected=1, scored=1, status='complete')
    monkeypatch.setattr(task_module, 'run', lambda *a, **kw: report)
    prepared = tmp_path / 'prepared'
    write_json(prepared / 'manifest.json', dict(records=[]))
    cfg = ConfigDict(models=[dict(abbr='test')], datasets=[[
        dict(abbr='vabench', path=str(prepared),
             eval_cfg=dict(source_root='source', cache_dir='cache'))]],
        work_dir=str(tmp_path), cli_args={})
    if has_valid_metric:
        VABenchEvalTask(cfg).run()
    else:
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
    assert 'vabench/total' not in parsed['test']
    assert len(parsed['test']) == 15
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
