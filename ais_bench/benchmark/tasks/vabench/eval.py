"""AISBench eval-only VABench task with an isolated official backend."""
import argparse
import os.path as osp
from pathlib import Path
import shlex
import subprocess
import sys
import threading

from mmengine.config import Config

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.utils.core.abbr import (get_infer_output_path,
                                               task_abbr_from_cfg)
from ais_bench.benchmark.tasks.vabench.engine import run, successful
from ais_bench.benchmark.utils.vabench.data import (
    read_json, select_indices, write_json)
from ais_bench.benchmark.utils.vabench import DIMENSIONS, MLLM_DIMENSIONS


def _metric_group_count(dimensions):
    return sum(d not in MLLM_DIMENSIONS for d in dimensions) + int(
        any(d in MLLM_DIMENSIONS for d in dimensions))


@TASKS.register_module()
class VABenchEvalTask(BaseTask):
    name_prefix = 'VABenchEval'
    log_subdir = 'logs/eval'
    output_subdir = 'results'

    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_gpus = int(cfg.get('eval', {}).get('runner', {}).get('task', {}).get('num_gpus', 1))

    def get_command(self, cfg_path, template):
        command = [sys.executable, '-m', 'ais_bench.benchmark.tasks.vabench.eval', str(cfg_path)]
        rendered = subprocess.list2cmdline(command) if sys.platform == 'win32' else shlex.join(command)
        return template.format(task_cmd=rendered)

    def run(self, task_state_manager=None):
        failed = []
        for dataset in self.dataset_cfgs:
            options = dict(dataset.get('eval_cfg', {}))
            source = options.pop('source_root')
            cache = options.pop('cache_dir')
            prepared = dataset['path']
            destination = get_infer_output_path(
                self.model_cfg, dataset, osp.join(self.work_dir, self.output_subdir))
            records = read_json(osp.join(prepared, 'manifest.json'))['records']
            test_range = (dataset.get('reader_cfg') or {}).get('test_range')
            if test_range is None:
                # Direct task construction may bypass the CLI's reader_cfg fill.
                num_prompts = self.cli_args.get('num_prompts')
                if num_prompts:
                    test_range = f'[:{num_prompts}]'
            sample_count = len(select_indices(len(records), test_range))
            if test_range is not None:
                self.logger.info(
                    f'VABench reader_cfg.test_range={test_range!r}: '
                    f'{sample_count}/{len(records)} samples')
            group_count = _metric_group_count(options.get('dimensions', DIMENSIONS))
            if task_state_manager:
                task_state_manager.update_task_state(dict(
                    status='evaluating', total_count=group_count, finish_count=0,
                    progress_description='none',
                    other_kwargs={'dimension': 'starting', 'sample_count': sample_count}))

            def _on_progress(done, total, dimensions):
                if task_state_manager:
                    task_state_manager.update_task_state(dict(
                        status='evaluating', total_count=total, finish_count=done,
                        progress_description='none',
                        other_kwargs={'dimension': ','.join(dimensions), 'sample_count': sample_count}))

            options.pop('test_range', None)
            result_path = Path(destination)
            artifacts = result_path.with_name(f'{result_path.stem}_artifacts')
            report = run(prepared, source, cache, artifacts,
                         progress=_on_progress if task_state_manager else None,
                         test_range=test_range, **options)
            write_json(destination, report)
            if not successful(report):
                failed.append(dataset.get('abbr', prepared))
            if task_state_manager:
                task_state_manager.update_task_state(dict(
                    status='evaluating', total_count=group_count, finish_count=group_count,
                    progress_description='none',
                    other_kwargs={'dimension': 'done', 'sample_count': sample_count}))
        if failed:
            raise RuntimeError(f'VABench incomplete; reports preserved: {failed}')

    def get_output_paths(self, file_extension='json'):
        return [get_infer_output_path(self.model_cfg, dataset,
                                     osp.join(self.work_dir, self.output_subdir), file_extension)
                for dataset in self.dataset_cfgs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    manager = TaskStateManager(tmp_path=osp.join(cfg['work_dir'], 'status_tmp'),
                               task_name=task_abbr_from_cfg(cfg),
                               is_debug=cfg.get('cli_args', {}).get('debug', False))
    monitor = threading.Thread(target=manager.launch, daemon=True)
    monitor.start()
    try:
        manager.update_task_state(dict(status='start', task_log_path=osp.join(
            'logs/eval', f'{task_abbr_from_cfg(cfg)}.out')))
        VABenchEvalTask(cfg).run(manager)
    except Exception:
        manager.update_task_state({'status': 'error'})
        raise
    else:
        manager.update_task_state({'status': 'finish'})
    finally:
        monitor.join(timeout=10)


if __name__ == '__main__':
    main()
