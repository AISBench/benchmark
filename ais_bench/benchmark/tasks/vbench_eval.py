"""VBench 1.0 evaluation task for video/image quality metrics on GPU or NPU."""
import argparse
import json
import os
import os.path as osp
import statistics
import sys
import threading
import time

from mmengine.config import Config, ConfigDict

from ais_bench.benchmark.registry import TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.utils.core.abbr import (
    dataset_abbr_from_cfg,
    get_infer_output_path,
    model_abbr_from_cfg,
    task_abbr_from_cfg,
)
from ais_bench.benchmark.utils.logging import AISLogger
from typing import List


@TASKS.register_module()
class VBenchEvalTask(BaseTask):
    """VBench 1.0 evaluation task. Runs VBench metrics on a folder of videos."""

    name_prefix = 'VBenchEval'
    log_subdir = 'logs/eval'
    output_subdir = 'results'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        self.num_gpus = 1

    def get_command(self, cfg_path, template):
        sys.path.insert(0, os.getcwd())
        script_path = __file__
        python = sys.executable
        command = f'{python} {script_path} {cfg_path}'
        return template.format(task_cmd=command)

    def _ensure_vbench_in_path(self):
        """Prepend third_party and third_party/detectron2 to sys.path so vbench and detectron2 resolve to ais_bench copy."""
        # __file__ = ais_bench/benchmark/tasks/vbench_eval.py -> pkg_root = ais_bench
        pkg_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
        third_party = osp.join(pkg_root, 'third_party')
        detectron2_parent = osp.join(third_party, 'detectron2')
        for path in (third_party, detectron2_parent):
            if path not in sys.path:
                sys.path.insert(0, path)

    def _infer_mode(self, dataset_cfg: ConfigDict, eval_cfg: ConfigDict) -> str:
        """Infer VBench mode when not explicitly provided in eval_cfg."""
        mode = eval_cfg.get('mode')
        if mode:
            return mode
        if eval_cfg.get('category'):
            return 'vbench_category'
        has_prompt = bool(
            eval_cfg.get('prompt_file')
            or eval_cfg.get('prompt_list')
            or dataset_cfg.get('prompt_list')
        )
        if has_prompt:
            return 'custom_input'
        return 'vbench_standard'

    def _wrap_results(self, raw_results: dict) -> dict:
        """Convert raw VBench per-dimension results to {accuracy, details} schema."""
        details = {}
        scores = []
        for dim, value in raw_results.items():
            dim_detail = {}
            if isinstance(value, dict):
                dim_detail = value
                score = value.get('score') or value.get('mean_score')
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                score, video_results = value
                dim_detail = {'score': score, 'video_results': video_results}
            else:
                dim_detail = {'value': value}
                score = None
            if isinstance(score, (int, float)):
                scores.append(score)
            details[dim] = dim_detail
        accuracy = statistics.mean(scores) if scores else 0.0
        return {'accuracy': accuracy * 100, 'details': details}

    def run(self, task_state_manager: TaskStateManager | None = None):
        self._ensure_vbench_in_path()
        from vbench import VBench, set_progress_callback
        from vbench.distributed import dist_init, get_rank, dist_destroy

        for dataset_cfg in self.dataset_cfgs:
            eval_cfg = dataset_cfg.get('eval_cfg') or {}
            # videos_path: required, from path or videos_path
            videos_path = dataset_cfg.get('videos_path') or dataset_cfg.get('path')
            if not videos_path or not osp.isdir(videos_path):
                raise ValueError(
                    f"VBench dataset must have 'path' or 'videos_path' pointing to a video directory, got: {videos_path}"
                )
            # device: cuda | npu
            device_str = eval_cfg.get('device') or 'cuda'
            if device_str not in ('cuda', 'npu'):
                device_str = 'cuda'
            # full_json_dir: VBench full info json
            full_json_dir = dataset_cfg.get('full_json_dir') or eval_cfg.get('full_json_dir')
            if not full_json_dir or not osp.isfile(full_json_dir):
                # default under third_party/vbench
                pkg_root = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
                default_full = osp.join(pkg_root, 'third_party', 'vbench', 'VBench_full_info.json')
                if osp.isfile(default_full):
                    full_json_dir = default_full
                else:
                    raise FileNotFoundError(
                        f"VBench full_info json not found. Set dataset full_json_dir or place VBench_full_info.json at {default_full}"
                    )
            # dimension_list
            dimension_list = dataset_cfg.get('dimension_list') or eval_cfg.get('dimension_list')
            if not dimension_list:
                dimension_list = [
                    'subject_consistency', 'background_consistency', 'aesthetic_quality',
                    'imaging_quality', 'object_class', 'multiple_objects', 'color',
                    'spatial_relationship', 'scene', 'temporal_style', 'overall_consistency',
                    'human_action', 'temporal_flickering', 'motion_smoothness', 'dynamic_degree',
                    'appearance_style',
                ]
            # output dir: work_dir/results/<model_abbr>/
            model_abbr = model_abbr_from_cfg(self.model_cfg)
            dataset_abbr = dataset_abbr_from_cfg(dataset_cfg)
            output_dir = osp.join(self.work_dir, self.output_subdir, model_abbr)
            os.makedirs(output_dir, exist_ok=True)

            dist_init(device=device_str)
            if get_rank() == 0:
                self.logger.info(
                    f"VBench eval: videos_path={videos_path}, device={device_str}, "
                    f"dimensions={len(dimension_list)}, output_dir={output_dir}"
                )

            import torch
            device = torch.device(device_str)
            vbench = VBench(device, full_json_dir, output_dir)

            # 注册进度回调，将 VBench 内部的维度进度映射到 TaskStateManager
            if task_state_manager is not None:
                def _on_progress(dimension: str, finished: int, total: int, video_path: str | None = None, **_):
                    # 仅在 rank0 上上报，避免多卡重复
                    if get_rank() != 0:
                        return
                    state = {
                        "status": "evaluating",
                        "total_count": total,
                        "finish_count": finished,
                        "other_kwargs": {
                            "dimension": dimension,
                        },
                    }
                    task_state_manager.update_task_state(state)

                set_progress_callback(_on_progress)

            # Infer mode if not explicitly provided
            mode = self._infer_mode(dataset_cfg, eval_cfg)

            prompt_list = dataset_cfg.get('prompt_list') or eval_cfg.get('prompt_list') or []
            prompt_file = eval_cfg.get('prompt_file')
            if prompt_file and osp.isfile(prompt_file):
                with open(prompt_file, 'r') as f:
                    prompt_list = json.load(f)
                assert isinstance(prompt_list, dict), "prompt_file must be JSON dict {video_path: prompt}"

            kwargs = {}
            if eval_cfg.get('category'):
                kwargs['category'] = eval_cfg['category']
            if eval_cfg.get('imaging_quality_preprocessing_mode'):
                kwargs['imaging_quality_preprocessing_mode'] = eval_cfg['imaging_quality_preprocessing_mode']

            try:
                raw_results = vbench.evaluate(
                    videos_path=videos_path,
                    name=dataset_abbr,
                    prompt_list=prompt_list,
                    dimension_list=dimension_list,
                    local=eval_cfg.get('load_ckpt_from_local', False),
                    read_frame=eval_cfg.get('read_frame', False),
                    mode=mode,
                    **kwargs,
                )

                if get_rank() == 0:
                    # Wrap raw VBench results to {accuracy, details} schema and save.
                    wrapped = self._wrap_results(raw_results)
                    final_out = get_infer_output_path(
                        self.model_cfg,
                        dataset_cfg,
                        osp.join(self.work_dir, self.output_subdir),
                    )
                    os.makedirs(osp.dirname(final_out), exist_ok=True)
                    with open(final_out, 'w', encoding='utf-8') as f:
                        json.dump(wrapped, f, ensure_ascii=False, indent=4)
                    self.logger.info(f"VBench wrapped results saved to {final_out}")
            finally:
                dist_destroy()

    def get_output_paths(self, file_extension: str = "json") -> List[str]:
        """Paths to wrapped VBench result files: results/<model_abbr>/<dataset_abbr>.json."""
        paths = []
        for dataset_cfg in self.dataset_cfgs:
            paths.append(
                get_infer_output_path(
                    self.model_cfg,
                    dataset_cfg,
                    osp.join(self.work_dir, self.output_subdir),
                    file_extension,
                )
            )
        return paths


def parse_args():
    parser = argparse.ArgumentParser(description='VBench evaluation task')
    parser.add_argument('config', help='Config file path')
    return parser.parse_args()


if __name__ == '__main__':
    logger = AISLogger()
    args = parse_args()
    cfg = Config.fromfile(args.config)
    task_state_manager = TaskStateManager(
        tmp_path=osp.join(cfg['work_dir'], 'status_tmp'),
        task_name=task_abbr_from_cfg(cfg),
        is_debug=cfg['cli_args']['debug'],
    )
    manager_t = threading.Thread(target=task_state_manager.launch, args=())
    manager_t.start()
    task_state_manager.update_task_state({
        'status': 'start',
        'task_log_path': osp.join('logs/eval', f'{task_abbr_from_cfg(cfg)}.out'),
    })
    start_time = time.perf_counter()
    try:
        task = VBenchEvalTask(cfg)
        task.run(task_state_manager)
    except Exception as e:
        task_state_manager.update_task_state({'status': 'error'})
        raise e
    end_time = time.perf_counter()
    logger.info(f'VBench evaluation task time elapsed: {end_time - start_time:.2f}s')
    task_state_manager.update_task_state({'status': 'finish'})
    manager_t.join()
