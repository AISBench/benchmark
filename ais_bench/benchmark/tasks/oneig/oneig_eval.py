"""
OneIG 评测任务模块

参照 VBenchEvalTask 模式，继承 BaseTask 并覆写 run()。
所有5个子任务统一走自定义评测路径：
- Alignment/Text：evaluator 内部调用 Judge 模型（LLM-as-Judge）
- Reasoning/Style/Diversity：evaluator 内部调用 ML 模型

Judge 推理逻辑已内聚到 OneIGAlignmentEvaluator 和 OneIGTextEvaluator 中，
Task 层只需构建 test_set 和 evaluator，调用 evaluator.score() 即可。
"""

import os
import os.path as osp
import sys
import threading
import time

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from ais_bench.benchmark.registry import ICL_EVALUATORS, TASKS
from ais_bench.benchmark.tasks.base import BaseTask, TaskStateManager
from ais_bench.benchmark.utils.core.abbr import (
    dataset_abbr_from_cfg,
    get_infer_output_path,
    task_abbr_from_cfg,
)
from ais_bench.benchmark.utils.config import build_dataset_from_cfg
from ais_bench.benchmark.utils.logging import AISLogger

@TASKS.register_module()
class OneIGEvalTask(BaseTask):
    """OneIG 评测任务，参照 VBenchEvalTask 模式覆写 run()。

    遍历 dataset_cfgs，对每个子任务：
    1. 构建 test_set（OneIGDataset.load()）
    2. 构建 evaluator（ICL_EVALUATORS.build()）
    3. 调用 evaluator.score(predictions=[], references=[], test_set=test_set)
    4. 保存结果到 work_dir/results/

    Judge 推理（Alignment/Text）由 evaluator 内部完成，
    无需 Task 层处理 predictions 文件或 JDG Dataset。
    """

    name_prefix = 'OneIGEval'
    log_subdir = 'logs/eval'
    output_subdir = 'results'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        self.num_gpus = max(
            c.get('eval_cfg', {}).get('num_gpus', 0)
            for c in sum([self.dataset_cfgs], []))

    def get_command(self, cfg_path, template):
        sys.path.insert(0, os.getcwd())
        script_path = __file__
        python = sys.executable
        command = f'{python} {script_path} {cfg_path}'
        return template.format(task_cmd=command)

    def run(self, task_state_manager=None):
        self.task_state_manager = task_state_manager

        for dataset_cfg in self.dataset_cfgs:
            task_type = dataset_cfg.get('task_type', '')
            eval_cfg = dataset_cfg.get('eval_cfg', {})

            # 构建 test_set
            ds = build_dataset_from_cfg(
                dataset_cfg, task_state_manager=task_state_manager)
            test_set = ds.test

            # 构建 evaluator
            icl_evaluator = ICL_EVALUATORS.build(eval_cfg['evaluator'])

            # 设置输出目录
            out_path = get_infer_output_path(
                self.model_cfg, dataset_cfg,
                osp.join(self.work_dir, self.output_subdir))
            results_dir = osp.dirname(out_path)
            icl_evaluator._out_dir = results_dir
            mkdir_or_exist(results_dir)

            # 统一调用 evaluator.score()
            # Judge 推理（Alignment/Text）由 evaluator 内部完成
            self.logger.info(
                f"[OneIG] Evaluating {dataset_abbr_from_cfg(dataset_cfg)} "
                f"(task_type={task_type})")

            result = icl_evaluator.score(
                predictions=[],
                references=[],
                test_set=test_set
            )

            # 保存结果
            result_wo_details = {k: v for k, v in result.items() if k != 'details'}
            self.logger.info(
                f'Task {task_abbr_from_cfg(self.cfg)}: {result_wo_details}')

            mkdir_or_exist(osp.dirname(out_path))
            mmengine.dump(result, out_path, ensure_ascii=False, indent=4)
            self.logger.info(f"[OneIG] Results saved to {out_path}")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='OneIG Evaluation Task')
    parser.add_argument('config', help='Config file path')
    return parser.parse_args()


if __name__ == '__main__':
    logger = AISLogger()
    args = parse_args()
    cfg = Config.fromfile(args.config)
    task_state_manager = TaskStateManager(
        tmp_path=os.path.join(cfg["work_dir"], "status_tmp"),
        task_name=task_abbr_from_cfg(cfg),
        is_debug=cfg.get("cli_args", {}).get("debug", False),
    )
    manager_t = threading.Thread(target=task_state_manager.launch, args=())
    manager_t.start()
    task_state_manager.update_task_state({
        "status": "start",
        "task_log_path": os.path.join("logs/eval/", f"{task_abbr_from_cfg(cfg)}.out"),
    })
    start_time = time.perf_counter()
    try:
        task = OneIGEvalTask(cfg)
        task.run(task_state_manager)
    except Exception as e:
        task_state_manager.update_task_state({"status": "error"})
        raise e
    end_time = time.perf_counter()
    logger.info(f'OneIG evaluation task time elapsed: {end_time - start_time:.2f}s')
    task_state_manager.update_task_state({"status": "finish"})
    manager_t.join()
