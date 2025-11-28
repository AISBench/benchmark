import argparse
import os
import os.path as osp
import random
import threading
import sys
import time
from typing import Any

from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from ais_bench.benchmark.registry import (ICL_INFERENCERS, ICL_RETRIEVERS, TASKS)
from ais_bench.benchmark.tasks.base import BaseTask
from ais_bench.benchmark.utils.config import build_dataset_from_cfg
from ais_bench.benchmark.utils.core.abbr import task_abbr_from_cfg, model_abbr_from_cfg
from ais_bench.benchmark.tasks.base import TaskStateManager
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.core.types import check_type
from ais_bench.benchmark.utils.logging.error_codes import TINFER_CODES
from ais_bench.benchmark.utils.logging.exceptions import ParameterValueError
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_local_inferencer import BaseLocalInferencer



@TASKS.register_module()
class OpenICLInferTask(BaseTask):
    """OpenICL Inference Task.

    This task is used to run the inference process.
    """

    name_prefix = 'OpenICLInfer'
    log_subdir = 'logs/infer'
    output_subdir = 'predictions'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        run_cfg = self.model_cfg.get('run_cfg', {})
        self.num_gpus = run_cfg.get('num_gpus', 0)
        self.num_procs = run_cfg.get('num_procs', 1)
        self.nnodes = run_cfg.get('nnodes', 1)
        self.node_rank = run_cfg.get('node_rank', 0)
        self.master_addr = run_cfg.get('master_addr', "localhost")
        self.logger.debug(f"Local infer task config: {run_cfg}")

    def get_command(self, cfg_path, template):
        """Get the command template for the task.

        Args:
            cfg_path (str): The path to the config file of the task.
            template (str): The template which have '{task_cmd}' to format
                the command.
        """
        sys.path.append(os.getcwd())
        script_path = __file__
        backend_keys = ['VLLM', 'Lmdeploy']
        use_backend = any(
            key in str(self.model_cfg.get('type', ''))
            or key in str(self.model_cfg.get('llm', {}).get('type', ''))
            for key in backend_keys)
        if self.num_gpus > 1 and not use_backend and self.nnodes == 1:
            port = random.randint(12000, 32000)
            command = (f'torchrun --master_port={port} '
                       f'--nproc_per_node {self.num_procs} '
                       f'{script_path} {cfg_path}')
        elif self.nnodes > 1:
            port = 12345
            command = (f'torchrun --master_port={port} '
                       f'--nproc_per_node {self.num_procs} '
                       f'--nnodes {self.nnodes} '
                       f'--node_rank {self.node_rank} '
                       f'--master_addr {self.master_addr} '
                       f'{script_path} {cfg_path}')
        else:
            python = sys.executable
            command = f'{python} {script_path} {cfg_path}'

        return template.format(task_cmd=command)

    def run(self, task_state_manager):
        self.logger.info(f'Task {task_abbr_from_cfg(self.cfg)}')
        self.task_state_manager: TaskStateManager = task_state_manager

        self.max_out_len = self.model_cfg.get('max_out_len', None)
        self.batch_size = self.model_cfg.get('batch_size', None)
        self.min_out_len = self.model_cfg.get('min_out_len', None)

        num_return_sequences = getattr(self.model_cfg, 'generation_kwargs', {}).pop('num_return_sequences', 1)
        check_type(num_return_sequences, int)
        if num_return_sequences <= 0:
            raise ParameterValueError(
                TINFER_CODES.NUM_RETURN_SEQUENCES_NOT_POSITIVE,
                f"num_return sequences must be a positive integer, but got {num_return_sequences}",
            )
        if num_return_sequences > 1:
            self.logger.info(f'num_return_sequences is greater than 1, each data will be infer independently {num_return_sequences} times')

        self.infer_cfg = self.dataset_cfgs[0]['infer_cfg']
        self.sub_cfg = {
            'models': [self.model_cfg],
            'datasets': [self.dataset_cfgs],
        }
        self._inference()

    def build_inference(self):
        inferencer_cfg = self.infer_cfg['inferencer']
        inferencer_cfg['model_cfg'] = self.model_cfg
        self._set_default_value(inferencer_cfg, 'max_out_len',
                                self.max_out_len)
        self._set_default_value(inferencer_cfg, 'min_out_len',
                                self.min_out_len)
        self._set_default_value(inferencer_cfg, 'batch_size', self.batch_size)
        inferencer_cfg['max_seq_len'] = self.model_cfg.get('max_seq_len')
        self.logger.debug(f'Inferencer config: {inferencer_cfg}')
        self.inferencer: BaseLocalInferencer = ICL_INFERENCERS.build(inferencer_cfg)
        self.inferencer.set_task_state_manager(self.task_state_manager)

    def _inference(self):
        self.logger.info(
            f'Start inferencing {task_abbr_from_cfg(self.sub_cfg)}')

        retrievers = []
        for dataset_cfg in self.dataset_cfgs:
            infer_cfg = dataset_cfg["infer_cfg"]
            dataset = build_dataset_from_cfg(dataset_cfg)
            retriever_cfg = infer_cfg["retriever"].copy()
            retriever_cfg["dataset"] = dataset
            retriever_cfg["prompt_template"] = infer_cfg.get("prompt_template", None)
            retriever_cfg["ice_template"] = infer_cfg.get("ice_template", None)
            retriever = ICL_RETRIEVERS.build(retriever_cfg)
            retrievers.append(retriever)

        # set inferencer's default value according to model's config'
        self.task_state_manager.update_task_state({"status": "load model"})
        self.build_inference()

        out_dir = osp.join(self.work_dir, 'predictions', model_abbr_from_cfg(self.model_cfg))
        mkdir_or_exist(out_dir)
        self.logger.debug(f'Local infer task output directory: {out_dir}')

        self.inferencer.inference(retrievers,
                                 output_json_filepath=out_dir)

    def _set_default_value(self, cfg: ConfigDict, key: str, value: Any):
        if key not in cfg:
            cfg[key] = value


def parse_args():
    parser = argparse.ArgumentParser(description='Model Inferencer')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logger = AISLogger()
    args = parse_args()
    cfg = Config.fromfile(args.config)
    task_state_manager = TaskStateManager(
        tmp_path=os.path.join(cfg["work_dir"], "status_tmp"),
        task_name=task_abbr_from_cfg(cfg),
        is_debug=cfg["cli_args"]["debug"],
    )
    manager_t = threading.Thread(
        target=task_state_manager.launch,
        args=()
    )
    manager_t.start()
    task_state_manager.update_task_state(
        {
            "status": "start",
            "task_log_path": os.path.join("logs/infer/", f"{task_abbr_from_cfg(cfg)}.out"),
        }
    )
    start_time = time.perf_counter()
    try:
        inferencer: OpenICLInferTask = OpenICLInferTask(cfg)
        inferencer.run(task_state_manager)
    except Exception as e:
        task_state_manager.update_task_state({"status": "error"})
        raise e

    end_time = time.perf_counter()
    logger.info(f'Local infer task time elapsed: {end_time - start_time:.2f}s')
    task_state_manager.update_task_state({"status": "finish"})
    manager_t.join()
