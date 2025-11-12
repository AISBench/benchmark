import os
import os.path as osp
from typing import Dict, List, Optional

from mmengine.config import Config, ConfigDict

from ais_bench.benchmark.registry import PARTITIONERS
from ais_bench.benchmark.utils.core.abbr import get_infer_output_path, model_abbr_from_cfg, dataset_abbr_from_cfg
from ais_bench.benchmark.utils.logging.error_codes import PARTI_CODES
from .base import BasePartitioner


@PARTITIONERS.register_module()
class NaivePartitioner(BasePartitioner):
    """Naive task partitioner. This partitioner will generate a task for each n
    model-dataset pairs.

    Args:
        out_dir (str): The output directory of tasks.
        n (int): The number of model-dataset pairs in each task.
        keep_keys (List[str]): The keys to be kept from the experiment config
            to the task config.
    """

    def __init__(self,
                 out_dir: str,
                 n: int = 1,
                 keep_keys: Optional[List[str]] = None):
        super().__init__(out_dir=out_dir, keep_keys=keep_keys)
        self.n = n

    def partition(self,
                  model_dataset_combinations: List[Dict[str,
                                                        List[ConfigDict]]],
                  work_dir: str,
                  out_dir: str,
                  add_cfg: Dict = {}) -> List[Dict]:
        """Partition model-dataset pairs into tasks. Each task is defined as a
        dict and will run independently as a unit. Its structure is as
        follows:

        .. code-block:: python

            {
                'models': [],  # a list of model configs
                'datasets': [[]],  # a nested list of dataset configs, each
                                    list corresponds to a model
                'work_dir': '',  # the work dir
            }

        Args:
            model_dataset_combinations (List[Dict]): List of
                `{models: [...], datasets: [...]}` dicts. Each dict contains
                a list of model configs and a list of dataset configs.
            work_dir (str): The work dir for the task.
            out_dir (str): The full output path for the task, intended for
                Partitioners to check whether the task is finished via the
                existency of result file in this directory.

        Returns:
            List[Dict]: A list of tasks.
        """

        tasks = []
        is_predictions = osp.basename(osp.normpath(out_dir)) == 'predictions'
        for comb in model_dataset_combinations:
            for model in comb["models"]:
                chunks = []
                model_abbr = model_abbr_from_cfg(model)
                for dataset in comb["datasets"]:
                    filename = get_infer_output_path(model, dataset, out_dir)
                    dataset_abbr = dataset_abbr_from_cfg(dataset)
                    tmp_data = osp.join(osp.dirname(filename), "tmp_" + dataset_abbr)
                    task_name = "[" + model_abbr + "/" + dataset_abbr + "]"
                    if osp.exists(filename) and is_predictions: # skip if not in predictions mode
                        stat_info = os.stat(filename)
                        if stat_info.st_uid != os.getuid():
                            self.logger.error(
                                PARTI_CODES.OUT_DIR_PERMISSION_DENIED,
                                f"Current user can't modify {filename}, reuse will not enable."
                            )
                            continue
                        if osp.exists(tmp_data):
                            self.logger.warning(
                                f"Partial results found of {task_name}, {filename} will be overwritten."
                            )
                        else:
                            self.logger.info(f"{task_name} has been finished, skip.")
                            continue
                    chunks.append(dataset)

                for i in range(0, len(chunks), self.n):
                    task = Config({
                        'models': [model],
                        'datasets': [chunks[i:i + self.n]],
                        'work_dir': work_dir,
                        **add_cfg
                    })
                    tasks.append(task)
        return tasks
