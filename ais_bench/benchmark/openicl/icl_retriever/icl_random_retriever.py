"""Random Retriever."""

from typing import Optional, Dict

import numpy as np
from tqdm import trange

from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever


class RandomRetriever(BaseRetriever):
    """Random Retriever. Each in-context example of the test prompts is
    retrieved in a random way.

    **WARNING**: This class has not been tested thoroughly. Please use it with
    caution.
    """

    def __init__(self,
                 dataset,
                 ice_template: Optional[Dict] = None,
                 prompt_template: Optional[Dict] = None,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 ice_num: Optional[int] = 1,
                 seed: Optional[int] = 43) -> None:
        super().__init__(dataset, ice_template, prompt_template, ice_separator, ice_eos_token, ice_num)
        self.seed = seed
        self.logger.info(f"Random Retriever initialized with seed {self.seed}")

    def retrieve(self):
        np.random.seed(self.seed)
        num_idx = len(self.index_ds)
        rtr_idx_list = []
        self.logger.info('Random retrieving data for test set...')
        for _ in trange(len(self.test_ds), disable=not self.is_main_process):
            idx_list = np.random.choice(num_idx, self.ice_num,
                                        replace=False).tolist()
            rtr_idx_list.append(idx_list)
        np.random.seed(None) # reset seed
        return rtr_idx_list
