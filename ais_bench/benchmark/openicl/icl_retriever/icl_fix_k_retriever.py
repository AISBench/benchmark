"""Random Retriever."""

from typing import List, Optional, Dict

from tqdm import trange

from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever
from ais_bench.benchmark.registry import ICL_RETRIEVERS
from ais_bench.benchmark.utils.logging.error_codes import ICLR_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchValueError


@ICL_RETRIEVERS.register_module()
class FixKRetriever(BaseRetriever):
    """Fix-K Retriever. Each in-context example of the test prompts is
    retrieved as the same K examples from the index set.

    Args:
        dataset (`BaseDataset`): Any BaseDataset instances.
            Attributes of ``reader``, ``train`` and ``test`` will be used.
        fix_id_list (List[int]): List of in-context example indices for every
            test prompts.
        ice_separator (`Optional[str]`): The separator between each in-context
            example template when origin `PromptTemplate` is provided. Defaults
            to '\n'.
        ice_eos_token (`Optional[str]`): The end of sentence token for
            in-context example template when origin `PromptTemplate` is
            provided. Defaults to '\n'.
        ice_num (`Optional[int]`): The number of in-context example template
            when origin `PromptTemplate` is provided. Defaults to 1.
    """

    def __init__(
        self,
        dataset,
        fix_id_list: List[int],
        ice_template: Optional[Dict] = None,
        prompt_template: Optional[Dict] = None,
        ice_separator: Optional[str] = "\n",
        ice_eos_token: Optional[str] = "\n",
        ice_num: Optional[int] = 1,
    ) -> None:
        super().__init__(
            dataset,
            ice_template,
            prompt_template,
            ice_separator,
            ice_eos_token,
            ice_num,
        )
        self.fix_id_list = fix_id_list
        self.logger.info(
            f"Fix-K Retriever initialized with {len(self.fix_id_list)} in-context example indices for each test example"
        )

    def retrieve(self):
        """Retrieve the in-context example index for each test example."""
        num_idx = len(self.index_ds)
        for idx in self.fix_id_list:
            if idx >= num_idx:
                raise AISBenchValueError(
                    ICLR_CODES.FIX_K_RETRIEVER_INDEX_OUT_OF_RANGE,
                    f"Fix-K retriever index {idx} is out of range of {num_idx}",
                )
        rtr_idx_list = []
        for _ in trange(len(self.test_ds), disable=not self.is_main_process):
            rtr_idx_list.append(self.fix_id_list)
        return rtr_idx_list
