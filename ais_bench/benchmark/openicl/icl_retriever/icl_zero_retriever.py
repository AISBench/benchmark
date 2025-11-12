"""Zeroshot Retriever."""

from typing import List, Optional
from typing import Dict

from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever
from ais_bench.benchmark.registry import ICL_RETRIEVERS


@ICL_RETRIEVERS.register_module()
class ZeroRetriever(BaseRetriever):
    """Zeroshot Retriever. The retriever returns empty list for all queries.

    Args:
        dataset_cfg (`Config`): Dataset config.
        ice_template (`Optional[Dict]`): The template for
            in-context example. Defaults to None.
        prompt_template (`Optional[Dict]`): The template for
            prompt. Defaults to None.
        ice_eos_token (`Optional[str]`): The end of sentence token for
            in-context example template when origin `PromptTemplate` is
            provided. Defaults to ''.
    """

    def __init__(
        self,
        dataset,
        ice_template: Optional[Dict] = None,
        prompt_template: Optional[Dict] = None,
        ice_separator: Optional[str] = "",
        ice_eos_token: Optional[str] = "",
    ) -> None:
        super().__init__(dataset, ice_template, prompt_template, ice_separator, ice_eos_token, 0)
        self.logger.info("Zero Retriever initialized, returning empty shot case for all queries")

    def retrieve(self) -> List[List]:
        rtr_idx_list = [[] for _ in range(len(self.test_ds))]
        return rtr_idx_list
