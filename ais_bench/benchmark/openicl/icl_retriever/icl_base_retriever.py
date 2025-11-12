"""Basic Retriever."""

from abc import abstractmethod
from typing import Dict, List, Optional

from mmengine.dist import is_main_process

from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import ICLR_CODES
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchValueError,
    AISBenchImplementationError,
)


class BaseRetriever:
    """Base class for In-context Learning Example Retriever, without any
    retrieval method implemented.

    Args:
        dataset_cfg (`Config`): Dataset config.
        ice_separator (`Optional[str]`): The separator between each in-context
            example template when origin `PromptTemplate` is provided. Defaults
            to '\n'.
        ice_template (`Optional[Dict]`): The template for
            in-context example. Defaults to None.
        prompt_template (`Optional[Dict]`): The template for
            prompt. Defaults to None.
        ice_eos_token (`Optional[str]`): The end of sentence token for
            in-context example template when origin `PromptTemplate` is
            provided. Defaults to '\n'.
        ice_num (`Optional[int]`): The number of in-context example template
            when origin `PromptTemplate` is provided. Defaults to 1.
    """

    index_ds = None
    test_ds = None

    def __init__(
        self,
        dataset,
        ice_template: Optional[Dict] = None,
        prompt_template: Optional[Dict] = None,
        ice_separator: Optional[str] = "\n",
        ice_eos_token: Optional[str] = "\n",
        ice_num: Optional[int] = 1,
    ) -> None:
        self.logger = AISLogger()
        self.dataset = dataset
        self.ice_template = (
            None if ice_template is None else ICL_PROMPT_TEMPLATES.build(ice_template)
        )
        self.logger.debug(f"Ice template: {self.ice_template}")
        self.prompt_template = (
            None
            if prompt_template is None
            else ICL_PROMPT_TEMPLATES.build(prompt_template)
        )
        self.logger.debug(f"Prompt template: {self.prompt_template}")
        self.ice_separator = ice_separator
        self.ice_eos_token = ice_eos_token
        self.ice_num = ice_num
        self.is_main_process = is_main_process()
        self.dataset_reader = self.dataset.reader
        self.index_ds = self.dataset.train
        self.test_ds = self.dataset.test

    @abstractmethod
    def retrieve(self) -> List[List[int]]:
        """Retrieve the in-context example index for each test example."""
        raise AISBenchImplementationError(
            ICLR_CODES.UNKNOWN_ERROR,
            f"{self.__class__.__name__} hasn't been implemented yet",
        )

    def get_gold_ans(self):
        if self.dataset_reader.output_column:
            return self.dataset_reader.dataset["test"][
                self.dataset_reader.output_column
            ]
        else:
            return None

    def get_labels(self) -> List[str]:
        """Get the labels of the dataset, especially useful for ppl inferencer.
        If `ice_template` is provided, the labels will be the keys of the
        template. If `prompt_template` is provided, the labels will be the keys
        of the template. If neither of them is provided, the labels will be the
        unique values of the output column.

        Args:
            ice_template (`Optional[PromptTemplate]`): The template for
                in-context example. Defaults to None.
            prompt_template (`Optional[PromptTemplate]`): The template for
                prompt. Defaults to None.
        """
        if self.prompt_template is not None and isinstance(
            self.prompt_template.template, Dict
        ):
            labels = list(self.prompt_template.template.keys())
        elif (
            self.ice_template is not None
            and self.ice_template.ice_token is not None
            and isinstance(self.ice_template.template, Dict)
        ):
            labels = list(self.ice_template.template.keys())
        else:
            labels = list(set(self.test_ds[self.dataset_reader.output_column]))
        return labels

    def generate_ice(self, idx_list: List[int]) -> str:
        """Generate the in-context example for one test example. If
        `ice_template` is an instance of `PromptTemplate`, the `ice_separator`
        and `ice_eos_token` will be set as empty.

        Args:
            idx_list (`List[int]`): The index of in-context examples for the
                test example.
        """
        if self.ice_template is None and len(idx_list) > 0:
            raise AISBenchValueError(
                ICLR_CODES.TEMPLATE_ICE_TOKEN_NOT_IN_TEMPLATE,
                f"You have not specified ice_template while retrieving examples \
                                 from train set! Please either specify ice_template or use `ZeroRetriever`.",
            )

        if self.ice_template is not None and self.ice_template.prompt_type == "meta":
            ice_separator, ice_eos_token = "", ""
        else:
            ice_separator = self.ice_separator
            ice_eos_token = self.ice_eos_token

        generated_ice_list = []
        for idx in idx_list:
            generated_ice_list.append(
                self.ice_template.generate_ice_item(
                    self.index_ds[idx],
                    self.index_ds[idx][self.dataset_reader.output_column],
                )
            )
        if len(generated_ice_list) > 0 and isinstance(
            generated_ice_list[0], PromptList
        ):
            generated_ice = []
            for ice in generated_ice_list:
                generated_ice += ice + ice_separator
            generated_ice.append(ice_eos_token)
        else:
            generated_ice = ice_separator.join(generated_ice_list) + ice_eos_token
        return generated_ice

    def generate_label_prompt(
        self,
        idx: int,
        ice: str,
        label,
        remain_sep: Optional[bool] = False,
    ) -> str:
        """Generate the prompt for one test example in perpelxity evaluation
        with `prompt_template`. If `prompt_template` is not provided, the
        `ice_template` will be used to generate the prompt.

        Args:
            idx (`int`): The index of the test example.
            ice (`str`): The in-context example for the test example.
            label (`str`): The label of the test example.
            remain_sep (`Optional[bool]`): Whether to remain the sep token.
                Defaults to False.
        """
        if self.prompt_template is not None and self.ice_template is not None:
            if self.prompt_template.ice_token is not None:
                return self.prompt_template.generate_label_prompt_item(
                    self.test_ds[idx], ice, label, remain_sep
                )
            else:
                raise AISBenchImplementationError(
                    ICLR_CODES.IMPLEMENTATION_ERROR_ICE_TOKEN_NOT_PROVIDED,
                    f"ice_token of prompt_template is not provided",
                )
        elif self.ice_template is not None and self.prompt_template is None:
            if self.ice_template.ice_token is not None:
                return self.ice_template.generate_label_prompt_item(
                    self.test_ds[idx], ice, label, remain_sep
                )
            else:
                raise AISBenchImplementationError(
                    ICLR_CODES.IMPLEMENTATION_ERROR_ICE_TOKEN_NOT_PROVIDED,
                    f"ice_token of ice_template is not provided",
                )
        elif self.ice_template is None and self.prompt_template is not None:
            return self.prompt_template.generate_label_prompt_item(
                self.test_ds[idx], ice, label, remain_sep
            )
        else:
            raise AISBenchImplementationError(
                ICLR_CODES.IMPLEMENTATION_ERROR_PROMPT_TEMPLATE_NOT_PROVIDED,
                f"Leaving prompt as empty is not supported",
            )

    def generate_prompt_for_generate_task(
        self,
        idx,
        ice,
        gen_field_replace_token="",
    ):
        """Generate the prompt for one test example in generative evaluation
        with `prompt_template`. If `prompt_template` is not provided, the
        `ice_template` will be used to generate the prompt. The token
        represented by `gen_field_replace_token` will not be replaced by the
        generated text, or it will leaks the answer.

        Args:
            idx (`int`): The index of the test example.
            ice (`str`): The in-context example for the test example.
            gen_field_replace_token (`str`): The token of the answer in the
                prompt. Defaults to ''.
        """
        if self.prompt_template is not None and self.ice_template is not None:
            if self.prompt_template.ice_token is not None:
                return self.prompt_template.generate_item(
                    self.test_ds[idx],
                    output_field=self.dataset_reader.output_column,
                    output_field_replace_token=gen_field_replace_token,
                    ice_field_replace_token=ice,
                )
            else:
                raise AISBenchImplementationError(
                    ICLR_CODES.IMPLEMENTATION_ERROR_ICE_TOKEN_NOT_PROVIDED,
                    f"ice_token of prompt_template is not provided",
                )

        elif self.ice_template is not None and self.prompt_template is None:
            if self.ice_template.ice_token is not None:
                return self.ice_template.generate_item(
                    self.test_ds[idx],
                    output_field=self.dataset_reader.output_column,
                    output_field_replace_token=gen_field_replace_token,
                    ice_field_replace_token=ice,
                )
            else:
                raise AISBenchImplementationError(
                    ICLR_CODES.IMPLEMENTATION_ERROR_ICE_TOKEN_NOT_PROVIDED,
                    f"ice_token of ice_template is not provided",
                )
        elif self.ice_template is None and self.prompt_template is not None:
            return self.prompt_template.generate_item(
                self.test_ds[idx],
                output_field=self.dataset_reader.output_column,
                output_field_replace_token=gen_field_replace_token,
                ice_field_replace_token=ice,
            )
        else:
            raise AISBenchImplementationError(
                ICLR_CODES.IMPLEMENTATION_ERROR_PROMPT_TEMPLATE_NOT_PROVIDED,
                f"Leaving prompt as empty is not supported",
            )
