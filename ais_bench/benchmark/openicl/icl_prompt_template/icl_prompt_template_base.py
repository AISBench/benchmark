"""Prompt Template."""

from typing import Dict, Hashable, List, Optional, Union

from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.utils.core.types import check_type_list
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import ICLR_CODES
from ais_bench.benchmark.utils.logging.exceptions import (
    AISBenchValueError,
    AISBenchImplementationError,
)

PromptType = Union[PromptList, str, dict]


@ICL_PROMPT_TEMPLATES.register_module()
class BasePromptTemplate:
    """In-context Learning Prompt Template Class This class represents a
    template that guides the generation of prompts in the retrieval or
    inference process.

    Attributes:
        template (:obj:`Dict` or :obj:`str`): A custom template dictionary or
            string. If a dictionary, the keys of the dictionary represent the
            values of the output_column, and the values represent the
            corresponding generated statement. If a string, it represents a
            string template.
        ice_token(:obj:`str`, optional): A string that represents the specific
            token mapping from in-context examples. None if you want to use
            this template only to generate in-context examples, otherwise it
            can be used to generate the final prompt that is fed into the PLM.
            The ice_token will be invisible when generating in-context
            examples.
    """

    def __init__(
        self,
        template: Union[Dict, str],
        ice_token: Optional[str] = None,
        sep_token: Optional[str] = None,
    ) -> None:
        self.logger = AISLogger()
        self.template = template
        if not isinstance(self.template, (str, Dict)):
            raise AISBenchValueError(
                ICLR_CODES.TEMPLATE_TYPE_ERROR,
                f"Prompt template must be a str or a dict, but got {type(self.template)}",
            )
        self.ice_token = check_type_list(ice_token, [None, str])
        self.sep_token = check_type_list(sep_token, [None, str])
        # A sign used to distinguish the prompt type
        self.prompt_type = "origin"
        self._check_template_legacy()

    def _check_template_legacy(self):
        if isinstance(self.template, Dict):
            # Check if it's the label-prompt type or just a meta prompt type
            ctr = sum(key in self.template for key in ("begin", "round", "end"))
            self.prompt_type = "meta" if ctr == len(self.template.keys()) else "origin"
            self.logger.debug(
                f"Prompt template type: {self.prompt_type} with keys: {self.template.keys()}"
            )

            # Check if token exists in values of tp_dict
            for tp_dict_val in self.template.values():
                if not isinstance(tp_dict_val, (str, list, dict)):
                    raise AISBenchValueError(
                        ICLR_CODES.TEMPLATE_VALUE_TYPE_ERROR,
                        f"dictionary of template expects a str, list or a dict, but got {type(tp_dict_val)}, value: {tp_dict_val}",
                    )
                if (
                    isinstance(tp_dict_val, str)
                    and self.ice_token
                    and self.ice_token not in tp_dict_val
                ):
                    raise AISBenchValueError(
                        ICLR_CODES.TEMPLATE_ICE_TOKEN_NOT_IN_VALUE,
                        f"'{self.ice_token}' not in '{tp_dict_val}'",
                    )

        if isinstance(self.template, str):
            if self.ice_token and self.ice_token not in self.template:
                raise AISBenchValueError(
                    ICLR_CODES.TEMPLATE_ICE_TOKEN_NOT_IN_VALUE,
                    f"'{self.ice_token}' not in '{self.template}'",
                )

    def generate_ice_item(self, entry: Dict, label: Hashable) -> PromptType:
        """Generate in-context example based on the provided :obj:`entry` data.

        Args:
            entry (:obj:`Dict`): A piece of data to be used for generating the
                in-context example.
            label (:obj:`Hashable`): The value of the output field.

        Returns:
            PromptType: The generated in-context example.
        """
        raise AISBenchImplementationError(
            ICLR_CODES.UNKNOWN_ERROR,
            f"{self.__class__.__name__} does not supported to be called in base classes",
        )

    def generate_label_prompt_item(
        self,
        entry: Dict,
        ice: PromptType,
        label: Hashable,
        remain_sep: Optional[bool] = False,
    ) -> str:
        raise AISBenchImplementationError(
            ICLR_CODES.UNKNOWN_ERROR,
            f"{self.__class__.__name__} does not supported to be called in base classes",
        )

    def generate_item(
        self,
        entry: Dict,
        output_field: Optional[Hashable] = None,
        output_field_replace_token: Optional[str] = "",
        ice_field_replace_token: Optional[str] = "",
    ) -> PromptType:
        raise AISBenchImplementationError(
            ICLR_CODES.UNKNOWN_ERROR,
            f"{self.__class__.__name__} does not supported to be called in base classes",
        )

    def _check_prompt_template(obj) -> "BasePromptTemplate":
        if isinstance(obj, BasePromptTemplate):
            return obj
        else:
            raise AISBenchValueError(
                ICLR_CODES.TEMPLATE_TYPE_ERROR,
                f"Expect a BasePromptTemplate object, but got {type(obj)}",
            )

    def __repr__(self):
        return (
            f"BasePromptTemplate({{\n\ttemplate: {self.template},\n\t"
            f"ice_token: {self.ice_token}\n}})"
        )

    def _encode_template(
        self, prompt_template: Union[List[Union[str, Dict]], str], ice: bool
    ) -> PromptType:
        """Encode the raw template given in the config into a str or a
        PromptList.

        Args:
            prompt_template (List[Dict]] or str): The raw template given in the
                config, used for generating the prompt. If it's a string, the
                result will be directly returned.
            ice (bool): If the template is used for generating in-context
                examples.

        Returns:
            PromptType: The encoded template.
        """
        if isinstance(prompt_template, str):
            return prompt_template

        prompt = PromptList()

        # TODO: Why can't we generate begin & end for ice template?
        # To fix this, first we need to allow specifying prompt_template
        # only
        if "begin" in prompt_template and not ice:
            prompt.append(dict(section="begin", pos="begin"))
            if isinstance(prompt_template["begin"], list):
                prompt += prompt_template["begin"]
            else:
                prompt.append(prompt_template["begin"])
            prompt.append(dict(section="begin", pos="end"))

        if ice:
            prompt.append(dict(section="ice", pos="begin"))
        else:
            prompt.append(dict(section="round", pos="begin"))
        prompt += prompt_template["round"]
        if ice:
            prompt.append(dict(section="ice", pos="end"))
        else:
            prompt.append(dict(section="round", pos="end"))

        if "end" in prompt_template and not ice:
            prompt.append(dict(section="end", pos="end"))
            if isinstance(prompt_template["end"], list):
                prompt += prompt_template["end"]
            else:
                prompt.append(prompt_template["end"])
            prompt.append(dict(section="end", pos="end"))

        return prompt
