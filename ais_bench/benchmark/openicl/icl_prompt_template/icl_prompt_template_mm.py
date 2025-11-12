"""Multimodal Prompt Template."""

from typing import Dict, Hashable, Optional, Union

from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES
from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_base import (
    BasePromptTemplate,
    PromptType,
)
from ais_bench.benchmark.utils.logging.error_codes import ICLR_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchValueError


@ICL_PROMPT_TEMPLATES.register_module()
class MMPromptTemplate(BasePromptTemplate):
    """Image-Text Multi-Modal Prompt Template Class This class represents a
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

    def check_mm_template(self):
        if not isinstance(self.template, dict) or "round" not in self.template.keys():
            return False
        for data in self.template["round"]:
            if "prompt_mm" not in data.keys():
                return False
        return True

    def get_mm_template(self, item):
        """
        change format
        """
        item = item["prompt_mm"]
        res = []
        for key in item.keys():
            if key not in ["text", "image_url", "video_url", "audio_url"]:
                raise AISBenchValueError(
                    ICLR_CODES.MULTIMODAL_TEMPLATE_TYPE_ERROR,
                    f"The keys in prompt_mm must be one of: text, image_url, video_url or audio_url, but got {key}"
                )
            res.append({"type": key, key: item[key]})
        return res

    def generate_item(
        self,
        entry: Dict,
        output_field: Optional[Hashable] = None,
        output_field_replace_token: Optional[str] = "",
        ice_field_replace_token: Optional[str] = "",
    ) -> PromptType:
        """Generate an item based on the provided :obj:`entry` data, as well as
        optional output field and ice field tokens.

        Warning:
            This method is only used in generation task, i.e. GenInferencer.

        Args:
            entry (:obj:`Dict`): A piece of data.
            output_field (:obj:`Hashable`, optional): Column name of output
                field. Defaults to :obj:`None`.
            output_field_replace_token (:obj:`str`, optional): Tokens used to
                replace output field. Defaults to ``''``.
            ice_field_replace_token (str, optional): Tokens used to replace
                the :obj:`ice_token`. Defaults to ``''``.

        Returns:
            PromptType: The generated item.
        """
        if not self.check_mm_template():
            self.logger.warning(f'Expected to get template with round and prompt_mm, but got {self.template}')
        template = self._encode_template(self.template, ice=False)
        template = template.format(**entry)
        for i, item in enumerate(template):
            if "prompt_mm" in item:
                template[i]["prompt_mm"] = self.get_mm_template(item)
        return template

    def __repr__(self):
        return (
            f"MMPromptTemplate({{\n\ttemplate: {self.template},\n\t"
            f"ice_token: {self.ice_token}\n}})"
        )
