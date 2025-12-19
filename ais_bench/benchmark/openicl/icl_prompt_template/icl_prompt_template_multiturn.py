"""Multiturn Dialogue Prompt Template."""

from typing import Dict, Hashable, Optional

from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES
from ais_bench.benchmark.utils.prompt import PromptList, get_round_index
from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_base import (
    BasePromptTemplate,
    PromptType,
)


@ICL_PROMPT_TEMPLATES.register_module()
class MultiTurnPromptTemplate(BasePromptTemplate):
    """Multi-turn dialogue Prompt Template Class This class represents a
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
        template = self._encode_template(self.template, ice=False)
        dialog_templates = PromptList()
        left_idx, right_idx = get_round_index(template)
        left_template, right_template = PromptList(template[:left_idx]), PromptList(template[right_idx:])
        template = PromptList(template[left_idx:right_idx])
        for question, answer in zip(entry["question"], entry["answer"]):
            cur_entry = {"question": question, "answer": answer}
            dialog_templates += template.format(**cur_entry)

        return left_template + dialog_templates + right_template

    def __repr__(self):
        return (
            f"MultiTurnPromptTemplate({{\n\ttemplate: {self.template},\n\t"
            f"ice_token: {self.ice_token}\n}})"
        )
