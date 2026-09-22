from __future__ import annotations

from typing import Any, Hashable

from ais_bench.benchmark.openicl.icl_prompt_template import MMPromptTemplate
from ais_bench.benchmark.registry import ICL_PROMPT_TEMPLATES

from ..multimodal import expand_prompt_image_refs


@ICL_PROMPT_TEMPLATES.register_module()
class Base64RefMMPromptTemplate(MMPromptTemplate):
    """Expand compact image references to Base64 data URLs at prompt generation time."""

    def __init__(self, media: dict[str, str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.media = dict(media)

    def generate_item(
        self,
        entry: dict,
        output_field: Hashable | None = None,
        output_field_replace_token: str | None = "",
        ice_field_replace_token: str | None = "",
    ):
        prompt = super().generate_item(
            entry,
            output_field=output_field,
            output_field_replace_token=output_field_replace_token,
            ice_field_replace_token=ice_field_replace_token,
        )
        return expand_prompt_image_refs(prompt, self.media)
