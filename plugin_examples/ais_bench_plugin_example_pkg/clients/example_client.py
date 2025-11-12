from abc import ABC

import uuid
from ais_bench.benchmark.clients.base_client import BaseClient
from ais_bench.benchmark.utils import MiddleData
from ais_bench.benchmark.registry import CLIENTS


@CLIENTS.register_module()
class ExampleClient(BaseClient, ABC):
    def construct_request_body(
        self,
        inputs: dict,
        parameters: dict = None,
    ) -> dict:
        data = dict(
            messages = inputs,
            stream = False,
        )
        data = data | parameters
        return data

    def update_middle_data(self, res: dict, inputs: MiddleData):
        try:
            generated_text = res['choices'][0]['message'].get('content', '')
            reasoning_content = res['choices'][0]['message'].get('reasoning_content', '')
        except Exception as e:
            raise RuntimeError(f"Process response failed and the reason is {e}")
        if generated_text:
            inputs.output = generated_text
            inputs.num_generated_chars = len(generated_text)
        if reasoning_content:
            inputs.output_reasoning = reasoning_content
            inputs.num_generated_chars += len(reasoning_content)
