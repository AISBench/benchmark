# flake8: noqa
# yapf: disable
import time
from typing import Dict, List, Optional, Union


from ais_bench.benchmark.models.local_models.base import BaseModel
from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES
from ais_bench.benchmark.models.local_models.huggingface_above_v4_33 import (_convert_chat_messages,
                                                                            _get_meta_template,
                                                                            )

PromptType = Union[PromptList, str]
VLLM_MAX_IMAGE_INPUT_NUM = 24


@MODELS.register_module()
class VLLMOfflineVLModel(BaseModel):
    """Model wrapper for Qwen2.5-VL VLLM Offline models.

    Args:
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
    """

    def __init__(self,
                 path: str,
                 model_kwargs: dict = dict(),
                 sample_kwargs: dict = dict(),
                 vision_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,

                 **other_kwargs):
        self.logger = AISLogger()
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            self.logger.error(UTILS_CODES.DEPENDENCY_MODULE_IMPORT_ERROR, "pip install vllm")
        self.path = path
        self.max_out_len = other_kwargs.get('max_out_len', None)
        self.template_parser = _get_meta_template(meta_template)
        self.llm = LLM(model=self.path, **model_kwargs)

        if any(item in self.path.lower() for item in ['omni']):
            from transformers import Qwen2_5OmniProcessor
            self.processor = Qwen2_5OmniProcessor.from_pretrained(self.path)
        elif any(item in self.path.lower() for item in ['2.5', '2_5', 'qwen25', 'mimo']):
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.path)
        else:
            from transformers import Qwen2VLProcessor
            self.processor = Qwen2VLProcessor.from_pretrained(self.path)

        sample_kwargs.update({"max_tokens": self.max_out_len})
        self.sampling_params = SamplingParams(**sample_kwargs)
        self.limit_mm_per_prompt = VLLM_MAX_IMAGE_INPUT_NUM
        self.min_pixels = vision_kwargs.pop('min_pixels', None)
        self.max_pixels = vision_kwargs.pop('max_pixels', None)
        self.total_pixels = vision_kwargs.pop('total_pixels', None)
        self.fps = vision_kwargs.pop('fps', 2)
        self.nframe = vision_kwargs.pop('nframe', 128)
        self.FRAME_FACTOR = 2
        self.post_process = False


    def format_image_input(self, inputs):
        for i in range(len(inputs)):
            if not isinstance(inputs[i], list) or len(inputs[i]) != 1 or not isinstance(inputs[i][0], dict):
                self.logger.warning("Invalid input format, please check it!")
            prompt = []
            for item in inputs[i][0]['prompt']:
                if item['type']=='image_url':
                    image_url = {'type': 'image', 'image': item['image_url']}
                    if self.min_pixels is not None:
                        image_url['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        image_url['max_pixels'] = self.max_pixels
                    if self.total_pixels is not None:
                        image_url['total_pixels'] = self.total_pixels
                    prompt.append(image_url)
                else:
                    prompt.append(item)
            inputs[i][0]['prompt'] = prompt

    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                 **kwargs) -> List[str]:

        self.format_image_input(inputs)
        messages = _convert_chat_messages(inputs)
        batch_size = len(messages)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # process vision
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        # step-2: conduct model forward to generate output
        start_time = time.perf_counter()
        outputs = self.llm.generate(
            {
                "prompt": text[0],
                "multi_modal_data": {"image": image_inputs},
            },
            sampling_params=self.sampling_params,
        )
        end_time = time.perf_counter()

        # step-3: decode the output
        for o in outputs:
            generated_text = o.outputs[0].text
        if self.post_process:
            resp = generated_text.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                generated_text = resp[:end]
        return generated_text