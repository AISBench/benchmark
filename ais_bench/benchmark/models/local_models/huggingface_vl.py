# flake8: noqa
# yapf: disable
import time
from typing import Dict, List, Optional, Union

import torch
from mmengine.device import is_npu_available


from ais_bench.benchmark.models.local_models.base import BaseModel
from ais_bench.benchmark.models import LMTemplateParser
from ais_bench.benchmark.models import APITemplateParser
from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.logging import get_logger
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.utils.logging import AISLogger
from ais_bench.benchmark.models.local_models.huggingface_above_v4_33 import (_get_possible_max_seq_len,
                                                                            _convert_chat_messages,
                                                                            _get_meta_template,
                                                                            _set_model_kwargs_torch_dtype)

PromptType = Union[PromptList, str, dict]


@MODELS.register_module()
class HuggingFaceQwen2VLwithChatTemplate(BaseModel):
    """Model wrapper for Qwen2.5-VL HuggingFace models designed for chat.

    Args:
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
    """

    def __init__(self,
                 path: str,
                 model_kwargs: dict = dict(),
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 peft_path: Optional[str] = None,
                 peft_kwargs: dict = dict(),
                 tokenizer_only: bool = False,
                 generation_kwargs: dict = dict(),
                 max_seq_len: Optional[int] = None,
                 meta_template: Optional[Dict] = None,
                 pad_token_id: Optional[int] = None,
                 fastchat_template: Optional[str] = None,
                 stop_words: Optional[str] = [],
                 mode: str = 'none',
                 vision_kwargs: dict = dict(),
                 **other_kwargs):
        super().__init__(
            path,
            max_seq_len,
            tokenizer_only,
            meta_template,
            generation_kwargs,
            False,
        )
        self.logger = AISLogger()
        self.path = path
        self.tokenizer_only = tokenizer_only
        self.template_parser = _get_meta_template(meta_template)
        self.max_seq_len = _get_possible_max_seq_len(max_seq_len, path)
        self.max_out_len = other_kwargs.get('max_out_len', None)
        self._load_tokenizer(tokenizer_path or path, tokenizer_kwargs, pad_token_id)
        if not tokenizer_only:
            self._load_model(path=path, kwargs=model_kwargs, peft_path=peft_path, peft_kwargs=peft_kwargs)
        self.generation_kwargs = generation_kwargs
        self.fastchat_template = fastchat_template
        self.stop_words = list(set(stop_words + self._get_potential_stop_words(path)))
        assert mode in ['none', 'mid']
        self.mode = mode
        self.logger.info(f'using stop words: {self.stop_words}')
        self.latencies, self.counts, self.timestamps = [], [], []
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(path)
        self.min_pixels = vision_kwargs.pop('min_pixels', None)
        self.max_pixels = vision_kwargs.pop('max_pixels', None)
        self.total_pixels = vision_kwargs.pop('total_pixels', None)
        self.fps = vision_kwargs.pop('fps', 2)
        self.nframe = vision_kwargs.pop('nframe', 128)
        self.FRAME_FACTOR = 2

    def handle_perf_result(self, output_filepath, output_filename):
        e2e_latency = max(self.timestamps) - min(self.timestamps)
        return {"Benchmark Duration":{"total":str(round(e2e_latency, 4)) + ' ms'}}

    def _load_tokenizer(self, path: Optional[str], kwargs: dict, pad_token_id: Optional[int] = None):
        from transformers import AutoTokenizer, GenerationConfig

        DEFAULT_TOKENIZER_KWARGS = dict(padding_side='left', truncation_side='left', trust_remote_code=True)
        tokenizer_kwargs = DEFAULT_TOKENIZER_KWARGS
        tokenizer_kwargs.update(kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(path, **tokenizer_kwargs)

        # A patch for some models without pad_token_id
        if pad_token_id is not None:
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f'Using {pad_token_id} as pad_token_id')
            elif self.tokenizer.pad_token_id != pad_token_id:
                self.logger.warning(f'pad_token_id is not consistent. Using {pad_token_id} as pad_token_id')
            self.tokenizer.pad_token_id = pad_token_id
            return
        if self.tokenizer.pad_token_id is not None:
            return
        self.logger.warning('pad_token_id is not set for the tokenizer.')
        generation_config = GenerationConfig.from_pretrained(path)
        if generation_config.pad_token_id is not None:
            self.logger.warning(f'Using {generation_config.pad_token_id} as pad_token_id.')
            self.tokenizer.pad_token_id = generation_config.pad_token_id
            return
        if self.tokenizer.eos_token_id is not None:
            self.logger.warning(f'Using eos_token_id {self.tokenizer.eos_token_id} as pad_token_id.')
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            return
        raise ValueError('pad_token_id is not set for this tokenizer. Please set `pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        from transformers import AutoModel, Qwen2_5_VLForConditionalGeneration

        DEFAULT_MODEL_KWARGS = dict(device_map='auto', trust_remote_code=True)
        model_kwargs = DEFAULT_MODEL_KWARGS
        model_kwargs.update(kwargs)
        model_kwargs = _set_model_kwargs_torch_dtype(model_kwargs)
        self.logger.debug(f'using model_kwargs: {model_kwargs}')
        if is_npu_available():
            model_kwargs['device_map'] = 'npu'
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, **model_kwargs, attn_implementation='flash_attention_2')
        except ValueError:
            self.logger.error("cannot load model, please check it!")

        if peft_path is not None:
            from peft import PeftModel
            peft_kwargs['is_trainable'] = False
            self.model = PeftModel.from_pretrained(self.model, peft_path, **peft_kwargs)

        self.model.eval()

    def _get_potential_stop_words(self, path: Optional[str]):
        from transformers import GenerationConfig
        potential_stop_words = []
        try:
            generation_config = GenerationConfig.from_pretrained(path)
        except:
            generation_config = None
        if generation_config and hasattr(generation_config, 'eos_token_id'):
            if isinstance(generation_config.eos_token_id, int):
                potential_stop_words.append(self.tokenizer.decode(generation_config.eos_token_id))
            else:
                assert isinstance(generation_config.eos_token_id, list)
                for token_id in generation_config.eos_token_id:
                    potential_stop_words.append(self.tokenizer.decode(token_id))
        if self.tokenizer.eos_token is not None:
            potential_stop_words.append(self.tokenizer.eos_token)
        potential_stop_words = list(set(potential_stop_words))
        potential_stop_words = [s for s in potential_stop_words if s]
        return potential_stop_words

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

        generation_kwargs = self.generation_kwargs.copy()
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=text, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        # step-2: conduct model forward to generate output
        start_time = time.perf_counter()
        outputs = self.model.generate(**inputs, **generation_kwargs, max_new_tokens=self.max_out_len)
        end_time = time.perf_counter()
        outputs = outputs[:, inputs['input_ids'].shape[1]:]

        # step-3: decode the output
        decodeds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for stop in stopping_criteria:
            decodeds = [t.split(stop)[0] for t in decodeds]

        if hasattr(self, "do_performance") and self.do_performance:
            self.latencies.append(end_time - start_time)
            self.counts.append(batch_size)
            self.timestamps.extend([end_time, start_time])
            return None
        else:
            return decodeds

    def get_token_len(self, prompt: str) -> int:
        m = _convert_chat_messages([prompt])[0]
        t = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, return_dict=True)
        return len(t['input_ids'])