import os, sys
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from ais_bench.benchmark.models.local_models.base import BaseModel
from ais_bench.benchmark.models import APITemplateParser
from ais_bench.benchmark.registry import MODELS

from mindspore import Tensor, Model
from mindformers import  MindFormerConfig, build_context
from mindformers.models import build_network
from mindformers.core.parallel_config import build_parallel_config
from mindformers.utils.load_checkpoint_utils import get_load_path_after_hf_convert
from mindformers.trainer.utils import transform_and_load_checkpoint



class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        batch_size: int,
    ):
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence,
                                             add_special_tokens=False)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # compare the last len(stop) tokens
        lookback_ids_batch = input_ids[:, -self.sequence_id_len:]
        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)
        for i, done in enumerate(self.done_tracker):
            if done:
                continue
            self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def drop_error_generation_kwargs(generation_kwargs: dict) -> dict:
    for key in ['is_synthetic', 'batch_size', 'do_performance']:
        if key in generation_kwargs:
            generation_kwargs.pop(key)
    return generation_kwargs


@MODELS.register_module()
class MindFormerModel(BaseModel):

    launcher: str = "msrun"

    def __init__(self,
                 path: str,
                 checkpoint: Optional[str] = None,
                 yaml_cfg_file: Optional[str] = None,
                 batch_size: int = 1,
                 max_seq_len: int = 2048,
                 tokenizer_path: Optional[str] = None,
                 tokenizer_kwargs: dict = dict(),
                 tokenizer_only: bool = False,
                 generation_kwargs: dict = dict(),
                 meta_template: Optional[Dict] = None,
                 extract_pred_after_decode: bool = False,
                 batch_padding: bool = False,
                 pad_token_id: Optional[int] = None,
                 mode: str = 'none',
                 use_fastchat_template: bool = False,
                 end_str: Optional[str] = None,
                 **kwargs):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         tokenizer_only=tokenizer_only,
                         meta_template=meta_template)
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.pretrained_model_path = path
        if mode not in ['none', 'mid']:
            raise ValueError(f"mode must be 'none' or 'mid', but got {mode}")
        self.mode = mode
        if not yaml_cfg_file:
            raise ValueError('`yaml_cfg_file` is required for MindFormerModel') 
        self.config = MindFormerConfig(yaml_cfg_file)
        self.checkpoint = checkpoint
        self._load_tokenizer(path=path,
                             tokenizer_path=tokenizer_path,
                             tokenizer_kwargs=tokenizer_kwargs)
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(self.config, self.batch_size, self.max_seq_len)
        self.generation_kwargs = generation_kwargs
        self.use_fastchat_template = use_fastchat_template
        self.end_str = end_str

    def _load_tokenizer(self, path: str, tokenizer_path: Optional[str],
                        tokenizer_kwargs: dict):
        from transformers import AutoTokenizer, GenerationConfig

        DEFAULT_TOKENIZER_KWARGS = dict(padding_side='left', truncation_side='left', trust_remote_code=True)
        kwargs = DEFAULT_TOKENIZER_KWARGS.copy()
        kwargs.update(tokenizer_kwargs)

        load_path = tokenizer_path if tokenizer_path else path
        self.tokenizer = AutoTokenizer.from_pretrained(load_path, **kwargs)

        pad_token_id = self.pad_token_id

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

        try:
            generation_config = GenerationConfig.from_pretrained(path)
        except Exception:
            generation_config = None

        if generation_config and generation_config.pad_token_id is not None:
            self.logger.warning(f'Using {generation_config.pad_token_id} as pad_token_id.')
            self.tokenizer.pad_token_id = generation_config.pad_token_id
            return
        if self.tokenizer.eos_token_id is not None:
            self.logger.warning(f'Using eos_token_id {self.tokenizer.eos_token_id} as pad_token_id.')
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            return
        raise ValueError('pad_token_id is not set for this tokenizer. Please set `pad_token_id={PAD_TOKEN_ID}` in model_cfg.')

    def _set_config_from_yaml(self):
        if self.checkpoint is not None:
            self.config.load_checkpoint = self.checkpoint
        elif self.checkpoint is None and self.config.load_checkpoint is None:
            self.config.load_checkpoint = self.path
        self.config.model.pretrained_model_dir = self.pretrained_model_path
        self.config.model.model_config.seq_length = self.max_seq_len
        build_context(self.config)
        build_parallel_config(self.config)

    def _load_model(self, config, batch_size, max_seq_len):

        self._set_config_from_yaml()
        try:   
            self.model = build_network(
                config.model,
                default_args={
                "parallel_config": config.parallel_config,
                "moe_config": config.moe_config
            })
            self.logger.info("..........Network Built Successfully..........")
            self.model.set_train(False)
            config.load_checkpoint = get_load_path_after_hf_convert(config, self.model)
            self.logger.info(f"load checkpoint path : {config.load_checkpoint}")
            run_mode = config.get("run_mode", None)
            if run_mode == "predict":
                self.model.load_weights(config.load_checkpoint)
            else:
                model = Model(self.model)
                input_ids = Tensor(np.ones((batch_size, max_seq_len), dtype=np.int32))
                infer_data = self.model.prepare_inputs_for_predict_layout(input_ids)
                transform_and_load_checkpoint(config, model, self.model, infer_data, do_eval=True)

            self.logger.info("..........Checkpoint Load Successfully..........")
        except ValueError as e:
            raise ValueError('Failed to load MindFormers model, please check configuration') from e


    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 min_out_len: Optional[int] = None,
                 stopping_criteria: List[str] = [],
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.
            min_out_len (Optional[int]): The minimum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        generation_kwargs = kwargs.copy()
        generation_kwargs.update(self.generation_kwargs)
        
        messages = list(inputs)
        batch_size = len(messages)
        prompt_char_lens = None
        
        if self.extract_pred_after_decode:
            prompt_char_lens = [len(text) for text in messages]

        if self.use_fastchat_template:
            try:
                from fastchat.model import get_conversation_template
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    'Fastchat is not implemented. You can use '
                    "'pip install \"fschat[model_worker,webui]\"' "
                    'to implement fastchat.')
            for idx, text in enumerate(messages):
                conv = get_conversation_template('vicuna')
                conv.append_message(conv.roles[0], text)
                conv.append_message(conv.roles[1], None)
                messages[idx] = conv.get_prompt()
        if self.mode == 'mid':
            assert len(messages) == 1
            tokens = self.tokenizer(messages, padding=False, truncation=False, return_tensors='np')
            input_ids = tokens['input_ids']
            if input_ids.shape[-1] > self.max_seq_len:
                input_ids = np.concatenate([input_ids[:, : self.max_seq_len // 2], input_ids[:, - self.max_seq_len // 2:]], axis=-1)
            tokens = {'input_ids': input_ids}
        else:
            tokenize_kwargs = dict(
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors='np'
            )
            tokens = self.tokenizer(messages, **tokenize_kwargs)
        
        input_ids = tokens['input_ids']
        if len(messages) > 1:
            attention_mask = tokens.get('attention_mask')
            prompt_token_lens = (
                attention_mask.sum(axis=1).astype(int).tolist()
                if attention_mask is not None else
                [input_ids.shape[1]] * batch_size
            )
        else:
            prompt_token_lens = [len(ids) for ids in input_ids]

        input_ids_tensor = Tensor(input_ids)

        if min_out_len is not None:
            generation_kwargs['min_new_tokens'] = min_out_len
        generation_kwargs['max_new_tokens'] = max_out_len
        generation_kwargs.setdefault('top_k', 1)
        generation_kwargs.setdefault('return_dict_in_generate', False)

        origin_stopping_criteria = list(stopping_criteria)
        if stopping_criteria:
            if self.tokenizer.eos_token is not None:
                stopping_criteria = stopping_criteria + [
                    self.tokenizer.eos_token
                ]
            stopping_list = transformers.StoppingCriteriaList([
                *[
                    MultiTokenEOSCriteria(sequence, self.tokenizer,
                                          input_ids_tensor.shape[0])
                    for sequence in stopping_criteria
                ],
            ])
            generation_kwargs['stopping_criteria'] = stopping_list
        
        generation_kwargs = drop_error_generation_kwargs(generation_kwargs)

        outputs = self.model.generate(input_ids=input_ids_tensor,
                                      **generation_kwargs)

        if isinstance(outputs, dict):
            outputs = outputs.get('sequences', outputs)
            if outputs is None:
                raise ValueError("Model output dictionary is missing 'sequence' key.")

        sequences = [seq.tolist() for seq in outputs]

        if not self.extract_pred_after_decode:
            sequences = [
                seq[prompt_len:]
                for seq, prompt_len in zip(sequences, prompt_token_lens)
            ]

        decodeds = [
            self.tokenizer.decode(seq, skip_special_tokens=True)
            for seq in sequences
        ]

        if self.extract_pred_after_decode and prompt_char_lens is not None:
            decodeds = [
                text[length:]
                for text, length in zip(decodeds, prompt_char_lens)
            ]

        if self.end_str:
            decodeds = [text.split(self.end_str)[0] for text in decodeds]
        if origin_stopping_criteria:
            for token in origin_stopping_criteria:
                decodeds = [text.split(token)[0] for text in decodeds]
        return decodeds
