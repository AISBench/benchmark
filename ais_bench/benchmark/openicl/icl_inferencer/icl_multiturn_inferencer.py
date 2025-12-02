"""Multiturn Direct Generation Inferencer."""

from multiprocessing import BoundedSemaphore
from typing import List, Optional
import uuid
import copy
import asyncio
import aiohttp

from ais_bench.benchmark.models.output import RequestOutput
from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_api_inferencer import BaseApiInferencer
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_local_inferencer import BaseLocalInferencer
from ais_bench.benchmark.openicl.icl_inferencer.output_handler.gen_inferencer_output_handler import GenInferencerOutputHandler
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES
from ais_bench.benchmark.utils.logging.exceptions import ParameterValueError


@ICL_INFERENCERS.register_module()
class MultiTurnGenInferencer(BaseApiInferencer, BaseLocalInferencer):
    """Multiturn Generation Inferencer class to directly evaluate by generation.

    Attributes:
        model_cfg
        batch_size (:obj:`int`, optional): Batch size for the
            :obj:`DataLoader`.
        output_json_filepath (:obj:`str`, optional): File path for output
            `JSON` file.
        gen_field_replace_token (:obj:`str`, optional): Used to replace the
            generation field token when generating prompts.
        save_every (:obj:`int`, optional): Save intermediate results every
            `save_every` iters. Defaults to 1.
    """

    def __init__(
        self,
        model_cfg,
        stopping_criteria: List[str] = [],
        batch_size: Optional[int] = 1,
        mode: Optional[str] = "infer",
        gen_field_replace_token: Optional[str] = "",
        output_json_filepath: Optional[str] = "./icl_inference_output",
        save_every: Optional[int] = 1,
        infer_mode: str = "every",
        **kwargs,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            batch_size=batch_size,
            mode=mode,
            output_json_filepath=output_json_filepath,
            save_every=save_every,
            **kwargs,
        )
        self.stopping_criteria = list(stopping_criteria) if stopping_criteria else []
        self.gen_field_replace_token = gen_field_replace_token or ""

        self.output_handler = GenInferencerOutputHandler(perf_mode=self.perf_mode, save_every=self.save_every)
        self.infer_mode = infer_mode
        self.logger.info(f"Multiturn Inferencer infer with mode: {self.infer_mode}")

    async def do_request(
        self, data: dict, token_bucket: BoundedSemaphore, session: aiohttp.ClientSession
    ):
        """Execute a single inference request.

        Args:
            data: Dictionary containing request.
            token_bucket: Semaphore for rate limiting
            session: HTTP session for the request
        """
        if self.infer_mode == "every":
            await self.infer_every(data, session)
        elif self.infer_mode == "last":
            await self.infer_last(data, session)
        elif self.infer_mode == "every_with_gt":
            await self.infer_every_with_gt(data, session)
        else:
            raise ParameterValueError(ICLI_CODES.MULTITRUN_MODE_OUT_OF_RANGE,
                                      f"Multiturn dialogue infer model only supports every、last or every_with_gt, but got {self.infer_mode}")
        await self.status_counter.case_finish()

    async def infer_last(self, data: dict, session: aiohttp.ClientSession):
        """Conducts a single inference on the entire multi-turn dialogue at once.

        Args:
            data: Dictionary containing request
            session: HTTP session for the request
        """
        data = copy.deepcopy(data)
        index = data.pop("index")
        chat = data.pop("prompt")
        data_abbr = data.pop("data_abbr")
        max_out_len = data.pop("max_out_len")
        gold = data.pop("gold", None)
        uid = str(uuid.uuid4()).replace("-", "")
        start_prompt, chat, end_prompt = chat[:1], chat[1:-1], chat[-1:]
        bot_indices = [i for i, item in enumerate(chat) if item['role'] == 'BOT']
        history = PromptList(chat[:bot_indices[-1]])
        history = await asyncio.to_thread(self.model.parse_template, PromptList(start_prompt + history + end_prompt), mode="gen")
        turn_id = 0
        output = RequestOutput(self.perf_mode)
        output.turn_id, output.uuid = turn_id, uid

        max_out_len = max_out_len[-1] if isinstance(max_out_len, list) else max_out_len
        await self.status_counter.post()
        await self.model.generate(history, max_out_len, output, session=session, **data)
        if output.success:
            await self.status_counter.rev()
        else:
            await self.status_counter.failed()
        await self.status_counter.finish()
        await self.output_handler.report_cache_info(index, history, output, data_abbr, gold)


    async def infer_every(self, data: dict, session: aiohttp.ClientSession):
        """Performs turn-by-turn inference, concatenating the model's previous output into the context.

        Args:
            data: Dictionary containing request
            session: HTTP session for the request
        """
        data = copy.deepcopy(data)
        index = data.pop("index")
        chat = data.pop("prompt")
        data_abbr = data.pop("data_abbr")
        max_out_len = data.pop("max_out_len")
        gold = data.pop("gold", None)
        uid = str(uuid.uuid4()).replace("-", "")
        start_prompt, chat, end_prompt = chat[:1], chat[1:-1], chat[-1:]
        bot_indices = [i for i, item in enumerate(chat) if item['role'] == 'BOT']
        turn_id = 0
        for i in bot_indices:
            # TODO: use thread to parse the template
            history = await asyncio.to_thread(self.model.parse_template, PromptList(start_prompt + chat[:i] + end_prompt), mode="gen")
            output = RequestOutput(self.perf_mode)
            output.turn_id, output.uuid = turn_id, uid
            max_out_len = max_out_len[turn_id] if isinstance(max_out_len, list) else max_out_len
            await self.status_counter.post()
            await self.model.generate(history, max_out_len, output, session=session, **data)
            turn_id += 1
            if output.success:
                await self.status_counter.rev()
                chat[i]["prompt"] = output.content
                await self.status_counter.finish()
                await self.output_handler.report_cache_info(index, history, output, data_abbr, gold)
            else: # Exit the current for loop; if it fails, subsequent rounds will no longer send requests.
                await self.status_counter.failed()
                self.logger.warning("Request failed; subsequent rounds of conversation are terminated.")
                await self.status_counter.finish()
                await self.output_handler.report_cache_info(index, history, output, data_abbr, gold)
                break


    async def infer_every_with_gt(self, data: dict, session: aiohttp.ClientSession):
        """Carries out turn-by-turn inference,
            always appending the ground-truth response from the dataset as context.

        Args:
            data: Dictionary containing request
            session: HTTP session for the request
        """
        data = copy.deepcopy(data)
        index = data.pop("index")
        chat = data.pop("prompt")
        data_abbr = data.pop("data_abbr")
        max_out_len = data.pop("max_out_len")
        gold = data.pop("gold", None)
        uid = str(uuid.uuid4()).replace("-", "")
        start_prompt, chat, end_prompt = chat[:1], chat[1:-1], chat[-1:]
        bot_indices = [i for i, item in enumerate(chat) if item['role'] == 'BOT']
        turn_id = 0
        for i in bot_indices:
            history = await asyncio.to_thread(self.model.parse_template, PromptList(start_prompt + chat[:i] + end_prompt), mode="gen")
            output = RequestOutput(self.perf_mode)
            output.turn_id, output.uuid = turn_id, uid
            max_out_len = max_out_len[turn_id] if isinstance(max_out_len, list) else max_out_len
            await self.status_counter.post()
            await self.model.generate(history, max_out_len, output, session=session, **data)
            turn_id += 1
            if output.success:
                await self.status_counter.rev()
                await self.status_counter.finish()
                await self.output_handler.report_cache_info(index, history, output, data_abbr, gold)
            else: # Exit the current for loop; if it fails, subsequent rounds will no longer send requests.
                await self.status_counter.failed()
                self.logger.warning("Request failed; subsequent rounds of conversation are terminated.")
                await self.status_counter.finish()
                await self.output_handler.report_cache_info(index, history, output, data_abbr, gold)
                break

    def get_data_list(
        self,
        retriever: BaseRetriever,
    ):
        """Generate data list for inference.

        Args:
            retriever: The retriever instance to get data from

        Returns:
            List of data dictionaries for inference
        """
        # TODO: reuse mode, load tmp results only and infer unprocessed data
        data_abbr = retriever.dataset.abbr
        ice_idx_list = retriever.retrieve()
        prompt_list = []
        for idx, ice_idx in enumerate(ice_idx_list):
            ice = retriever.generate_ice(ice_idx)
            dialog_prompts = retriever.generate_prompt_for_generate_task(
                idx,
                ice,
                gen_field_replace_token=self.gen_field_replace_token,
            )

            prompt_list.append(dialog_prompts)
        gold_ans = retriever.get_gold_ans()
        data_list = []
        for index, prompt in enumerate(prompt_list):
            # Required field information
            data_list.append(
                {
                    "prompt": prompt,
                    "data_abbr": data_abbr,
                    "index": index,
                    "max_out_len": self.model.max_out_len,
                }
            )
        if gold_ans is not None:
            for index, gold in enumerate(gold_ans):
                data_list[index]["gold"] = gold

        # Dataset-specified max_out_len has highest priority
        max_out_lens = retriever.dataset_reader.get_max_out_len()
        if max_out_lens is not None:
            self.logger.warning(f"Dataset-specified max_out_len has highest priority, use dataset-specified max_out_len")
            for index, max_out_len in enumerate(max_out_lens):
                data_list[index]["max_out_len"] = (
                    max_out_len if max_out_len else self.model.max_out_len
                )

        return data_list