import math
from typing import List, Optional
import copy
import uuid
from multiprocessing import BoundedSemaphore

from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES
import aiohttp
from tqdm import tqdm

from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_api_inferencer import BaseApiInferencer
from ais_bench.benchmark.openicl.icl_inferencer.output_handler.ppl_inferencer_output_handler import PPLInferencerOutputHandler, PPLRequestOutput, PPLResponseOutput
from ais_bench.benchmark.utils.logging.error_codes import ICLE_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchValueError

@ICL_INFERENCERS.register_module()
class PPLInferencer(BaseApiInferencer):
    """Per-option perplexity (log-likelihood) based inferencer for MCQ tasks.

    This inferencer expects the retriever to provide fields:
    - index: sample id
    - prompt: shared prefix (context + question + options header)
    - options: list[str] candidate answers (continuations)
    - gold: gold label (index or value)
    - data_abbr: abbr for sample
    - max_out_len: not used here but kept for collate compatibility
    """

    def __init__(
        self,
        model_cfg,
        batch_size: Optional[int] = 1,
        mode: Optional[str] = "infer",
        output_json_filepath: Optional[str] = "./icl_inference_output",
        output_json_filename: Optional[str] = 'predictions',
        labels: Optional[List] = None,
        save_every: Optional[int] = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            batch_size=batch_size,
            mode=mode,
            output_json_filepath = output_json_filepath,
            save_every=save_every,
            **kwargs,
        )
        if self.perf_mode:
            raise AISBenchValueError(ICLI_CODES.PERF_MODE_NOT_SUPPORTED_FOR_PPL_INFERENCE, f"Perf mode is not supported for PPL inference")
        if model_cfg.get("stream") == True:
            raise AISBenchValueError(ICLI_CODES.STREAM_MODE_NOT_SUPPORTED_FOR_PPL_INFERENCE, f"Stream mode is not supported for PPL inference")
        self.labels = labels
        # Reuse generation output handler in accuracy mode (stores simple prediction string/int)
        self.output_handler = PPLInferencerOutputHandler(perf_mode=self.perf_mode, 
                                                        save_every=self.save_every)

    def get_data_list(
        self,
        retriever: BaseRetriever,
        ice_template=None,
        prompt_template=None,
        normalizing_str: Optional[str] = None,
    ):
        data_list = []
        ice = []
        data_abbr = retriever.dataset.abbr
        ice_idx_list = retriever.retrieve()
        if self.labels is None:
            labels = retriever.get_labels()
        else:
            labels = self.labels
        
        # Generate in-context examples for testing inputs
        for idx in range(len(ice_idx_list)):
            ice.append(retriever.generate_ice(ice_idx_list[idx]))

        for idx in range(len(ice_idx_list)):
            token_num_list = []
            tmp_data = {"index": idx, "qa": {}, "gold": None}
            data_list.append(tmp_data) 

            for label in labels:
                prompt_kwargs = {
                    'idx': idx,
                    'ice': ice[idx],
                    'label': label,
                }
                prompt = retriever.generate_label_prompt(**prompt_kwargs)
                prompt_token_num = self.model.get_token_len_from_template(prompt, mode='ppl')
                token_num_list.append(prompt_token_num)
                parsed_prompt = self.model.parse_template(prompt, mode="ppl")
                data_list[idx]["qa"][label] = {"prompt": parsed_prompt, "token_num": prompt_token_num}
        max_out_lens = retriever.dataset_reader.get_max_out_len()
        if max_out_lens is not None:
            self.logger.warning(f"Dataset-specified max_out_len has highest priority, use dataset-specified max_out_len")
            for index, max_out_len in enumerate(max_out_lens):
                data_list[index]["max_out_len"] = max_out_len
        else:
            for idx in range(len(data_list)):
                data_list[idx]["max_out_len"] = self.model.max_out_len
        gold_answer = retriever.get_gold_ans()
        if len(data_list) != len(gold_answer):
            raise AISBenchValueError(ICLE_CODES.PREDICTION_LENGTH_MISMATCH, f"The length of data_list and gold_answer is not the same: {len(data_list)} != {len(gold_answer)}")
        for idx in range(len(data_list)):
            data_list[idx]["gold"] = gold_answer[idx]
            data_list[idx]["data_abbr"] = data_abbr
        return data_list

    def _calc_prediction(self, ppl_list: List[dict]) -> Optional[str]:
        if not ppl_list:
            return None
        label, min_ppl = None, float('inf')
        for item in ppl_list:
            if item["ppl"] < min_ppl:
                label = item["label"]
                min_ppl = item["ppl"]
        return label

    async def do_request(
        self, data: dict, token_bucket: BoundedSemaphore, session: aiohttp.ClientSession
    ):
        """Execute a single inference request.
        
        Args:
            data: Dictionary containing request data
            token_bucket: Semaphore for rate limiting
            session: HTTP session for the request
        """
        data = copy.deepcopy(data)
        idx = data.pop("index")
        qa = data.pop("qa")
        gold = data.pop("gold")
        max_out_len = data.pop("max_out_len")
        data_abbr = data.pop("data_abbr")
        uid = uuid.uuid4().hex[:8]
        ppl_list = []
        resp_output = PPLResponseOutput(self.perf_mode)
        resp_output.success = True
        for label, qa_data in qa.items():
            prompt = qa_data["prompt"]
            output = PPLRequestOutput(self.perf_mode)
            output.uuid = uid
            await self.status_counter.post()
            await self.model.get_ppl(prompt, max_out_len, output, session=session, **data)
            await self.status_counter.finish()
            resp_output.input.append(output.input)
            if output.success:
                await self.status_counter.rev()
                ppl_list.append({"label": label, "ppl": output.ppl if output.success else 0})
                resp_output.origin_prompt_logprobs.append(output.origin_prompt_logprobs)
            else:
                await self.status_counter.failed()
                resp_output.success = False
                resp_output.error_info = output.error_info
                break
        resp_output.uuid = uid
        if resp_output.success:
            resp_output.content = self._calc_prediction(ppl_list)
        else:
            resp_output.content = None
        resp_output.label_ppl_list = ppl_list
        await self.status_counter.case_finish()
        await self.output_handler.report_cache_info(idx, prompt, resp_output, data_abbr, gold)