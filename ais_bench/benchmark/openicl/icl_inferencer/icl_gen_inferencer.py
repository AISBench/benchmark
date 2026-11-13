"""Direct Generation Inferencer."""

import copy
import uuid
from multiprocessing import BoundedSemaphore
from typing import List, Optional, Dict

import aiohttp

from ais_bench.benchmark.models.output import RequestOutput
from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_api_inferencer import BaseApiInferencer
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_local_inferencer import BaseLocalInferencer
from ais_bench.benchmark.openicl.icl_inferencer.output_handler.gen_inferencer_output_handler import GenInferencerOutputHandler


@ICL_INFERENCERS.register_module()
class GenInferencer(BaseApiInferencer, BaseLocalInferencer):
    """Generation Inferencer class to directly evaluate by generation.

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

        self.output_handler = GenInferencerOutputHandler(
            perf_mode=self.perf_mode,
            save_every=self.save_every,
            response_anomaly_enabled=self.response_anomaly_enabled,
            response_anomaly_payload_storage=(
                self.response_anomaly_payload_storage
            ),
        )

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
        index = data.pop("index")
        input = data.pop("prompt")
        data_abbr = data.pop("data_abbr")
        max_out_len = data.pop("max_out_len")
        gold = data.pop("gold", None)

        session_id = data.pop("session_id", "")
        data.pop("timestamp", None)
        data.pop("stage", None)
        is_manage_req = False
        raw_agent_hint = data.get("agent_hint", {})
        if isinstance(raw_agent_hint, dict):
            cm = raw_agent_hint.get("context_management", {})
            if isinstance(cm, dict) and cm.get("manage_request") is True:
                is_manage_req = True

        if not getattr(self, "pass_agent_hint", False):
            if is_manage_req:
                self.logger.info(
                    f"[session-aware] skip manage_request since pass_agent_hint=False, "
                    f"session_id={session_id} data_id={index}"
                )
                return
            data.pop("agent_hint", None)

        uid = str(uuid.uuid4()).replace("-", "")
        output = RequestOutput(self.perf_mode)
        output.uuid = uid
        output.data_id = index

        output.session_id = session_id
        if getattr(self, "pass_agent_hint", False):
            self.logger.info(
                f"[agent_hint][generate] data_id={index} session_id={session_id} "
                f"agent_hint_in_data={'agent_hint' in data} "
                f"agent_hint={data.get('agent_hint')}"
            )
            output.extra_perf_data["manage_request"] = is_manage_req

        await self.status_counter.post()
        await self.model.generate(input, max_out_len, output, session=session, **data)
        if output.success:
            await self.status_counter.rev()
        else:
            await self.status_counter.failed()
        await self.status_counter.finish()
        await self.status_counter.case_finish()

        await self.output_handler.report_cache_info(index, input, output, data_abbr, gold)

    def batch_inference(
        self,
        datum,
    ) -> None:
        """Perform batch inference on the given dataloader.

        Args:
            dataloader: DataLoader containing the inference data

        Returns:
            List of inference results
        """
        indexs = datum.pop("index")
        inputs = datum.pop("prompt")
        data_abbrs = datum.pop("data_abbr")
        max_out_lens = datum.pop("max_out_len")
        golds = datum.pop("gold", [None] * len(inputs))
        if self.model.is_api:
            outputs = self.model.generate(inputs, max_out_lens, **datum)
        else:
            outputs = self.model.generate(inputs, self.model.max_out_len, **datum)
        # TODO: save output to json
        for index, input, output, data_abbr, gold in zip(
            indexs, inputs, outputs, data_abbrs, golds
        ):
            self.output_handler.report_cache_info_sync(
                index, input, output, data_abbr, gold
            )

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
        data_abbr = retriever.dataset.abbr
        ice_idx_list = retriever.retrieve()
        prompt_list = []
        for idx, ice_idx in enumerate(ice_idx_list):
            ice = retriever.generate_ice(ice_idx)
            prompt = retriever.generate_prompt_for_generate_task(
                idx,
                ice,
                gen_field_replace_token=self.gen_field_replace_token,
            )
            parsed_prompt = self.model.parse_template(prompt, mode="gen")
            prompt_list.append(parsed_prompt)
        self.logger.info(f"Apply ice template finished")
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
            self.logger.warning("Dataset-specified max_out_len has highest priority, use dataset-specified max_out_len")
            for index, max_out_len in enumerate(max_out_lens):
                data_list[index]["max_out_len"] = max_out_len if max_out_len else self.model.max_out_len

        timestamps = retriever.dataset_reader.get_timestamp()
        if timestamps is not None:
            for index, timestamp in enumerate(timestamps):
                if isinstance(timestamp, (int, float)):
                    data_list[index]["timestamp"] = timestamp / 1000 # ms to s

        raw_test = None
        try:
            raw_reader = retriever.dataset_reader.dataset
            raw_test = raw_reader['test'] if 'test' in raw_reader else raw_reader
        except Exception:
            raw_test = None

        # Columns that are either consumed into the prompt or are control
        # fields and therefore must NOT be forwarded into the request body.
        _consumed = {"prompt", "question", "data_abbr", "index", "max_out_len",
                     "gold", "timestamp", "session_id", "answer"}
        if raw_test is not None and len(raw_test) > 0:
            for col in raw_test.column_names:
                if col in _consumed:
                    continue
                try:
                    values = raw_test[col]
                    for index, val in enumerate(values):
                        if index < len(data_list):
                            data_list[index][col] = val
                except Exception:
                    continue

        # ---- attach session_id (top-level column OR agent_hint.session_id) ----
        # Priority: explicit top-level "session_id" column > agent_hint.session_id
        session_ids = retriever.dataset_reader.get_session_id()
        if session_ids is not None:
            for index, sid in enumerate(session_ids):
                data_list[index]["session_id"] = str(sid) if sid is not None else ""
        elif raw_test is not None and len(raw_test) > 0 and \
                "agent_hint" in raw_test.column_names:
            # Auto-extract session_id from nested agent_hint.session_id
            try:
                agent_hints = raw_test["agent_hint"]
                for index, ah in enumerate(agent_hints):
                    if index >= len(data_list):
                        break
                    sid = ""
                    if isinstance(ah, dict):
                        sid = ah.get("session_id") or ""
                    data_list[index]["session_id"] = str(sid)
                self.logger.info(
                    "[session-aware] auto-extracted session_id from agent_hint.session_id"
                )
            except Exception as e:
                self.logger.warning(
                    f"[session-aware] failed to extract session_id from agent_hint: {e}"
                )
        else:
            # No session info at all -> fall back to per-request (blank session)
            for index in range(len(data_list)):
                data_list[index]["session_id"] = ""

        # ---- attach stage (top-level column OR agent_hint.stage) ----
        # stage marks a request's position within its session:
        #   start / stage-N / end. It drives session end-of-life decisions in
        # the session-aware worker loop (a session ends at stage=end).
        stages = None
        if raw_test is not None and len(raw_test) > 0:
            if "stage" in raw_test.column_names:
                stages = raw_test["stage"]
            elif "agent_hint" in raw_test.column_names:
                try:
                    agent_hints = raw_test["agent_hint"]
                    stages = [
                        (ah.get("stage") if isinstance(ah, dict) else None)
                        for ah in agent_hints
                    ]
                except Exception:
                    stages = None
        if stages is not None:
            for index, st in enumerate(stages):
                if index < len(data_list):
                    data_list[index]["stage"] = str(st) if st is not None else ""

        # Validate per-session start/end marking (non-fatal; warn only).
        session_stage_map: Dict[str, List[str]] = {}
        for sample in data_list:
            sid = sample.get("session_id", "")
            session_stage_map.setdefault(sid, []).append(sample.get("stage", ""))
        for sid, stages_of_sid in session_stage_map.items():
            if not stages_of_sid:
                continue
            # Filter empty stages (missing markers) for inspection.
            meaningful = [s for s in stages_of_sid if s]
            if not meaningful:
                self.logger.warning(
                    f"[session-aware] session '{sid}' has no 'stage' markers "
                    "(no start/end). It cannot end based on stage; will end on queue drain."
                )
                continue
            first = meaningful[0].lower()
            last = meaningful[-1].lower()
            if first != "start":
                self.logger.warning(
                    f"[session-aware] session '{sid}' first meaningful stage is "
                    f"'{first}', expected 'start'."
                )
            if last != "end":
                self.logger.warning(
                    f"[session-aware] session '{sid}' last meaningful stage is "
                    f"'{last}', expected 'end'."
                )
        return data_list
