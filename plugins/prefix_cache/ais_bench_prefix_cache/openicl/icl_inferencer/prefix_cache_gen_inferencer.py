from __future__ import annotations

import asyncio
from collections import defaultdict

from ais_bench.benchmark.openicl.icl_inferencer.icl_gen_inferencer import GenInferencer
from ais_bench.benchmark.registry import ICL_INFERENCERS


class LaneSequencer:
    def __init__(self):
        self._conditions: dict[tuple[str, int], asyncio.Condition] = defaultdict(asyncio.Condition)
        self._next: dict[tuple[str, int], int] = defaultdict(int)

    async def wait_turn(self, lane: tuple[str, int], sequence: int) -> None:
        condition = self._conditions[lane]
        async with condition:
            await condition.wait_for(lambda: self._next[lane] == sequence)

    async def complete(self, lane: tuple[str, int]) -> None:
        condition = self._conditions[lane]
        async with condition:
            self._next[lane] += 1
            condition.notify_all()


@ICL_INFERENCERS.register_module()
class PrefixCacheGenInferencer(GenInferencer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lane_sequencer = LaneSequencer()

    def get_data_list(self, retriever):
        data_list = super().get_data_list(retriever)
        source = retriever.dataset_reader.dataset["test"]
        if len(data_list) != len(source):
            raise ValueError("Prefix Cache Dataset order changed before inference")
        for index, data in enumerate(data_list):
            row = source[index]
            for field in ("dp_rank", "group_id", "lane_sequence", "cache_mode"):
                data[field] = row[field]
        return data_list

    async def do_request(self, data, token_bucket, session):
        if data.get("cache_mode") != "cold":
            return await super().do_request(data, token_bucket, session)
        lane = (str(data["group_id"]), int(data["dp_rank"]))
        sequence = int(data["lane_sequence"])
        await self._lane_sequencer.wait_turn(lane, sequence)
        try:
            return await super().do_request(data, token_bucket, session)
        finally:
            await self._lane_sequencer.complete(lane)
