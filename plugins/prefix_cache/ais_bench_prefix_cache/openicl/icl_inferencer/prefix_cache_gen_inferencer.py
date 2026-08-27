from __future__ import annotations

import asyncio
from collections import defaultdict

from ais_bench.benchmark.openicl.icl_inferencer.icl_gen_inferencer import GenInferencer
from ais_bench.benchmark.registry import ICL_INFERENCERS


class LaneSequencer:
    """按 lane 串行化同组请求的发送顺序（cold 模式专用）。

    同一 lane（group × DP rank）上的请求必须严格按 lane_sequence 依次发送，
    才能保证后续请求命中前序请求写入的缓存前缀。用 asyncio.Condition 实现
    类似"红绿灯"的排他放行。
    """

    def __init__(self):
        # 每个 lane 一个条件变量与"下一个允许的序号"游标。
        self._conditions: dict[tuple[str, int], asyncio.Condition] = defaultdict(asyncio.Condition)
        self._next: dict[tuple[str, int], int] = defaultdict(int)

    async def wait_turn(self, lane: tuple[str, int], sequence: int) -> None:
        """阻塞等待，直到 lane 的放行序号等于当前请求的 sequence。"""
        condition = self._conditions[lane]
        async with condition:
            await condition.wait_for(lambda: self._next[lane] == sequence)

    async def complete(self, lane: tuple[str, int]) -> None:
        """标记当前请求已完成，推进 lane 的放行序号并唤醒等待者。"""
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
        # 把每行请求的路由元数据（DP rank / group / lane / 缓存模式）透传到数据项，
        # 供 do_request 决定是否需要串行放行。
        for index, data in enumerate(data_list):
            row = source[index]
            for field in ("dp_rank", "group_id", "lane_sequence", "cache_mode"):
                data[field] = row[field]
        return data_list

    async def do_request(self, data, token_bucket, session):
        # 仅 cold 模式需要串行：按 lane 放行，保证同组请求命中彼此写入的前缀缓存。
        if data.get("cache_mode") != "cold":
            return await super().do_request(data, token_bucket, session)
        lane = (str(data["group_id"]), int(data["dp_rank"]))
        sequence = int(data["lane_sequence"])
        await self._lane_sequencer.wait_turn(lane, sequence)
        try:
            return await super().do_request(data, token_bucket, session)
        finally:
            # 无论成功失败都要放行下一个请求，避免死锁。
            await self._lane_sequencer.complete(lane)

