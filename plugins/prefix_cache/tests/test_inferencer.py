import asyncio
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


class _Registry:
    def register_module(self):
        return lambda cls: cls


class _FakeAISLogger:
    def debug(self, *args, **kwargs):
        pass


class _FakeGenInferencer:
    events = []

    async def do_request(self, data, token_bucket, session):
        sequence = int(data["lane_sequence"])
        input_length = int(data["input_length"])
        self.events.append(("start", sequence, input_length))
        await asyncio.sleep(0)
        self.events.append(("end", sequence, input_length))
        return sequence


def _load_inferencer_module():
    module_path = (
        Path(__file__).parents[1]
        / "ais_bench_prefix_cache"
        / "openicl"
        / "icl_inferencer"
        / "prefix_cache_gen_inferencer.py"
    )
    module_name = "_prefix_cache_gen_inferencer_test_module"
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_file_location(module_name, module_path)
    )

    packages = {
        name: ModuleType(name)
        for name in (
            "ais_bench",
            "ais_bench.benchmark",
            "ais_bench.benchmark.utils",
            "ais_bench.benchmark.utils.logging",
            "ais_bench.benchmark.openicl",
            "ais_bench.benchmark.openicl.icl_inferencer",
        )
    }
    for package in packages.values():
        package.__path__ = []
    gen_module = ModuleType(
        "ais_bench.benchmark.openicl.icl_inferencer.icl_gen_inferencer"
    )
    gen_module.GenInferencer = _FakeGenInferencer
    registry_module = ModuleType("ais_bench.benchmark.registry")
    registry_module.ICL_INFERENCERS = _Registry()
    logger_module = ModuleType("ais_bench.benchmark.utils.logging.logger")
    logger_module.AISLogger = _FakeAISLogger
    stubs = packages | {
        gen_module.__name__: gen_module,
        registry_module.__name__: registry_module,
        logger_module.__name__: logger_module,
        module_name: module,
    }
    with patch.dict(sys.modules, stubs):
        assert module is not None
        assert module.__spec__ is not None
        assert module.__spec__.loader is not None
        module.__spec__.loader.exec_module(module)
    return module


class PrefixCacheInferencerTest(unittest.TestCase):
    def test_cold_lane_sends_short_to_long_despite_reverse_task_creation(self):
        module = _load_inferencer_module()
        inferencer = module.PrefixCacheGenInferencer.__new__(
            module.PrefixCacheGenInferencer
        )
        inferencer._lane_sequencer = module.LaneSequencer()
        _FakeGenInferencer.events = []

        async def exercise():
            tasks = []
            for sequence, input_length in ((2, 32), (1, 16), (0, 8)):
                tasks.append(
                    asyncio.create_task(
                        inferencer.do_request(
                            {
                                "cache_mode": "cold",
                                "group_id": "group-0",
                                "dp_rank": 0,
                                "lane_sequence": sequence,
                                "input_length": input_length,
                            },
                            None,
                            None,
                        )
                    )
                )
                await asyncio.sleep(0)
            self.assertEqual(await asyncio.gather(*tasks), [2, 1, 0])

        asyncio.run(exercise())
        self.assertEqual(
            _FakeGenInferencer.events,
            [
                ("start", 0, 8),
                ("end", 0, 8),
                ("start", 1, 16),
                ("end", 1, 16),
                ("start", 2, 32),
                ("end", 2, 32),
            ],
        )


if __name__ == "__main__":
    unittest.main()
