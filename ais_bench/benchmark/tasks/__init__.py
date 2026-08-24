"""Lazy task class re-exports.

The heavy task backends (swebench / oneig / vbench / openicl) transitively
pull in torch and other heavyweight dependencies at import time. To keep the
CLI importable in the dependency-isolated agent environment, task classes are
imported on demand via PEP 562 module-level ``__getattr__`` instead of eager
``from ... import *`` re-exports.
"""

import importlib as _importlib
from typing import Any

_TASK_CLASS_MODULES: dict[str, str] = {
    "OpenICLEvalTask": "ais_bench.benchmark.tasks.openicl_eval",
    "OpenICLInferTask": "ais_bench.benchmark.tasks.openicl_infer",
    "OpenICLApiInferTask": "ais_bench.benchmark.tasks.openicl_api_infer",
    "SWEBenchInferTask": "ais_bench.benchmark.tasks.swebench.swebench_infer",
    "SWEBenchEvalTask": "ais_bench.benchmark.tasks.swebench.swebench_eval",
    "SWEBenchProInferTask": "ais_bench.benchmark.tasks.swebench_pro.swebench_pro_infer",
    "SWEBenchProEvalTask": "ais_bench.benchmark.tasks.swebench_pro.swebench_pro_eval",
    "VBenchEvalTask": "ais_bench.benchmark.tasks.vbench_eval",
    "OneIGEvalTask": "ais_bench.benchmark.tasks.oneig.oneig_eval",
}


def __getattr__(name: str) -> Any:
    module_path = _TASK_CLASS_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _importlib.import_module(module_path)
    cls = getattr(module, name)
    globals()[name] = cls
    return cls


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_TASK_CLASS_MODULES.keys()))
