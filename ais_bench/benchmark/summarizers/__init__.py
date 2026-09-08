"""Lazy summarizer class re-exports.

``default_perf`` pulls in plotly, and the vbench/swebench/oneig summarizers
pull further heavy backends. To keep the CLI importable in the
dependency-isolated agent environment, summarizer classes are imported on
demand via PEP 562 module-level ``__getattr__``.
"""

import importlib as _importlib
from typing import Any

_SUMMARIZER_CLASS_MODULES: dict[str, str] = {
    "DefaultSummarizer": "ais_bench.benchmark.summarizers.default",
    "DefaultSubjectiveSummarizer": "ais_bench.benchmark.summarizers.default_subjective",
    "DefaultPerfSummarizer": "ais_bench.benchmark.summarizers.default_perf",
    "VBenchSummarizer": "ais_bench.benchmark.summarizers.vbench",
    "SWEBenchSummarizer": "ais_bench.benchmark.summarizers.swebench",
    "HarborSummarizer": "ais_bench.benchmark.summarizers.harbor",
    "DeepSWESummarizer": "ais_bench.benchmark.summarizers.deepswe",
    "SWEBenchProSummarizer": "ais_bench.benchmark.summarizers.swebench_pro",
    "OneIGSummarizer": "ais_bench.benchmark.summarizers.oneig",
}


def __getattr__(name: str) -> Any:
    module_path = _SUMMARIZER_CLASS_MODULES.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _importlib.import_module(module_path)
    cls = getattr(module, name)
    globals()[name] = cls
    return cls


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_SUMMARIZER_CLASS_MODULES.keys()))
