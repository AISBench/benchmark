"""Dataset class re-exports (guarded).

Each dataset backend is star-imported into this package namespace. Several
backends require heavyweight optional dependencies (huggingface ``datasets``,
torch, transformers, ...); importing them eagerly would break the
dependency-isolated agent environment. Each import is therefore guarded so a
missing optional dependency only skips that backend. Dataset loading itself
goes through the ``LOAD_DATASET`` registry with dotted paths, which imports
the concrete submodule directly and is unaffected by a skipped re-export.
"""

# flake8: noqa: F401, F403
_DATASET_MODULES = [
    "ais_bench.benchmark.datasets.aime2024",
    "ais_bench.benchmark.datasets.aime2025",
    "ais_bench.benchmark.datasets.aime2026",
    "ais_bench.benchmark.datasets.gsm8k",
    "ais_bench.benchmark.datasets.ceval",
    "ais_bench.benchmark.datasets.boolq",
    "ais_bench.benchmark.datasets.mmlu",
    "ais_bench.benchmark.datasets.gpqa",
    "ais_bench.benchmark.datasets.math",
    "ais_bench.benchmark.datasets.drop_simple_eval",
    "ais_bench.benchmark.datasets.synthetic",
    "ais_bench.benchmark.datasets.mmlu_pro",
    "ais_bench.benchmark.datasets.humaneval",
    "ais_bench.benchmark.datasets.livecodebench",
    "ais_bench.benchmark.datasets.mgsm",
    "ais_bench.benchmark.datasets.piqa",
    "ais_bench.benchmark.datasets.agieval",
    "ais_bench.benchmark.datasets.arc",
    "ais_bench.benchmark.datasets.winogrande",
    "ais_bench.benchmark.datasets.mbpp",
    "ais_bench.benchmark.datasets.hellaswag",
    "ais_bench.benchmark.datasets.triviaqa",
    "ais_bench.benchmark.datasets.cmmlu",
    "ais_bench.benchmark.datasets.humanevalx.humanevalx",
    "ais_bench.benchmark.datasets.bbh",
    "ais_bench.benchmark.datasets.race",
    "ais_bench.benchmark.datasets.textvqa",
    "ais_bench.benchmark.datasets.videobench",
    "ais_bench.benchmark.datasets.vbench",
    "ais_bench.benchmark.datasets.vocalsound",
    "ais_bench.benchmark.datasets.lambada",
    "ais_bench.benchmark.datasets.lcsts",
    "ais_bench.benchmark.datasets.siqa",
    "ais_bench.benchmark.datasets.xsum",
    "ais_bench.benchmark.datasets.sharegpt",
    "ais_bench.benchmark.datasets.mtbench",
    "ais_bench.benchmark.datasets.longbench",
    "ais_bench.benchmark.datasets.longbenchv2",
    "ais_bench.benchmark.datasets.bfcl.bfcl",
    "ais_bench.benchmark.datasets.custom",
    "ais_bench.benchmark.datasets.infovqa",
    "ais_bench.benchmark.datasets.docvqa",
    "ais_bench.benchmark.datasets.omnidocbench.omnidocbench",
    "ais_bench.benchmark.datasets.mm_custom",
    "ais_bench.benchmark.datasets.mmmu",
    "ais_bench.benchmark.datasets.mmmu_pro",
    "ais_bench.benchmark.datasets.csl",
    "ais_bench.benchmark.datasets.chid",
    "ais_bench.benchmark.datasets.huggingface",
    "ais_bench.benchmark.datasets.cluewsc",
    "ais_bench.benchmark.datasets.eprstmt",
    "ais_bench.benchmark.datasets.tnews",
    "ais_bench.benchmark.datasets.videomme",
    "ais_bench.benchmark.datasets.mathvision",
    "ais_bench.benchmark.datasets.mmstar",
    "ais_bench.benchmark.datasets.dapo_math",
    "ais_bench.benchmark.datasets.mooncake_trace",
    "ais_bench.benchmark.datasets.swebench",
    "ais_bench.benchmark.datasets.swebench_pro",
    "ais_bench.benchmark.datasets.refcoco",
    "ais_bench.benchmark.datasets.hle",
    "ais_bench.benchmark.datasets.aa_lcr",
    "ais_bench.benchmark.datasets.realworldqa",
    "ais_bench.benchmark.datasets.oneig",
    "ais_bench.benchmark.datasets.geometry3k",
]

for _mod_name in _DATASET_MODULES:
    try:
        exec(f"from {_mod_name} import *", globals())  # noqa: S102
    except ImportError:
        # optional dependency missing (e.g. huggingface datasets / torch);
        # skip this backend, registry-based loading still works per module
        pass

try:
    from ais_bench.benchmark.datasets.humanevalx import (  # noqa: F401
        humanevalx,
        humaneval_x_eval,
        humaneval_x_utils,
    )
except ImportError:
    pass

del _mod_name, _DATASET_MODULES
