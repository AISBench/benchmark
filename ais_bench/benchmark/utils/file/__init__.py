from ais_bench.benchmark.utils.file.file import *  # noqa: F401, F403

# `load_tokenizer` pulls in `transformers` at import time, which would break
# the dependency-isolated agent environment. It is re-exported lazily via
# PEP 562 module-level ``__getattr__``.
_LAZY_FILE_EXPORTS = ("load_tokenizer", "AISTokenizer")


def __getattr__(name):
    if name not in _LAZY_FILE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from ais_bench.benchmark.utils.file.load_tokenizer import (  # noqa: F401
        AISTokenizer,
        load_tokenizer,
    )
    globals()["load_tokenizer"] = load_tokenizer
    globals()["AISTokenizer"] = AISTokenizer
    return globals()[name]


def __dir__():
    return sorted(set(globals().keys()) | set(_LAZY_FILE_EXPORTS))
