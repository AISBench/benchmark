"""Compatibility alias for ais_bench.benchmark.utils.vabench.data."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module('ais_bench.benchmark.utils.vabench.data')
_sys.modules[__name__] = _implementation
