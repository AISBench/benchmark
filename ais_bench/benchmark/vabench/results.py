"""Compatibility alias for ais_bench.benchmark.summarizers.vabench_results."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module('ais_bench.benchmark.summarizers.vabench_results')
_sys.modules[__name__] = _implementation
