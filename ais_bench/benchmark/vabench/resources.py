"""Compatibility alias for ais_bench.benchmark.tasks.vabench.resources."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module('ais_bench.benchmark.tasks.vabench.resources')
_sys.modules[__name__] = _implementation
