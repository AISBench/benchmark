"""Compatibility alias for ais_bench.benchmark.tasks.vabench.worker."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module('ais_bench.benchmark.tasks.vabench.worker')
if __name__ == '__main__':
    raise SystemExit(_implementation.main())
_sys.modules[__name__] = _implementation
