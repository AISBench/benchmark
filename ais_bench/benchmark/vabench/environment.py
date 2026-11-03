"""Compatibility alias for ais_bench.tools.vabench.environment."""
import importlib as _importlib
import sys as _sys

_implementation = _importlib.import_module('ais_bench.tools.vabench.environment')
if __name__ == '__main__':
    raise SystemExit(_implementation.main())
_sys.modules[__name__] = _implementation
