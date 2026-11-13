"""AISBench P1 adapter for SWE-bench DinD.

Implements:
- ``SwebenchDindTask`` (subclass of AISBench ``BaseTask``)
- ``HarborTaskCompat`` (translates harbor JobResult → AISBench schema)
- ``install()`` (symlinks this package into aisbench's runtime/)
"""
from .install import install
from .result_writer import (
    write_result,
    read_harbor_result,
    SCHEMA_DOC,
)

__all__ = ["install", "write_result", "read_harbor_result", "SCHEMA_DOC"]