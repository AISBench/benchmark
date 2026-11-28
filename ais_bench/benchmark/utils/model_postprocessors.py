"""Deprecated compatibility shim.

Use ``ais_bench.benchmark.utils.postprocess.model_postprocessors`` instead.
"""

from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.postprocess.model_postprocessors import (  # noqa: F401
    extract_non_reasoning_content,
)

logger = AISLogger()
logger.warning("ais_bench.benchmark.utils.model_postprocessors is deprecated, please use ais_bench.benchmark.utils.postprocess.model_postprocessors instead.")
