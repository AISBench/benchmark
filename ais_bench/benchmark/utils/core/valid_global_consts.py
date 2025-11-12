import os

import ais_bench.benchmark.global_consts as global_consts
from ais_bench.benchmark.utils.logging import get_logger

logger = get_logger()

def get_max_chunk_size():
    if not isinstance(global_consts.MAX_CHUNK_SIZE, int):
        logger.warning("MAX_CHUNK_SIZE is invalid, using default value 65536B(64KB)")
        return 2**16
    if not (1 <= global_consts.MAX_CHUNK_SIZE <= 2**24):
        logger.warning(
            f"MAX_CHUNK_SIZE is out of range, using default value 65536B(64KB): {global_consts.MAX_CHUNK_SIZE}"
        )
        return 2**16
    return global_consts.MAX_CHUNK_SIZE

def get_request_time_out():
    value = global_consts.REQUEST_TIME_OUT
    if value is None:
        return None
    if not isinstance(value, (int,float)):
        logger.warning("REQUEST_TIME_OUT is invalid, using default value None")
        return None
    if not (0 <= value <= 3600 * 24):
        logger.warning(
            f"REQUEST_TIME_OUT is out of range, using default value None: {value}"
        )
        return None
    return value
