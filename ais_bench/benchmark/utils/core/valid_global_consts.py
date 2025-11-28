from ais_bench.benchmark import global_consts
from ais_bench.benchmark.utils.logging import AISLogger

logger = AISLogger()

def get_max_chunk_size():
    """Get validated MAX_CHUNK_SIZE from global constants.
    
    Returns:
        int: Validated chunk size in bytes (default: 65536B/64KB)
        
    Valid range: 1 to 16777216 (16MB)
    """
    if not isinstance(global_consts.MAX_CHUNK_SIZE, int):
        logger.warning(
            f"MAX_CHUNK_SIZE is invalid (type: {type(global_consts.MAX_CHUNK_SIZE).__name__}), "
            f"using default value 65536B (64KB)"
        )
        return 2**16
    if not (1 <= global_consts.MAX_CHUNK_SIZE <= 2**24):
        logger.warning(
            f"MAX_CHUNK_SIZE is out of range [1, {2**24}]: {global_consts.MAX_CHUNK_SIZE}, "
            f"using default value 65536B (64KB)"
        )
        return 2**16
    return global_consts.MAX_CHUNK_SIZE

def get_request_time_out():
    """Get validated REQUEST_TIME_OUT from global constants.
    
    Returns:
        int, float, or None: Validated timeout in seconds (default: None)
        
    Valid range: 0 to 86400 (24 hours), or None for no timeout
    """
    value = global_consts.REQUEST_TIME_OUT
    if value is None:
        return None
    if not isinstance(value, (int, float)):
        logger.warning(
            f"REQUEST_TIME_OUT is invalid (type: {type(value).__name__}), "
            f"expected int/float or None, using default value None"
        )
        return None
    if not (0 <= value <= 3600 * 24):
        logger.warning(
            f"REQUEST_TIME_OUT is out of range [0, {3600 * 24}]: {value}, "
            f"using default value None"
        )
        return None
    return value
