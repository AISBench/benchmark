import aiohttp

from ais_bench.benchmark.spec_decode.snapshot import SpecDecodeSnapshot, parse_spec_decode_metrics
from ais_bench.benchmark.utils.logging.logger import AISLogger

_METRICS_FETCH_TIMEOUT = aiohttp.ClientTimeout(total=5)

logger = AISLogger()


async def fetch_spec_decode_metrics_with_error(
    metrics_url: str,
) -> tuple[SpecDecodeSnapshot | None, str | None]:
    """GET {metrics_url} and return (snapshot, error_message) tuple.

    Returns (None, reason) on any failure so callers can distinguish
    "server not enabled" from "network error" for N/A display.
    """
    try:
        async with aiohttp.ClientSession(timeout=_METRICS_FETCH_TIMEOUT, trust_env=True) as s:
            async with s.get(metrics_url) as response:
                if response.status != 200:
                    msg = f"Metrics endpoint returned HTTP {response.status}"
                    logger.debug("%s for %s", msg, metrics_url)
                    return None, msg
                text = await response.text()

            snapshot = parse_spec_decode_metrics(text)
            if snapshot is None:
                msg = "No spec decode metrics found on server"
                logger.debug("%s (%s)", msg, metrics_url)
                return None, msg
            return snapshot, None
    except Exception as e:
        msg = f"Failed to fetch metrics from {metrics_url}: {e}"
        logger.debug(msg)
        return None, msg
