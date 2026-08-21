import ipaddress


def resolve_metrics_urls(models: list[dict]) -> list[str]:
    """Return deduplicated metrics URLs from a list of model configs.

    Args:
        models: List of model configuration dicts, each may contain
            ``host_ip`` (default ``"localhost"``) and ``host_port``
            (default ``8080``).

    Returns:
        Deduplicated list of metrics URL strings such as
        ``"http://10.0.0.1:8080/metrics"``.  IPv6 addresses are
        automatically wrapped in brackets (``[::1]``).  Hostnames are
        used as-is.  If no models are provided, an empty list is returned.
    """
    urls: list[str] = []
    seen: set[str] = set()

    for model in models:
        host = model.get("host_ip", "localhost")
        port = model.get("host_port", 8080)
        host = _normalize_host(host)
        url = f"http://{host}:{port}/metrics"
        if url not in seen:
            seen.add(url)
            urls.append(url)

    return urls


def _normalize_host(host: str) -> str:
    """Wrap IPv6 literals in brackets; leave IPv4 and hostnames unchanged."""
    try:
        ip = ipaddress.ip_address(host)
        if isinstance(ip, ipaddress.IPv6Address):
            return f"[{ip}]"
    except ValueError:
        pass
    return host
