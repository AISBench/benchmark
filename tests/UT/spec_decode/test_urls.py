"""Unit tests for spec_decode.urls — metrics URL resolution and IPv6 handling."""

from ais_bench.benchmark.spec_decode.urls import resolve_metrics_urls


class TestResolveMetricsUrls:
    """Tests for resolve_metrics_urls – the URL extraction and dedup logic."""

    # ------------------------------------------------------------------
    #  IPv4 – should NOT get brackets
    # ------------------------------------------------------------------
    def test_ipv4_no_brackets(self):
        models = [{"host_ip": "10.0.0.1", "host_port": 8080}]
        urls = resolve_metrics_urls(models)
        assert urls == ["http://10.0.0.1:8080/metrics"]

    # ------------------------------------------------------------------
    #  IPv6 – should be wrapped in brackets
    # ------------------------------------------------------------------
    def test_ipv6_full_address(self):
        models = [{"host_ip": "2001:db8::1", "host_port": 8080}]
        urls = resolve_metrics_urls(models)
        assert urls == ["http://[2001:db8::1]:8080/metrics"]

    def test_ipv6_loopback(self):
        models = [{"host_ip": "::1", "host_port": 9090}]
        urls = resolve_metrics_urls(models)
        assert urls == ["http://[::1]:9090/metrics"]

    def test_ipv6_zero_compressed(self):
        models = [{"host_ip": "::", "host_port": 8080}]
        urls = resolve_metrics_urls(models)
        assert urls == ["http://[::]:8080/metrics"]

    # ------------------------------------------------------------------
    #  Hostname – should pass through as-is
    # ------------------------------------------------------------------
    def test_hostname_no_brackets(self):
        models = [{"host_ip": "my-inference-server", "host_port": 8080}]
        urls = resolve_metrics_urls(models)
        assert urls == ["http://my-inference-server:8080/metrics"]

    def test_localhost_no_brackets(self):
        models = [{"host_ip": "localhost", "host_port": 8080}]
        urls = resolve_metrics_urls(models)
        assert urls == ["http://localhost:8080/metrics"]

    # ------------------------------------------------------------------
    #  Deduplication – same URL across multiple models → only one entry
    # ------------------------------------------------------------------
    def test_duplicate_urls_dedup(self):
        models = [
            {"host_ip": "10.0.0.1", "host_port": 8080},
            {"host_ip": "10.0.0.1", "host_port": 8080},
            {"host_ip": "10.0.0.2", "host_port": 9090},
        ]
        urls = resolve_metrics_urls(models)
        assert urls == [
            "http://10.0.0.1:8080/metrics",
            "http://10.0.0.2:9090/metrics",
        ]

    # ------------------------------------------------------------------
    #  Defaults
    # ------------------------------------------------------------------
    def test_default_host_and_port(self):
        """No host_ip/host_port → falls back to localhost:8080."""
        models = [{}]
        urls = resolve_metrics_urls(models)
        assert urls == ["http://localhost:8080/metrics"]

    def test_empty_models(self):
        """Empty model list → empty URL list."""
        assert resolve_metrics_urls([]) == []
