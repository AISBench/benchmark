import copy
import importlib.util
import json
import math
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


PROXY_PATH = (
    Path(__file__).resolve().parents[3] / "response_anomaly_fault_proxy.py"
)
SPEC = importlib.util.spec_from_file_location(
    "response_anomaly_fault_proxy", PROXY_PATH
)
fault_proxy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fault_proxy)


@pytest.mark.parametrize(
    ("fault_type", "expected_length", "expected_logprob"),
    [
        ("rare", 1, -7.0),
        ("garbled", 64, -7.0),
        ("repetition", 1024, -0.05),
    ],
)
def test_build_fault_payload(fault_type, expected_length, expected_logprob):
    tokens, topk_logprobs = fault_proxy.build_fault_payload(fault_type)

    assert len(tokens) == expected_length
    assert len(topk_logprobs) == expected_length
    assert all(item[token] == expected_logprob for token, item in zip(tokens, topk_logprobs))


def test_build_nan_fault_payload():
    tokens, topk_logprobs = fault_proxy.build_fault_payload("nan")

    assert tokens == [100]
    assert math.isnan(topk_logprobs[0][100])


def test_inject_fault_preserves_text_and_adds_vllm_payload():
    response = {
        "choices": [{"message": {"content": "real model output"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }

    injected = fault_proxy.inject_fault(copy.deepcopy(response), "rare")

    choice = injected["choices"][0]
    assert choice["message"]["content"] == "real model output"
    assert choice["token_ids"] == [100]
    assert choice["logprobs"]["content"][0]["token"] == "token_id:100"
    assert injected["response_anomaly_fault"] == "rare"
    assert injected["usage"]["completion_tokens"] == 1
    assert injected["usage"]["total_tokens"] == 6


def test_none_fault_leaves_response_unchanged():
    response = {"choices": [{"message": {"content": "unchanged"}}]}

    assert fault_proxy.inject_fault(response, "none") is response


def test_inject_fault_rejects_response_without_choice():
    with pytest.raises(ValueError, match=r"choices\[0\]"):
        fault_proxy.inject_fault({}, "rare")


def test_streaming_request_detection():
    assert fault_proxy.FaultProxyHandler._is_streaming_request(
        b'{"stream": true}'
    )
    assert not fault_proxy.FaultProxyHandler._is_streaming_request(
        b'{"stream": false}'
    )


def test_proxy_forwards_request_and_injects_fault():
    class UpstreamHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(content_length)
            body = json.dumps(
                {
                    "choices": [{"message": {"content": "real output"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever)
    upstream_thread.start()

    fault_proxy.FaultProxyHandler.upstream = (
        f"http://127.0.0.1:{upstream.server_port}"
    )
    fault_proxy.FaultProxyHandler.fault_type = "garbled"
    proxy = ThreadingHTTPServer(
        ("127.0.0.1", 0), fault_proxy.FaultProxyHandler
    )
    proxy_thread = threading.Thread(target=proxy.serve_forever)
    proxy_thread.start()

    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions",
            data=b'{"stream": false}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.load(response)

        assert payload["choices"][0]["message"]["content"] == "real output"
        assert len(payload["choices"][0]["token_ids"]) == 64
        assert payload["response_anomaly_fault"] == "garbled"
    finally:
        proxy.shutdown()
        proxy.server_close()
        upstream.shutdown()
        upstream.server_close()
        proxy_thread.join()
        upstream_thread.join()
