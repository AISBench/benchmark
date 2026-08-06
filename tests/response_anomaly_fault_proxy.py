#!/usr/bin/env python3
"""Non-streaming reverse proxy for response-anomaly fault injection tests.

This utility forwards requests to a real OpenAI-compatible vLLM endpoint and
replaces the first successful chat choice's token/logprob payload with a
deterministic anomaly. It is test-only and intentionally rejects streaming
requests.
"""

import argparse
import json
import math
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


FAULT_TYPES = ("none", "rare", "garbled", "repetition", "nan")
_REQUEST_HEADERS_TO_DROP = {
    "accept-encoding",
    "connection",
    "content-length",
    "host",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def build_fault_payload(fault_type):
    """Return deterministic token ids and top-k maps for one anomaly type."""
    if fault_type == "rare":
        tokens = [100]
        logprobs = [-7.0]
    elif fault_type == "garbled":
        tokens = [100 + index % 5 for index in range(64)]
        logprobs = [-7.0] * len(tokens)
    elif fault_type == "repetition":
        tokens = [100 + index % 3 for index in range(1024)]
        logprobs = [-0.05] * len(tokens)
    elif fault_type == "nan":
        tokens = [100]
        logprobs = [float("nan")]
    else:
        raise ValueError(f"Unsupported fault type: {fault_type}")

    topk_logprobs = [
        {token_id: logprob}
        for token_id, logprob in zip(tokens, logprobs)
    ]
    return tokens, topk_logprobs


def inject_fault(response, fault_type):
    """Inject a vLLM OpenAI-style token/logprob payload into a response."""
    if fault_type == "none":
        return response
    if not isinstance(response, dict):
        raise ValueError("Upstream response must be a JSON object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("Upstream response does not contain choices[0]")

    tokens, topk_logprobs = build_fault_payload(fault_type)
    content = []
    for token_id, token_logprobs in zip(tokens, topk_logprobs):
        sampled_logprob = token_logprobs[token_id]
        content.append(
            {
                "token": f"token_id:{token_id}",
                "logprob": sampled_logprob,
                "bytes": None,
                "top_logprobs": [
                    {
                        "token": f"token_id:{topk_token_id}",
                        "logprob": logprob,
                        "bytes": None,
                    }
                    for topk_token_id, logprob in token_logprobs.items()
                ],
            }
        )

    choice = choices[0]
    choice["token_ids"] = tokens
    choice["logprobs"] = {"content": content}
    response["response_anomaly_fault"] = fault_type
    usage = response.get("usage")
    if isinstance(usage, dict):
        usage["completion_tokens"] = len(tokens)
        prompt_tokens = usage.get("prompt_tokens")
        if isinstance(prompt_tokens, int):
            usage["total_tokens"] = prompt_tokens + len(tokens)
    return response


class FaultProxyHandler(BaseHTTPRequestHandler):
    upstream = ""
    fault_type = "none"
    timeout = 600.0

    def do_GET(self):
        self._proxy()

    def do_POST(self):
        self._proxy()

    def _proxy(self):
        request_body = self._read_request_body()
        if self.fault_type != "none" and self._is_streaming_request(request_body):
            self._send_json(
                400,
                {
                    "error": (
                        "response_anomaly_fault_proxy only supports stream=false; "
                        "set stream=False in the AISBench model config"
                    )
                },
            )
            return

        upstream_url = f"{self.upstream.rstrip('/')}{self.path}"
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in _REQUEST_HEADERS_TO_DROP
        }
        headers["Accept-Encoding"] = "identity"
        upstream_request = urllib.request.Request(
            upstream_url,
            data=request_body if request_body else None,
            headers=headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(
                upstream_request, timeout=self.timeout
            ) as upstream_response:
                status = upstream_response.status
                response_body = upstream_response.read()
                content_type = upstream_response.headers.get(
                    "Content-Type", "application/octet-stream"
                )
        except urllib.error.HTTPError as exc:
            status = exc.code
            response_body = exc.read()
            content_type = exc.headers.get("Content-Type", "application/json")
        except (OSError, urllib.error.URLError) as exc:
            self._send_json(502, {"error": f"Upstream request failed: {exc}"})
            return

        if self.fault_type != "none" and 200 <= status < 300:
            try:
                response = json.loads(response_body)
                response = inject_fault(response, self.fault_type)
                response_body = json.dumps(
                    response,
                    ensure_ascii=False,
                    allow_nan=True,
                ).encode("utf-8")
                content_type = "application/json"
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                self._send_json(502, {"error": f"Cannot inject fault: {exc}"})
                return

        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def _read_request_body(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        return self.rfile.read(content_length) if content_length > 0 else b""

    @staticmethod
    def _is_streaming_request(request_body):
        if not request_body:
            return False
        try:
            request = json.loads(request_body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        return isinstance(request, dict) and request.get("stream") is True

    def _send_json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Forward non-streaming AISBench requests to vLLM and inject a "
            "deterministic response anomaly payload."
        )
    )
    parser.add_argument(
        "--upstream",
        required=True,
        help="vLLM server origin, for example http://127.0.0.1:8000",
    )
    parser.add_argument("--fault", choices=FAULT_TYPES, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    upstream = urlsplit(args.upstream)
    if upstream.scheme not in ("http", "https") or not upstream.netloc:
        parser.error("--upstream must be an HTTP(S) server origin")
    if upstream.path not in ("", "/") or upstream.query or upstream.fragment:
        parser.error("--upstream must not contain a path, query, or fragment")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


def main():
    args = parse_args()
    FaultProxyHandler.upstream = args.upstream
    FaultProxyHandler.fault_type = args.fault
    FaultProxyHandler.timeout = args.timeout
    server = ThreadingHTTPServer((args.host, args.port), FaultProxyHandler)
    print(
        f"Response anomaly fault proxy listening on http://{args.host}:{args.port} "
        f"-> {args.upstream} (fault={args.fault}, stream=false only)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
