from flask import Response, jsonify
from api.base_api import load_yaml_config


class MetricsAPI:
    """处理 /metrics 端点请求，用于 spec-decode 异常场景测试。

    通过 api_config.yaml 的 metrics.mode 配置项控制行为：
      - error_http:      返回指定 HTTP 错误码
      - no_spec:         返回 200 但不含 spec_decode 计数器
      - static_counters: 返回 200 + 固定不变的 spec_decode 计数器
    """

    # Prometheus 文本格式中其他常见的 vllm 计数器（用于 no_spec 模式填充）
    _OTHER_VLLM_METRICS = [
        '# HELP vllm:num_requests_total Number of requests processed.',
        '# TYPE vllm:num_requests_total counter',
        'vllm:num_requests_total 42.0',
        '# HELP vllm:cache_config_info Cache configuration info.',
        '# TYPE vllm:cache_config_info gauge',
        'vllm:cache_config_info{block_size="16"} 1.0',
    ]

    def __init__(self):
        self.config = load_yaml_config()
        self.metrics_conf = self.config.get("metrics", {})

    def handle_metrics(self):
        """根据配置返回 /metrics 响应。"""
        mode = self.metrics_conf.get("mode", "static_counters")

        if mode == "error_http":
            return self._error_http_response()
        elif mode == "no_spec":
            return self._no_spec_response()
        elif mode == "static_counters":
            return self._static_counters_response()
        else:
            # 未知 mode，当作 no_spec 处理
            return self._no_spec_response()

    def _error_http_response(self):
        """返回指定 HTTP 错误码，模拟 /metrics 不可达或服务端错误。"""
        error_code = int(self.metrics_conf.get("error_code", 503))
        return Response(f"metrics endpoint error: {error_code}", status=error_code)

    def _no_spec_response(self):
        """返回 200 + 其他 vllm 计数器，但不含 spec_decode 计数器。

        触发 fetcher → snapshot.py 的 parse_spec_decode_metrics 返回 None，
        最终输出 "No spec decode metrics found on server"。
        """
        text = "\n".join(self._OTHER_VLLM_METRICS) + "\n"
        return Response(text, status=200, mimetype="text/plain")

    def _static_counters_response(self):
        """返回 200 + 固定 spec_decode 计数器，值不随请求变化。

        before/after 快照拿到相同值 → delta_draft_tokens=0
        → calculator.py 返回 None
        → 输出 "No spec decode activity detected during benchmark window"
        """
        counters = self.metrics_conf.get("counters", {})
        lines = [
            '# HELP vllm:spec_decode_num_drafts_total Number of draft-and-verify cycles.',
            '# TYPE vllm:spec_decode_num_drafts_total counter',
            f'vllm:spec_decode_num_drafts_total {counters.get("num_drafts", 100)}.0',
            '# HELP vllm:spec_decode_num_draft_tokens_total Number of draft tokens proposed.',
            '# TYPE vllm:spec_decode_num_draft_tokens_total counter',
            f'vllm:spec_decode_num_draft_tokens_total {counters.get("num_draft_tokens", 500)}.0',
            '# HELP vllm:spec_decode_num_accepted_tokens_total Number of accepted tokens.',
            '# TYPE vllm:spec_decode_num_accepted_tokens_total counter',
            f'vllm:spec_decode_num_accepted_tokens_total {counters.get("num_accepted_tokens", 450)}.0',
        ]

        accepted_per_pos = counters.get("accepted_per_pos", {})
        if accepted_per_pos:
            lines.append('# HELP vllm:spec_decode_num_accepted_tokens_per_pos_total Accepted tokens per position.')
            lines.append('# TYPE vllm:spec_decode_num_accepted_tokens_per_pos_total counter')
            for pos in sorted(accepted_per_pos.keys()):
                lines.append(
                    f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="{pos}"}} {accepted_per_pos[pos]}.0'
                )

        text = "\n".join(lines) + "\n"
        return Response(text, status=200, mimetype="text/plain")
