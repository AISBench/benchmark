import time
from api.base_api import BaseAPI
from flask import Flask, jsonify, request, Response

class TritonAPI(BaseAPI):
    def __init__(self):
        super().__init__()

    def generate_text(self, req_data):
        # 非流式场景的处理
        parameters = req_data.get("parameters", {})
        max_tokens = parameters.get("max_new_tokens", 1)
        output_list, output_length = self.gen_output_list(max_tokens, parameters.get("ignore_eos", False))
        return self.text_request_body("".join(output_list), output_length)

    def generate_stream(self, req_data):
        # 流式场景的处理
        parameters = req_data.get("parameters", {})
        max_tokens = parameters.get("max_new_tokens", 1)
        output_list, output_length = self.gen_output_list(max_tokens, parameters.get("ignore_eos", False))
        return Response(self.stream_request_body(output_list), mimetype='text/event-stream')

    def text_request_body(self, content, output_length=1):
        time.sleep(self.e2el)
        return jsonify({
            "text_output": content,
            "details":{
                "generated_tokens": output_length,
                "perf_stat": [["A ", float(self.e2el) / output_length]] * output_length
            }
        })

    def stream_request_body(self, output_list):
        total_out_len = 0
        for i, tokens in enumerate(output_list):
            total_out_len += len(tokens)
            if i == 0:
                time.sleep(self.ttft)
            else:
                time.sleep(self.tpot * len(tokens))
            response = {
                "text_output": tokens,
                "details":{
                    "generated_tokens": total_out_len,
                    "batch_size": 2,
                    "queue_wait_time": 2
                }
            }
            # 以SSE 格式发送响应
            s = f"data: {response}\n\n"
            s = s.replace("\'", "\"")
            yield s