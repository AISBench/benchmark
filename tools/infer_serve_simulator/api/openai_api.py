import time
from api.base_api import BaseAPI
from flask import Flask, jsonify, request, Response

class OpenAIAPI(BaseAPI):
    def __init__(self):
        super().__init__()

    def generate_text(self, req_data):
        # 非流式场景的处理
        max_tokens = req_data.get("max_tokens", 1)
        output_list, output_length = self.gen_output_list(max_tokens, req_data.get("ignore_eos", False))
        return self.text_request_body("".join(output_list), output_length)

    def generate_stream(self, req_data):
        # 流式场景的处理
        max_tokens = req_data.get("max_tokens", 1)
        output_list, output_length = self.gen_output_list(max_tokens, req_data.get("ignore_eos", False))
        return Response(self.stream_request_body(output_list, output_length), mimetype='text/event-stream')

    def text_request_body(self, content, output_length=1):
        time.sleep(self.e2el)
        return jsonify({
            "choices": [
                {
                    "text": content
                }
            ]
        })

    def stream_request_body(self, output_list, output_length=1):
        for i, tokens in enumerate(output_list):
            if i == 0:
                time.sleep(self.ttft)
            else:
                time.sleep(self.tpot * len(tokens))

            response = {
                "choices": [
                    {
                        "text": tokens
                    }
                ]
            }
            if i == len(output_list) - 1:
                response["usage"] = {}
                response["usage"]["completion_tokens"] = output_length
            # 以SSE 格式发送响应
            s = f"data: {response}\n\n"
            s = s.replace("\'", "\"")
            yield s
        yield "data: [DONE]"