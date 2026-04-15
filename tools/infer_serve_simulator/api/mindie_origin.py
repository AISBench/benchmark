import time
from api.base_api import BaseAPI
from flask import Flask, jsonify, request, Response

class MindIEOriginAPI(BaseAPI):
    def __init__(self):
        super().__init__()

    def generate_text(self, req_data):
        # 非流式场景的处理
        parameters = req_data.get("parameters", {})
        max_tokens = parameters.get("max_new_tokens", 1)
        output_list, _ = self.gen_output_list(max_tokens, parameters.get("ignore_eos", False))
        return self.text_request_body("".join(output_list))


    def generate_stream(self, req_data):
        # 流式场景的处理
        parameters = req_data.get("parameters", {})
        max_tokens = parameters.get("max_new_tokens", 1)
        output_list, _ = self.gen_output_list(max_tokens, parameters.get("ignore_eos", False))
        return Response(self.stream_request_body(output_list), mimetype='text/event-stream')


    def text_request_body(self, content):

        time.sleep(self.e2el)
        return jsonify({
            "generated_text": content
        })

    def stream_request_body(self, output_list):
        for i, tokens in enumerate(output_list):
            if i == 0:
                time.sleep(self.ttft)
            else:
                time.sleep(self.tpot * len(tokens))
            generated_text = "" if i != len(output_list) - 1 else "".join(output_list)
            response = {
                "generated_text": generated_text,
                "token":{
                    "text": tokens,
                }
            }
            # 以SSE 格式发送响应
            s = f"data: {response}\n\n"
            s = s.replace("\'", "\"")
            yield s


class MindIEOriginTokenAPI(MindIEOriginAPI):
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
        return Response(self.stream_request_body(output_list, output_length), mimetype='text/event-stream')

    def text_request_body(self, content, output_length=1):
        time.sleep(self.e2el)
        return jsonify({
            "generated_text": content,
            "details":{
                "generated_tokens": output_length
            }
        })

    def stream_request_body(self, output_list, output_length=1):
        for i, tokens in enumerate(output_list):
            if i == 0:
                time.sleep(self.ttft)
            else:
                time.sleep(self.tpot * len(tokens))
            response = {
                "token":{
                    "text": tokens,
                }
            }
            if i == len(output_list) - 1:
                response["details"] = {}
                response["details"]["generated_tokens"] = output_length

            # 以SSE 格式发送响应
            s = f"data: {response}\n\n"
            s = s.replace("\'", "\"")
            yield s