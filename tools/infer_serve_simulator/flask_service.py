
import os
import time
from flask import Flask, jsonify, request, Response
from api import OpenAIChatAPI, OpenAIAPI, TGIAPI, TritonAPI, MindIEOriginAPI, MindIEOriginTokenAPI
import threading

app = Flask(__name__)

active_connections = 0

lock = threading.Lock()

@app.before_request
def before_request(): # 添加连接数
    global active_connections
    with lock:
        active_connections += 1


@app.after_request
def after_request(response): # 释放连接数
    global active_connections
    with lock:
        active_connections -= 1
    return response


@app.route('/active_connections', methods=['GET'])
def get_active_connections(): # 获取当前服务进程下的连接数
    global active_connections
    return jsonify({"active_connections": f"{os.getpid()}: {active_connections}"})


@app.route('/v1/chat/completions', methods=['POST'])
def openai_chat_api_general(): # 处理服务的主函数
    req_data = request.get_json()
    api = OpenAIChatAPI() # 实例化OpenAIChatAPI 类
    if not req_data.get("stream", False): # 非流式场景的处理
        return api.generate_text(req_data)
    return api.generate_stream(req_data)


@app.route('/v1/completions', methods=['POST'])
def openai_api_general(): # 处理服务的主函数
    req_data = request.get_json()
    api = OpenAIAPI() # 实例化OpenAIAPI 类
    if not req_data.get("stream", False): # 非流式场景的处理
        return api.generate_text(req_data)
    return api.generate_stream(req_data)


@app.route('/generate', methods=['POST'])
def tgi_text_api(): # 处理服务的主函数
    req_data = request.get_json()
    api = TGIAPI() # 实例化OpenAIAPI 类
    return api.generate_text(req_data)


@app.route('/generate_stream', methods=['POST'])
def tgi_stream_api(): # 处理服务的主函数
    req_data = request.get_json()
    api = TGIAPI() # 实例化OpenAIAPI 类
    return api.generate_stream(req_data)


MODEL_NAME = "qwen"
@app.route(f'/v2/models/{MODEL_NAME}/generate', methods=['POST'])
def triton_text_api(): # 处理服务的主函数
    req_data = request.get_json()
    api = TritonAPI() # 实例化OpenAIAPI 类
    return api.generate_text(req_data)


@app.route(f'/v2/models/{MODEL_NAME}/generate_stream', methods=['POST'])
def triton_stream_api(): # 处理服务的主函数
    req_data = request.get_json()
    api = TritonAPI() # 实例化OpenAIAPI 类
    return api.generate_stream(req_data)


@app.route('/infer', methods=['POST'])
def mindie_origin_api(): # 处理服务的主函数
    req_data = request.get_json()
    api = MindIEOriginAPI() # 实例化OpenAIAPI 类
    if not req_data.get("stream", False): # 非流式场景的处理
        return api.generate_text(req_data)
    return api.generate_stream(req_data)


@app.route('/infer_token', methods=['POST'])
def mindie_origin_token_api(): # 处理服务的主函数
    req_data = request.get_json()
    api = MindIEOriginTokenAPI() # 实例化OpenAIAPI 类
    if not req_data.get("stream", False): # 非流式场景的处理
        return api.generate_text(req_data)
    return api.generate_stream(req_data)


if __name__ == '__main__': # 单进程用python3 直接执行
    app.run(host='xx.xx.xx.xx', port=5101, threaded=True) # host为IP，port为端口号