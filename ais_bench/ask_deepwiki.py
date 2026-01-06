import requests
import json

# 配置信息
MCP_URL = "https://mcp.deepwiki.com/sse"
QUESTION = "Devin 支持直接调用哪些 MCP 工具？"  # 你的查询问题

# 构造请求头
headers = {
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive"
}

# 构造请求体
payload = {
    "jsonrpc": "2.0",
    "id": f"req-{hash(QUESTION)}",  # 简单生成唯一 ID
    "method": "tool.call",
    "params": {
        "tool": "ask_question",
        "parameters": {"question": QUESTION,"repoName": "AISBench/benchmark"},
        "metadata": {"client": "python-custom-client", "timeout": 30}
    }
}

# 发送 POST 请求（stream=True 开启流模式）
response = requests.post(
    MCP_URL,
    headers=headers,
    json=payload,
    stream=True,
    timeout=30
)

# 解析 SSE 响应流
if response.status_code == 200:
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("data: "):
            # 提取并解析 JSON 数据
            data = json.loads(line[6:])  # 去掉前缀 "data: "
            print("收到响应：", data)

            # 常见响应类型（根据 MCP 协议）
            if data.get("method") == "tool.response":
                # 工具调用成功结果
                result = data["params"]["result"]
                print("工具返回结果：", result)
                break  # 结束流监听（若为一次性结果）
            elif data.get("method") == "tool.progress":
                # 工具调用进度更新（如加载中）
                progress = data["params"]["progress"]
                print(f"进度：{progress}%")
            elif data.get("error"):
                # 错误信息
                print("调用失败：", data["error"]["message"])
                break
else: