# aisbench ITL 计时与 vLLM Completions 对齐设计

## 目标

使 aisbench 的 VLLM 流式计时语义与 `vllm bench serve --backend vllm` 保持一致：仅将 `choices` 非空的响应视为输出计时事件；usage-only 响应和 `[DONE]` 消息均不影响 ITL、TPOT 或 E2EL。

## 范围

- 对 `VLLMCustomAPI` 和 `VLLMCustomAPIChat` 应用 token 事件过滤。
- 保持 TGI、Triton、MindIE 和 VITA 适配器现有的流式计时行为不变。
- 继续解析 usage-only 响应，确保 prompt token 和 completion token 数量仍能正确获取。
- 删除 `base_api.py` 中当前未提交的 `[ITL-DIAG]` 临时诊断代码。
- 修改生产代码前先添加回归测试。
- 本次变更不修改 ITL 聚合方式、百分位计算、CSV 列或报告中 `N` 字段的含义。

## 设计

### 协议感知的计时判断方法

在 `BaseAPIModel` 中增加一个可覆盖的方法：

```python
def should_record_stream_time_point(self, data: dict) -> bool:
    return True
```

基类默认返回 `True`，从而保持所有非 VLLM 适配器的现有行为。两个 VLLM 适配器均覆盖该方法：

```python
def should_record_stream_time_point(self, data: dict) -> bool:
    return bool(data.get("choices"))
```

### 流式数据处理流程

对于每一条非空、非注释且不等于 `[DONE]` 的流式消息：

1. 解码并解析 JSON 数据。
2. 调用 `should_record_stream_time_point(data)`。
3. 仅当该方法返回 `True` 时记录时间点。
4. 始终调用 `parse_stream_response(data, output)`，确保 usage-only 响应仍可写入 token 数量。

请求开始时间点的记录方式保持不变。如果响应中包含 1024 个携带 `choices` 的输出 chunk，最终计时数组将包含一个请求开始时间点和 1024 个输出事件，因此生成 1023 个 ITL。

## 兼容性

- 该设计与 vLLM v0.26 Completions 客户端一致：仅在 `choices` 非空分支内记录时间点。
- Chat 适配器采用相同过滤规则，使当前配置的 aisbench Chat 接口也可以按照同一套计时语义进行比较。
- 其他适配器继承基类默认判断，继续为每个已解析的 JSON chunk 记录时间点。
- JSON 解析失败时继续沿用现有错误处理路径，并且不记录时间点。

## 测试设计

新增针对性的异步测试，模拟包含以下内容的响应流：两个携带 `choices` 的 chunk、一个 usage-only chunk，以及 `[DONE]`。

必须验证：

- VLLM 适配器共记录三个时间点：一个请求开始时间点和两个 choice 事件时间点。
- usage-only chunk 不增加时间点。
- prompt token 和 completion token 数量可从 usage 响应中正确获取。
- 最终只产生一个 ITL。
- 选取一个有代表性的非 VLLM 适配器，验证其仍为每个已解析的数据 chunk 记录时间点。

在修改生产代码之前，测试必须在当前实现上因 usage-only chunk 被错误计时而失败。完成最小实现后，测试必须通过。

## 错误处理

本次变更不引入新的错误类型。现有 HTTP 错误、JSON 解析错误、重试和空响应处理逻辑均保持不变。

## 验收标准

- VLLM usage-only 响应不再产生亚毫秒级的尾部 ITL。
- VLLM usage token 统计保持正确。
- 非 VLLM 流式计时行为不发生变化。
- 针对性测试及相关现有 API 模型测试全部通过。
- 删除临时诊断代码后，`base_api.py` 能够通过 Python 语法解析。
