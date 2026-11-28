# 取证导航（不是兼容认证）

以下路径/行为核对于 AISBench `d99493ac1914ff41415196eece58f8f296f9e2d6`，2026-11-11。行号只作基线定位；核查时读取当前文件和调用者，不机械套用结论。

## 主适配与真实协议

| 场景 | 从哪里读 | 基线观察/核查重点 |
| --- | --- | --- |
| 配置与支持声明 | [模型配置说明](<../../../../docs/source_zh_cn/base_tutorials/all_params/models.md#L12-L91>)、[流式 Chat 配置](<../../../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py>)、[模型导出](<../../../../ais_bench/benchmark/models/__init__.py>) | 配置名不能代替实际类。文档注明不支持 `n`/`best_of` 多采样；不因 body 可携带就宣称评测支持。文档中概括性描述也可能落后实现 |
| Chat | [vllm_custom_api_chat.py](<../../../../ais_bench/benchmark/models/api_models/vllm_custom_api_chat.py#L140-L223>) | `_get_url` 拼 `v1/chat/completions`；str → user 消息，列表 role 映射后复制其他键；覆写 `max_tokens`/`model`，真值 `tools` 入 body；先放流式 usage 选项再合并 kwargs。解析遍历 choices 拼接文本/推理内容；tool calls 另看输出类，不能推断所有流式工具片段都被保留 |
| Completions | [vllm_custom_api.py](<../../../../ais_bench/benchmark/models/api_models/vllm_custom_api.py#L114-L185>) | `v1/completions`、`prompt`；覆写长度/模型，合并 kwargs **后**设置流式 usage 选项，与 Chat 顺序不同；主要读取第一个 choice 的 text。logprobs 的结构转换与 Chat 不同 |
| 共用 URL/自动发现 | [base_api.py:131–180](<../../../../ais_bench/benchmark/models/api_models/base_api.py#L131-L180>) | 自定义 `url` 是 base，规范化后由子类追加 endpoint；`requests.get` 自动发现 `v1/models` 的首个 id。空模型构造可能联网；离线测试显式给模型名或 mock 发现，不要盲目实例化 |
| 共用发送/错误 | [base_api.py:429–537](<../../../../ais_bench/benchmark/models/api_models/base_api.py#L429-L537>) | `aiohttp` POST JSON；仅 status 200 走成功解析，非 200 存 `reason` 而非解析错误 body，并正常返回；`generate` 的异常分支与 HTTP 失败重试不同。检查 `range(self.retry)`、取消、部分输出和重试清理，不以 docstring 代替控制流 |
| 共用 SSE/打点 | [分帧](<../../../../ais_bench/benchmark/models/api_models/base_api.py#L182-L216>)、[stream_infer](<../../../../ais_bench/benchmark/models/api_models/base_api.py#L482-L514>) | 空行分帧、跳过以 `:`/`id:` 开头的整块、去 `data:`、遇 `[DONE]` 停止；不是通用 SSE 合规证明。普通 EOF 也可到 `success=True`；检查缺终止、event/multiline 与断流是否符合目标链路。POST 前及解析 JSON 前打点，role/usage 事件也可能计入 |
| usage/输出/工具 | [Chat usage](<../../../../ais_bench/benchmark/models/api_models/vllm_custom_api_chat.py#L189-L200>)、[Output](<../../../../ais_bench/benchmark/models/output.py#L11-L31>)、[FunctionCallOutput](<../../../../ais_bench/benchmark/models/output.py#L170-L186>) | 真值 usage 才更新 prompt/completion tokens；字段缺省回到 0，`null` 值本身不一定被转换。Output 初值 0 不代表服务提供了计数。FunctionCallOutput 只保存首个 message；分别检查非流式工具调用与流式消费 |
| 模板/调用/计时消费 | [APITemplateParser](<../../../../ais_bench/benchmark/models/api_models/base_api.py#L693-L966>)、[GenInferencer](<../../../../ais_bench/benchmark/openicl/icl_inferencer/icl_gen_inferencer.py#L65-L94>)、[Output 打点](<../../../../ais_bench/benchmark/models/output.py#L125-L167>)、[性能计算器](<../../../../ais_bench/benchmark/calculators/base_perf_metric_calculator.py>) | 模板的 `_role2api_role` 仅构造 role/prompt；直接 get_request_body 复制其他键不代表模板入口保留。记录实际输出类；只有 perf_mode 打点。计时/评测消费要独立验证 |
| 异常检测扩展 | [BaseAPIModel 初始化](<../../../../ais_bench/benchmark/models/api_models/base_api.py#L108-L113>)、[Chat 注入](<../../../../ais_bench/benchmark/models/api_models/vllm_custom_api_chat.py#L159-L164>)、[异常检测文档](<../../../../docs/source_zh_cn/advanced_tutorials/response_anomaly_detection.md#L21-L25>) | 内部 `response_anomaly_enabled` 被移除，启用后注入 token-id 扩展；不能将扩展当官方 OpenAI 必选字段，也不默认启用检测来核查协议 |
| MindIE 原生 | [mindie_stream_api.py](<../../../../ais_bench/benchmark/models/api_models/mindie_stream_api.py#L72-L125>) | `infer`，`inputs`/`parameters.max_new_tokens`、body 固定 `stream=True`，读取 generated_text；与走 Chat 的 MindIE 服务分开 |
| TGI 原生 | [tgi_api.py](<../../../../ais_bench/benchmark/models/api_models/tgi_api.py#L73-L132>) | `generate`/`generate_stream`，`inputs`、`parameters.max_new_tokens`，以 generated_text 更新输出；不要套用 Chat delta/usage 契约 |
| Triton | [triton_api.py](<../../../../ais_bench/benchmark/models/api_models/triton_api.py#L73-L98>) | `v2/models/{model_name}/generate[_stream]`，`text_input`、parameters、`text_output`；核对模型 backend 与发布版本，不泛指所有 Triton API |
| VITA 自定义 | [vita_generate_api.py](<../../../../ais_bench/benchmark/models/api_models/vita_generate_api.py#L72-L144>) | `generate`、扁平 prompt 内容数组，file URL 处理、kwargs 覆盖和 text 数组解析均为独立契约，不因字段相似归入 OpenAI |

## 独立路径（仅在问题触及时追踪）

- **流量回放不是 BaseAPIModel**：[转换/headers/SSE](<../../../../ais_bench/benchmark/tasks/llm_io_replay/llm_io_replay.py#L120-L248>)、[ReplayClient.send](<../../../../ais_bench/benchmark/tasks/llm_io_replay/llm_io_replay.py#L781-L947>)。转换有白名单、补 trace/timestamp、null 内容变换；不等同原始请求透明重放。SSE 多行合并实现与主适配不同；HTTP 错误截取 body，网络异常另存；首 JSON 事件时间不一定是首文本 token。缺 usage 会估算并记录 `token_source`；不能把其数值与主链路 API usage 混为一谈。
- **prefix_cache 插件**：[Chat 精确 endpoint](<../../../../plugins/prefix_cache/ais_bench_prefix_cache/models/vllm_prefix_cache_chat_api.py>)、[Completions 发送覆写](<../../../../plugins/prefix_cache/ais_bench_prefix_cache/models/vllm_prefix_cache_api.py#L61-L197>)。后者剥离内部 DP 标记并构造自定义 header、覆写 text/stream 发送，不能只检查基类即覆盖插件行为。日志可能含 prompt，报告需脱敏。
- **SDK 示例的边界**：[旧插件示例](<../../../../plugin_examples/ais_bench_plugin_example_pkg/models/example_model.py#L9-L18>) 引用旧 `models.base_api`、`models.performance_api` 和 `clients`，这些模块在本基线对应位置不存在；不能据此臆造当前主适配的 SDK 调用链。`openai` 在 [runtime 依赖](<../../../../requirements/runtime.txt#L20>) 和 [api 依赖](<../../../../requirements/api.txt>) 中未固定版本。若问题来自外部插件/Agent 集成，先定位其实际客户端与版本，不自动扩大本指南的覆盖面。

## 现有测试与 docs 入口

- [test_base_api.py](<../../../../tests/UT/models/api_models/test_base_api.py>)：URL/发现、分帧、发送/重试、模板；部分 `async def test_*` 在 `unittest.TestCase` 下无 await 包装，不能把测试名被收集或绿色结果当执行证据。
- [test_vllm_custom_api_chat.py](<../../../../tests/UT/models/api_models/test_vllm_custom_api_chat.py>)、[test_vllm_custom_api.py](<../../../../tests/UT/models/api_models/test_vllm_custom_api.py>)：body、解析、usage/logprobs/LoRA；选择实际调用 `asyncio.run` 的 wrapper/同步用例，核对断言范围。
- [test_response_anomaly_payload.py](<../../../../tests/UT/models/api_models/test_response_anomaly_payload.py>)：扩展 payload，不代表官方完整协议覆盖。
- [回放 core 测试](<../../../../tests/test_llm_io_replay_core.py#L490-L614>)：fake session 的 SSE 与 usage；[插件适配测试](<../../../../plugins/prefix_cache/tests/test_aisbench_adapters.py>)：模拟 AISBench 类验证插件行为，不等于真实基类集成测试。
- 原生协议分别见 [MindIE 测试](<../../../../tests/UT/models/api_models/test_mindie_stream_api.py>)、[TGI 测试](<../../../../tests/UT/models/api_models/test_tgi_api.py>)、[Triton 测试](<../../../../tests/UT/models/api_models/test_triton_api.py>)；先检查 mock/异步调用方式再运行。
- 文档契约：[新模型接入](<../../../../docs/source_en/develop_guide/new_model.md>)、[性能指标语义](<../../../../docs/source_zh_cn/base_tutorials/results_intro/performance_metric.md>)。文档不是实际行为的替代证据，也不要直接照抄其中部署/评测命令作为本技能默认动作。

## 外部来源与版本

以下 URL 于 2026-11-11 确认；仅为可追溯入口，不是所有目标版本的统一基线。未用本次失败/截断的内容作字段判断。

| 来源类型 | 已确认入口与使用限制 |
| --- | --- |
| 官方 OpenAPI | [openai-openapi README](<https://github.com/openai/openai-openapi/blob/ce5f4b535b105acd27ca1473c8906d85207cb0cc/README.md>) 指向 [openapi.yaml 固定提交](<https://github.com/openai/openai-openapi/blob/ce5f4b535b105acd27ca1473c8906d85207cb0cc/openapi.yaml>)；已确认文件存在，约 4.3 MB，本次 gh_file 未返回可用 schema 正文。实际核查需获取目标章节/本地快照，不能据文件存在断言字段语义；不要沿用参考技能的浮动 `master` raw 地址 |
| 官方 SDK 类型 | 已读取 [openai-python v1.109.1 stream options](<https://github.com/openai/openai-python/blob/v1.109.1/src/openai/types/chat/chat_completion_stream_options_param.py>) 与 [chunk 类型](<https://github.com/openai/openai-python/blob/v1.109.1/src/openai/types/chat/chat_completion_chunk.py>)：include_usage 额外 chunk 在 `[DONE]` 前、choices 为空，其他 chunk usage 为 null，中断可能没有最终 usage。这是该 SDK tag 的类型/注释证据，不冒充已读取的官方 OpenAPI；真实项目版本未知时不能替它选此版本 |
| 官方语义网页 | [Chat create 旧入口](<https://platform.openai.com/docs/api-reference/chat/create>) 与 [Developers create 入口](<https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create>) 本次均 HTTP 403，**未取得正文**；应记录未知或改用可审计快照，而不是宣布字段不存在 |
| 后端文档 | [vLLM v0.6.6 OpenAI-compatible server](<https://docs.vllm.ai/en/v0.6.6/serving/openai_compatible_server.html>) 本次 HTTP 200；支持面、extra parameters、SDK `extra_body` 与直接 HTTP 合并 JSON 的区分可读。它只说明该后端版本，不是本次真实服务版本，也不能证明新 token-id 扩展在该版支持。其他后端按具体版本查官方资料，不猜 URL |
| 技能结构参考 | 已通过 gh_file 读取 [vLLM check-api-compat](<https://github.com/vllm-project/vllm/blob/8551e3a13f5ea07f46bb70a18c800990c9fd2a6a/.agents/skills/check-api-compat/SKILL.md>)；借鉴“固定问题、查来源、比较、分类”流程，不沿用服务端必须覆盖上游端点的责任边界 |
