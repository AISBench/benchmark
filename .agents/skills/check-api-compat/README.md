# check-api-compat

面向 AISBench **已有客户端协议适配**的证据化核查。入口：[SKILL.md](<SKILL.md>)；按需导航：[实现、测试、文档与外部来源](<references/evidence-map.md>)。本技能只有文档，无额外脚本或运行依赖；运行仓库 UT 仍需其现有环境。

## 何时使用

- 修改 Chat/Completions 或现有原生适配的字段/默认值、消息、透传规则、SSE、usage、错误处理时。
- 回答“字段是否被发出/覆盖/忽略”“为什么能调用却缺少计数或工具结果”“SDK 示例与 HTTP 请求为何不同”。
- 核查插件/回放路径时先定位实际发送者；不从一个类推广到全部 AISBench。

不用于证明服务端全面兼容，不默认补齐所有 OpenAI 端点/Anthropic，不诊断 CI，不部署或跑大规模评测；协议、计时和评测正确性分别取证。

## 最小输入与用法

给 Agent 加载本目录的完整技能，再提供：仓库 commit、相关 diff/问题、实际配置/类、协议和最终路径、目标后端/SDK 版本（未知可写未知）、字段/事件，以及脱敏输入/响应。不要提供密钥。

示例请求：

> 使用 check-api-compat：基于当前 checkout，核查 VLLMCustomAPIChat 的 stream_options 合并顺序和 usage-only chunk 消费。只读实现并使用现有 mock；固定对照来源，区分客户端行为、后端扩展、计时影响与未知，不联网真实端点。

另一个例子：

> 只检查 llm_io_replay 的 null 消息转换和 token_source，解释它与 BaseAPIModel 路径的不同；不要启动回放任务或将估算值视为官方 usage。

默认可查外部规范资料，但不代表获得真实推理端点、资源或凭据使用授权；受离线限制时应保留来源未知，不靠记忆补结论。

## 离线验证入口

从仓库根目录运行，先读所选测试确认 mock；以下是**按问题选用**的已有入口，不是本技能安装/验收门禁，也不是要求全部执行：

```bash
# Chat：选同步 wrapper，避免 unittest 下未 await 的 async test 假阳性
python -m pytest tests/UT/models/api_models/test_vllm_custom_api_chat.py -k "wrapper or response_anomaly_requests" -q

# 基类 URL/字节分帧：同步方法内部 asyncio.run；不运行真实端点
python -m pytest tests/UT/models/api_models/test_base_api.py -k "iter_lines or init_with_url" -q

# 回放：纯转换及 fake session；不调用任务 run/CLI
python -m pytest tests/test_llm_io_replay_core.py -k "request_conversion or endpoint_and_headers or replay_client" -q
```

依赖参考 [运行环境](<../../../requirements/runtime.txt>) 与 [测试环境](<../../../requirements/test.txt>)。先使用已有环境，不自动安装/升级包、拉模型或启用异常检测。收集失败/缺依赖照实报告，不以 mock 替换所有业务模块后宣称集成通过。

如果现有用例未覆盖问题，提出/编写最小离线回归用例须遵守用户允许的修改范围；文档核查本身无需改业务代码。新测试应显式 await 实际方法，mock HTTP session/发现请求，断言最终 JSON、输出、错误、usage 和打点，不使用真实地址/凭据。`async def test_*` 出现在 `unittest.TestCase` 下可能被收集却没执行；检查 wrapper、skip 与 RuntimeWarning，不只看通过总数。

维护此技能的纯文档检查：`git diff --check`，核对 frontmatter 的 name/description、Markdown 围栏、相对链接/行号、公共索引只新增一行，以及与当前实现的导航一致性。不新增覆盖率门槛或 CI workflow。

## 建议报告格式

```text
范围：commit / 实际类与配置 / 协议 / POST 最终路径 / stream / 输出类
责任与支持面：本次客户端承诺消费什么；哪些能力不适用
来源：官方规范 / SDK 类型 / 后端文档，固定版本或日期+快照；取证失败单列
字段或事件：输入 → 合并/移除 → wire JSON → 解析 → 保存/评测消费
预期：引文、URL、版本/章节；不能确认则未知
实际：file:line、脱敏最小样例、静态/离线/在线证据级别
分类：一致 / 扩展 / 支持子集 / 客户端缺陷 / 后端或文档差异 / 未知
影响：协议解析、计时语义、评测消费分别说明
验证：命令、通过/失败/skip/警告、未测分支；最小修复与回归方向
限制：版本缺失、来源不可读、环境缺依赖、未获在线授权等
```

不要把“未访问真实端点”改写成“服务已兼容”，也不要仅因支持子集而要求实现整个上游 API。
