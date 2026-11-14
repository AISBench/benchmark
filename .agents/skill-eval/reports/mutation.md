# 测评集自检（mutation test）

被测 SKILL：`ascend-llm-deploy-eval`；执行解释器：`python.exe`。

做法：把 SKILL 复制一份，逐条注入「真实可能发生的退化」，再跑对应用例。
用例必须由 PASS 变成 FAIL，否则说明断言太松，测评集本身不可信。

**检出 8/8。**

| 编号 | 注入的退化 | 期望失败用例 | 实际 | 结论 |
|---|---|---|---|---|
| M1 | 稳态正式请求数从 10× 退回 4× | C06 | C06=FAIL | 检出 |
| M2 | 去掉 ready 任务的 unresolved 拦截 | C02 | C02=FAIL | 检出 |
| M3 | 关闭开关时仍然注入 DEV_MODE（静默改参） | C08 | C08=FAIL | 检出 |
| M4 | 删除参考脚本环境与绑定的一致性校验（生成期 + 编译期两处） | C10 | C10=FAIL | 检出 |
| M5 | 代理入口标记消失（依赖检查会读不到命令） | C16 | C16=FAIL | 检出 |
| M6 | 模型目录挂载丢掉只读标记 | C15 | C15=FAIL | 检出 |
| M7 | AISBench retry 语义被改成 3（工具内部重试） | C17 | C17=FAIL | 检出 |
| M8 | 挂载模板参考文档被删除，SKILL.md 链接失效 | S02 | S02=FAIL | 检出 |

复现：`python tools/mutation_check.py --python <py> --deps <deps> --md reports/mutation.md`
