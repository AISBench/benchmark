# Ascend LLM Deploy Eval

面向昇腾模型部署与性能测试的 Agent skill。将自然语言需求整理为 `job_spec`，经配置审阅和现场预检后，自动执行部署、性能测试与资源释放。

## 快速开始

1. 将 [ascend-llm-deploy-eval](SKILL.md) 所在的完整目录安装到 Agent 的 skill 目录，并确认已加载。
2. 在 skill 目录使用 Python 3.10+ 安装 `requirements.txt`；准备免密 SSH、Docker、模型权重和部署参考，测试镜像内预装 AISBench。
3. 替换以下占位地址和路径后交给 Agent：

   ```text
   $ascend-llm-deploy-eval
   服务器：root@node-a.example、root@node-b.example，SSH免密。
   远程工作目录：/mnt/work/benchmark
   镜像：已核验的镜像ID；模型：DSV4 Flash。
   权重：/mnt/models/DeepSeek-V4-Flash
   数据：AISBench 原生合成。
   单机参考：D:/cmd_reference/deepseekv4flash/single-pd-split-1p1d/p-dp4-tp1_d-dp4-tp1_mooncake-hybrid
   双机参考：D:/cmd_reference/deepseekv4flash/double-pd-split-3p1d/dp1-pcp4-dspark1-3-mooncake-v2
   单机测试case：64K/1K，64并发（稳态），前缀重复目标50%、90%。
   双机测试case：64K/1K，128并发（稳态），前缀重复目标50%、90%。
   ```

1K=1024 tokens，输出长度是上限。省略总数时每个case正式请求数默认并发×4：本例单机256条、双机512条，另有预热1次，不主动清缓存。“稳态”映射steady_state=true并使用stable_stage；未提稳态或明确非稳态时使用default_perf；Agent 在执行前展示默认值与现场绑定。只测一种部署时删除另一种参考和case行。

前缀数据使用 AISBench 本体的单公共前缀，不需要 GSM8K、外部生成工具或代理脚本的单独路径；代理由所给参考目录提供。前缀重复目标不等于实际缓存命中率。原生tokenid解码后重新分词可能改变实际长度；记录长度和前缀偏差，不要求精确相等，不据此判失败或中止测试，详见[原生前缀数据](docs/generated-data.md)。

普通真实数据测试仍可明确提供JSONL/CSV路径，按原始数据测试；不要在原生合成示例中同时填写真实数据集路径。示例不是该镜像、模型和负载的通过承诺，执行前仍须现场审阅。

## 详细文档

- [使用与验证说明](docs/usage.md)：执行命令、整组任务、监控、检查项、资源释放与验证边界。
- [Skill 使用流程](SKILL.md)：操作步骤与能力范围。
- [任务配置示例](examples/job_spec.compact.json) · [现场绑定格式](docs/compact-cli.md)。
- [历史设计](docs/design.md) · [历史工程计划](docs/project-plan.md) · [决策记录](docs/decisions.md)。

- [开发与测试](docs/development.md)：目录约定、离线测试与现场验证边界。

- [原生 CLI 适配](docs/native-cli.md)：镜像安装、现场参数与结果语义。
