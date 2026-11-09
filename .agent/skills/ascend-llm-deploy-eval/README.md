# Ascend LLM Deploy Eval

面向昇腾模型部署与性能测试的 Agent skill。将自然语言需求整理为 `job_spec`，经配置审阅和现场预检后，自动执行部署、性能测试与资源释放。

## 快速开始

1. 将 [ascend-llm-deploy-eval](.) 安装到所用 Agent 的 skill 目录，并确认已加载。
2. 进入本 skill 目录，使用 Python 3.10+ 安装依赖：

   ```sh
   python -m pip install -r requirements.txt
   ```

3. 准备可免密 SSH 访问的 Linux 昇腾服务器、Docker、镜像、模型权重、外部数据集路径和部署参考脚本；测试镜像内预装 AISBench。替换以下占位内容后交给 Agent：

   ```text
   使用 $ascend-llm-deploy-eval 执行性能测试。
   服务器：root@node-a.example，SSH免密，任务与日志目录/work/benchmark（仓库外）。
   镜像：registry.example/vllm-ascend:verified，容器内已安装可运行的AISBench。
   模型：模型名称，权重与tokenizer路径/mnt/models/example-model。
   部署：单机混部，参考脚本/path/to/start.sh。
   数据集：服务器上的/data/benchmark/dataset.jsonl，JSONL格式，输入字段question、参考字段answer；按原始数据测试，不生成或复制数据集。
   测试：最大输出512 tokens，并发1、4，每个case正式请求100条，预热1次，请求速率不限。
   输入长度：使用数据集实际长度，不要求定长。
   缓存：不主动清理；统计使用default_perf。
   日志：启用AISBench默认落盘，保留主进程日志、原生worker日志和性能结果；提交前给出进展、服务日志、测试日志查看命令。
   ```

Agent 将收集缺失信息、整理配置并进行现场预检。示例资源仅为占位，实际执行前需完成配置审阅。

## 详细文档

- [使用与验证说明](docs/usage.md)：执行命令、整组任务、监控、检查项、资源释放与验证边界。
- [Skill 使用流程](SKILL.md)：操作步骤与能力范围。
- [任务配置示例](examples/job_spec.compact.json) · [现场绑定格式](docs/compact-cli.md)。
- [历史设计](docs/design.md) · [历史工程计划](docs/project-plan.md) · [决策记录](docs/decisions.md)。

- [开发与测试](docs/development.md)：目录约定、离线测试与现场验证边界。

- [原生 CLI 适配](docs/native-cli.md)：镜像安装、现场参数与结果语义。
