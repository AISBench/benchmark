# Ascend LLM Deploy Eval

面向昇腾模型部署与性能测试的 Agent skill。将自然语言需求整理为 `job_spec`，经配置审阅和现场预检后，自动执行部署、性能测试与资源释放。

## 快速开始

1. 将 [ascend-llm-deploy-eval](.) 安装到所用 Agent 的 skill 目录，并确认已加载。
2. 进入本 skill 目录，使用 Python 3.10+ 安装依赖：

   ```sh
   python -m pip install -r requirements.txt
   ```

3. 准备可免密 SSH 访问的 Linux 昇腾服务器、Docker、镜像、模型权重、语料、cache block 大小和部署参考脚本。替换以下占位内容后交给 Agent：

   ```text
   使用 $ascend-llm-deploy-eval 执行性能测试。
   服务器：root@node-a.example，SSH免密，工作目录/work/benchmark。
   镜像：registry.example/vllm-ascend:verified。
   模型：模型名称，权重/mnt/models/example-model。
   部署：单机混部，参考脚本/path/to/start.sh。
   测试：输入4096 tokens、输出512 tokens，并发1、4。
   ```

Agent 将收集缺失信息、整理配置并进行现场预检。示例资源仅为占位，实际执行前需完成配置审阅。

## 详细文档

- [使用与验证说明](docs/usage.md)：执行命令、整组任务、监控、检查项、资源释放与验证边界。
- [Skill 使用流程](SKILL.md)：操作步骤与能力范围。
- [任务配置示例](examples/job_spec.serial.json) · [现场绑定示例](examples/bindings.example.json) · [整组计划示例](examples/group-plan.example.json)。
- [当前设计](docs/design.md) · [工程计划](docs/project-plan.md) · [决策记录](docs/decisions.md)。

- [开发与测试](docs/development.md)：目录约定、离线测试与现场验证边界。

- [原生 CLI 适配](docs/native-cli.md)：镜像安装、现场参数与结果语义。
