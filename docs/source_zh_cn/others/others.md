# 🔜 即将推出
- **\[2025.9\]** 提供业界主流Agent测评能力，支持DeepSeek V3.1 Search/Code Agent测评
- **\[2025.10\]** 支持在AISBench框架下🔌插件化集成前沿测试基准，以应对业界愈发复杂多样化的测试任务
- **\[2025.11\]** 提供业界前沿的多模态测评能力
- [x] **\[2025.8\]** 将支持ShareGPT、BFCL等多轮对话数据集的性能评测。
- [x] **\[2025.8\]** 优化性能测评中评估eval阶段的计算效率，优化工具显存占用，补充工具使用规格说明。
- [x] **\[2025.7\]** 性能评测场景使用自定义数据集，将支持定义每条数据对应的最大输出长度限制。

# 🤝 致谢
- 本项目代码基于🔗 [OpenCompass](https://github.com/open-compass/opencompass)做拓展开发。
- 本项目部分数据集和提示词实现修改自[simple-evals](https://github.com/openai/simple-evals)。
- 本项目代码中打点的性能指标与[VLLM Benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks)对齐。
- 本项目的BFCL函数调用能力评估功能基于 [Berkeley Function Calling Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) 实现。