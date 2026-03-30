# English Version
## 🌟 Highlights

1. **Image Generation Evaluation**: Integrated the first image generation evaluation benchmark GEdit-Bench, supporting evaluation of image generation models. (#159, #161, #162)
2. **Judge Model Support**: Added support for integrating judge models for evaluation, providing more flexible evaluation methods. (#170, #171, #175, #202, #204, #214)
3. **Mooncake Trace Support**: Added Mooncake Trace dataset for performance evaluation, supporting timestamp-based request scheduling, hash_id caching, and reproducible prompt generation. (#108, #118, #126, #127, #128, #137)

---

## 🚀 New Features

### Datasets

- Dataset: Added GEdit-Bench, supporting image generation model evaluation. (#159, #161, #162)
- Dataset: Added Mooncake Trace, supporting timestamp-based request scheduling, hash_id caching, and reproducible prompt generation. (#108, #118, #126, #127, #128, #137)

### Models

- Model: Added judge model support for evaluating model output quality. (#170, #171, #175, #202, #204, #214)
- Model: vllm service interface supports parsing input token length from the server. (#100)

### Features

- Feature: Support for MMLU Pro overall score calculation. (#216)
- Feature: Benchmark supports lower request rates. (#148)
- Feature: Added transformers version limit. (#141)

---

## 🐛 Bug Fixes

- Fix: Issue where TTFT and TPOT had no data when running Kimi2.5. (#153)
- Fix: Vocalsound evaluation bug. (#208)
- Fix: Reuse inference bugs in judge inference cases. (#202)
- Fix: Some bugs with same abbreviations. (#198)
- Fix: Model configuration compatibility issues in datasets and postprocessors. (#190)
- Fix: IPv6 support issue. (#186)
- Fix: lmm_example spelling mistake. (#180)
- Fix: BFCL dependency error. (#179)
- Fix: BFCL serialization failure in multiturn. (#117)
- Fix: Issue where concurrency could not be fully utilized in multi-process mode. (#63)
- Fix: MBPP evaluator class missing logger. (#115)
- Fix: BFCL eval import bug. (#133)

---

## 🔄 CI/CD Optimizations

- CI/CD: MR automated UT case execution.
- CI/CD: Added workflow to prevent issue auto-close. (#207)
- CI/CD: Added new workflow for smoke test. (#178)
- CI/CD: Added smoke test case for judge model. (#175)
- CI/CD: Smoke test case optimization. (#123)
- CI/CD: Enabled workflow for smoke test. (#86)
- CI/CD: Added issue bot. (#84)
- CI/CD: Moved smoke test project of benchmark to its repo. (#80)
- CI/CD: Issue-bot workflow added "ask deepwiki" recommendation. (#92)
- CI/CD: Fixed PR workflow run condition conflict. (#97)

---

## 📚 Documentation Updates

- Documentation: Added docs for judge evaluate and GEdit extend benchmark. (#170, #168)
- Documentation: Fixed docs about GEdit-Bench and judge models. (#204)
- Documentation: Improved FAQ, added tiktoken error issues that may occur in needle-bench evaluation. (#215)
- Documentation: Updated dataset links in README.md. (#213)
- Documentation: Added Mooncake Trace dataset support docs. (#128)
- Documentation: Updated README with latest developments. (#77)
- Documentation: Added doc for replicating accuracy from model's tech report or paper. (#78)
- Documentation: Added "ask deepwiki" in README. (#90)
- Documentation: Modified document structure to make datasets content clearer in Get Started section. (#156)

---

## 🔧 Other Optimizations

- Optimization: Updated the default version name of AISBench benchmark to 3.1.0. (#116)
- Optimization: Added third party libraries. (#119)
- Optimization: Added 3.1 design documents. (#120, #121)
- Optimization: Updated PR template. (#103)
- Optimization: Added auto label. (#106)
- Optimization: Fixed documentation. (#96)

---

## 🎯 Technical Highlights

1. **Expanded Evaluation Capabilities**: Added image generation evaluation and judge model evaluation, enriching evaluation scenarios.
2. **Performance Optimization**: Improved evaluation efficiency through multi-process mode concurrency optimization and low request rate support.
3. **Dataset Expansion**: Added Mooncake Trace dataset, supporting real business traffic simulation.
4. **CI/CD Improvement**: Strengthened automated testing and quality assurance, improving code quality.
5. **Documentation Enhancement**: Provided more detailed user guides and technical documentation, enhancing user experience.

---

*Release Date: 2026-03-30*

# 简体中文版
## 🌟 亮点

1. **图像生成评测**：接入首个图像生成类评测基准GEdit-Bench，支持对图像生成模型进行评测。（#159, #161, #162）
2. **裁判模型支持**：支持接入裁判模型进行评估，提供更灵活的评测方式。（#170, #171, #175, #202, #204, #214）
3. **Mooncake Trace支持**：新增Mooncake Trace数据集性能测评，支持按时间戳调度请求、hash_id 缓存与可复现 prompt 生成。（#108, #118, #126, #127, #128, #137）

---

## 🚀 新特性

### 数据集

- 数据集：新增GEdit-Bench，支持图像生成模型评测。（#159, #161, #162）
- 数据集：新增Mooncake Trace，支持按时间戳调度请求、hash_id 缓存与可复现 prompt 生成。（#108, #118, #126, #127, #128, #137）

### 模型

- 模型：新增裁判模型支持，用于评估模型输出质量。（#170, #171, #175, #202, #204, #214）
- 模型：vllm服务接口支持从服务器解析输入token长度。（#100）

### 功能

- 功能：支持MMLU Pro整体分数计算。（#216）
- 功能：benchmark支持更低的请求率。（#148）
- 功能：添加transformers版本限制。（#141）

---

## 🐛 问题修复

- 修复：运行Kimi2.5时TTFT和TPOT无数据的问题。（#153）
- 修复：vocalsound评测bug。（#208）
- 修复：裁判推理中的重用推理bug。（#202）
- 修复：相同缩写的一些bug。（#198）
- 修复：数据集和后处理器中的模型配置兼容性问题。（#190）
- 修复：IPv6支持问题。（#186）
- 修复：lmm_example拼写错误。（#180）
- 修复：bfcl依赖错误。（#179）
- 修复：BFCL多轮对话序列化失败问题。（#117）
- 修复：多进程模式下并发不能充分利用的问题。（#63）
- 修复：mbpp评估器类缺少logger的问题。（#115）
- 修复：bfcl_eval导入bug。（#133）

---

## 🔄 CI/CD 优化

- CI/CD：MR自动化执行UT用例。
- CI/CD：添加工作流防止issue自动关闭。（#207）
- CI/CD：添加冒烟测试工作流。（#178）
- CI/CD：添加裁判模型冒烟测试用例。（#175）
- CI/CD：冒烟测试用例优化。（#123）
- CI/CD：启用冒烟测试工作流。（#86）
- CI/CD：添加issue bot。（#84）
- CI/CD：将基准测试的冒烟测试项目移至其仓库。（#80）
- CI/CD：Issue-bot工作流添加"ask deepwiki"推荐。（#92）
- CI/CD：修复PR工作流运行条件冲突。（#97）

---

## 📚 文档更新

- 文档：添加裁判评估和GEdit扩展基准的文档。（#170, #168）
- 文档：修复GEdit-Bench和裁判模型的文档。（#204）
- 文档：完善FAQ，补充needle-bench评测可能出现的tiktoken报错问题。（#215）
- 文档：更新README.md中的数据集链接。（#213）
- 文档：添加Mooncake Trace数据集支持文档。（#128）
- 文档：更新README最新进展。（#77）
- 文档：添加从模型技术报告或论文中复制精度的文档。（#78）
- 文档：在README中添加"ask deepwiki"。（#90）
- 文档：修改文档结构，使datasets内容在Get Started部分更清晰。（#156）

---

## 🔧 其他优化

- 优化：更新AISBench基准的默认版本名称为3.1.0。（#116）
- 优化：添加第三方库。（#119）
- 优化：添加3.1设计文档。（#120, #121）
- 优化：更新PR模板。（#103）
- 优化：添加自动标签。（#106）
- 优化：修复文档。（#96）

---

## 🎯 技术亮点

1. **扩展评测能力**：新增图像生成评测和裁判模型评估，丰富了评测场景。
2. **性能优化**：通过多进程模式并发优化和低请求率支持，提升评测效率。
3. **数据集扩展**：新增Mooncake Trace数据集，支持真实业务流量模拟。
4. **CI/CD完善**：加强自动化测试和质量保证，提高代码质量。
5. **文档完善**：提供更详细的使用指南和技术文档，提升用户体验。

---

*发布日期：2026-03-30*

