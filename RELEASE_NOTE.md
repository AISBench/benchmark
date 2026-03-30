# English Version
## 🌟 Highlights

1. **Image Generation Evaluation**: Integrated the first image generation evaluation benchmark GEdit-Bench, supporting evaluation of image generation models. (#159, #161, #162)
2. **Judge Model Support**: Added support for integrating judge models for evaluation, providing more flexible evaluation methods. (#170, #171, #175, #202, #204, #214)
3. **Mooncake Trace Support**: Added Mooncake Trace dataset for performance evaluation, supporting timestamp-based request scheduling, hash_id caching, and reproducible prompt generation. (#108, #118, #126, #127, #128, #137)

---

## 🚀 New Features

### Datasets

- **Dataset**: Added GEdit-Bench, supporting image generation model evaluation with comprehensive evaluation metrics. (#159, #161, #162)
- **Dataset**: Added Mooncake Trace, supporting timestamp-based request scheduling, hash_id caching, and reproducible prompt generation for realistic performance testing. (#108, #118, #126, #127, #128, #137)

### Models

- **Model**: Added judge model support for evaluating model output quality, enabling more flexible and customizable evaluation criteria. (#170, #171, #175, #202, #204, #214)
- **Model**: Enhanced vllm service interface to support parsing input token length from the server, improving request handling accuracy. (#100)

### Features

- **Feature**: Added support for MMLU Pro overall score calculation, providing comprehensive evaluation of model performance across multiple dimensions. (#216)
- **Feature**: Enhanced benchmark to support lower request rates, enabling more granular performance testing scenarios. (#148)
- **Feature**: Added transformers version limit to ensure compatibility and stability. (#141)

---

## 🐛 Bug Fixes

- **Fix**: Resolved issue where TTFT (Time To First Token) and TPOT (Time Per Output Token) had no data when running Kimi2.5 model. (#153)
- **Fix**: Corrected vocalsound evaluation bug to ensure accurate audio-related assessments. (#208)
- **Fix**: Fixed reuse inference bugs in judge inference cases to ensure consistent evaluation results. (#202)
- **Fix**: Resolved issues with same abbreviations causing conflicts in evaluation. (#198)
- **Fix**: Addressed model configuration compatibility issues in datasets and postprocessors. (#190)
- **Fix**: Implemented IPv6 support to ensure compatibility with modern network environments. (#186)
- **Fix**: Corrected lmm_example spelling mistake in documentation and code references. (#180)
- **Fix**: Resolved BFCL (Berkeley Function Calling Leaderboard) dependency error to ensure proper installation. (#179)
- **Fix**: Fixed BFCL serialization failure in multiturn scenarios to ensure consistent function calling evaluation. (#117)
- **Fix**: Resolved issue where concurrency could not be fully utilized in multi-process mode, improving evaluation efficiency. (#63)
- **Fix**: Added missing logger to MBPP (Mostly Basic Python Problems) evaluator class for better debugging. (#115)
- **Fix**: Corrected BFCL eval import bug to ensure proper module loading. (#133)

---

## 🔄 CI/CD Optimizations

- **CI/CD**: Implemented MR (Merge Request) automated UT (Unit Test) case execution to ensure code quality before merging. (#301)
- **CI/CD**: Added workflow to prevent issue auto-close, ensuring issues are properly addressed before closure. (#207)
- **CI/CD**: Added new workflow for smoke test to quickly verify basic functionality. (#178)
- **CI/CD**: Added smoke test case for judge model to ensure judge evaluation functionality works correctly. (#175)
- **CI/CD**: Optimized smoke test cases to improve test coverage and efficiency. (#123)
- **CI/CD**: Enabled workflow for smoke test to automate basic functionality verification. (#86)
- **CI/CD**: Added issue bot to automate issue management and triage. (#84)
- **CI/CD**: Moved smoke test project of benchmark to its repository for better organization. (#80)
- **CI/CD**: Added "ask deepwiki" recommendation to Issue-bot workflow for better issue resolution. (#92)
- **CI/CD**: Fixed PR workflow run condition conflict to ensure consistent CI/CD execution. (#97)

---

## 📚 Documentation Updates

- **Documentation**: Added comprehensive docs for judge evaluate and GEdit extend benchmark, including setup and usage guides. (#170, #168)
- **Documentation**: Fixed docs about GEdit-Bench and judge models to ensure accurate usage instructions. (#204)
- **Documentation**: Improved FAQ, adding tiktoken error issues that may occur in needle-bench evaluation with troubleshooting steps. (#215)
- **Documentation**: Updated dataset links in README.md to ensure accurate and up-to-date references. (#213)
- **Documentation**: Added detailed Mooncake Trace dataset support docs, including usage examples and configuration options. (#128)
- **Documentation**: Updated README with latest developments to keep users informed of new features. (#77)
- **Documentation**: Added doc for replicating accuracy from model's tech report or paper to ensure consistent evaluation methodology. (#78)
- **Documentation**: Added "ask deepwiki" in README to provide users with an additional support channel. (#90)
- **Documentation**: Modified document structure to make datasets content clearer in Get Started section, improving user navigation. (#156)

---

## 🔧 Other Optimizations

- **Optimization**: Updated the default version name of AISBench benchmark to 3.1.0 to reflect the significant updates. (#116)
- **Optimization**: Added third party libraries to support new features and improve functionality. (#119)
- **Optimization**: Added 3.1 design documents to provide technical insights into the architecture and future roadmap. (#120, #121)
- **Optimization**: Updated PR template to standardize contribution guidelines and improve code review process. (#103)
- **Optimization**: Added auto label functionality to automate issue and PR categorization. (#106)
- **Optimization**: Fixed documentation errors and inconsistencies to improve user experience. (#96)

---

## 🎯 Technical Highlights

1. **Expanded Evaluation Capabilities**: Added image generation evaluation and judge model evaluation, enriching evaluation scenarios with new dimensions of assessment.
2. **Performance Optimization**: Improved evaluation efficiency through multi-process mode concurrency optimization and low request rate support, enabling more accurate performance testing.
3. **Dataset Expansion**: Added Mooncake Trace dataset, supporting real business traffic simulation for more realistic performance evaluation.
4. **CI/CD Improvement**: Strengthened automated testing and quality assurance with comprehensive CI/CD workflows, improving code quality and reliability.
5. **Documentation Enhancement**: Provided more detailed user guides and technical documentation, enhancing user experience and reducing onboarding time.

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

- **数据集**：新增GEdit-Bench，支持图像生成模型评测，提供全面的评估指标。（#159, #161, #162）
- **数据集**：新增Mooncake Trace，支持按时间戳调度请求、hash_id 缓存与可复现 prompt 生成，用于真实场景的性能测试。（#108, #118, #126, #127, #128, #137）

### 模型

- **模型**：新增裁判模型支持，用于评估模型输出质量，提供更灵活和可定制的评估标准。（#170, #171, #175, #202, #204, #214）
- **模型**：增强vllm服务接口，支持从服务器解析输入token长度，提高请求处理准确性。（#100）

### 功能

- **功能**：新增MMLU Pro整体分数计算支持，提供模型在多个维度上的综合性能评估。（#216）
- **功能**：增强benchmark支持更低的请求率，实现更精细的性能测试场景。（#148）
- **功能**：添加transformers版本限制，确保兼容性和稳定性。（#141）

---

## 🐛 问题修复

- **修复**：解决运行Kimi2.5时TTFT（首token时间）和TPOT（每输出token时间）无数据的问题。（#153）
- **修复**：修正vocalsound评测bug，确保音频相关评估的准确性。（#208）
- **修复**：修复裁判推理中的重用推理bug，确保评估结果的一致性。（#202）
- **修复**：解决相同缩写导致评估冲突的问题。（#198）
- **修复**：解决数据集和后处理器中的模型配置兼容性问题。（#190）
- **修复**：实现IPv6支持，确保与现代网络环境的兼容性。（#186）
- **修复**：修正lmm_example拼写错误，确保文档和代码引用的准确性。（#180）
- **修复**：解决BFCL（伯克利函数调用排行榜）依赖错误，确保正确安装。（#179）
- **修复**：修复BFCL多轮对话序列化失败问题，确保函数调用评估的一致性。（#117）
- **修复**：解决多进程模式下并发不能充分利用的问题，提高评估效率。（#63）
- **修复**：为MBPP（Mostly Basic Python Problems）评估器类添加缺失的logger，改善调试体验。（#115）
- **修复**：修正BFCL eval导入bug，确保模块正确加载。（#133）

---

## 🔄 CI/CD 优化

- **CI/CD**：实现MR（合并请求）自动化UT（单元测试）用例执行，确保合并前的代码质量。（#301）
- **CI/CD**：添加工作流防止issue自动关闭，确保问题在关闭前得到妥善处理。（#207）
- **CI/CD**：添加冒烟测试工作流，快速验证基本功能。（#178）
- **CI/CD**：添加裁判模型冒烟测试用例，确保裁判评估功能正常工作。（#175）
- **CI/CD**：优化冒烟测试用例，提高测试覆盖率和效率。（#123）
- **CI/CD**：启用冒烟测试工作流，自动验证基本功能。（#86）
- **CI/CD**：添加issue bot，自动管理和分类问题。（#84）
- **CI/CD**：将基准测试的冒烟测试项目移至其仓库，提高组织效率。（#80）
- **CI/CD**：在Issue-bot工作流中添加"ask deepwiki"推荐，改善问题解决流程。（#92）
- **CI/CD**：修复PR工作流运行条件冲突，确保CI/CD执行的一致性。（#97）

---

## 📚 文档更新

- **文档**：添加裁判评估和GEdit扩展基准的综合文档，包括设置和使用指南。（#170, #168）
- **文档**：修复GEdit-Bench和裁判模型的文档，确保使用说明的准确性。（#204）
- **文档**：完善FAQ，添加needle-bench评测中可能出现的tiktoken错误问题及解决步骤。（#215）
- **文档**：更新README.md中的数据集链接，确保引用的准确性和时效性。（#213）
- **文档**：添加详细的Mooncake Trace数据集支持文档，包括使用示例和配置选项。（#128）
- **文档**：更新README最新进展，让用户了解新功能。（#77）
- **文档**：添加从模型技术报告或论文中复制精度的文档，确保评估方法的一致性。（#78）
- **文档**：在README中添加"ask deepwiki"，为用户提供额外的支持渠道。（#90）
- **文档**：修改文档结构，使datasets内容在Get Started部分更清晰，改善用户导航体验。（#156）

---

## 🔧 其他优化

- **优化**：更新AISBench基准的默认版本名称为3.1.0，以反映重大更新。（#116）
- **优化**：添加第三方库，支持新功能并改进功能。（#119）
- **优化**：添加3.1设计文档，提供架构和未来路线图的技术洞察。（#120, #121）
- **优化**：更新PR模板，标准化贡献指南并改进代码审查流程。（#103）
- **优化**：添加自动标签功能，自动对问题和PR进行分类。（#106）
- **优化**：修复文档错误和不一致之处，改善用户体验。（#96）

---

## 🎯 技术亮点

1. **扩展评测能力**：新增图像生成评测和裁判模型评估，丰富了评测场景，提供新的评估维度。
2. **性能优化**：通过多进程模式并发优化和低请求率支持，提高评估效率，实现更准确的性能测试。
3. **数据集扩展**：新增Mooncake Trace数据集，支持真实业务流量模拟，实现更真实的性能评估。
4. **CI/CD完善**：通过全面的CI/CD工作流加强自动化测试和质量保证，提高代码质量和可靠性。
5. **文档完善**：提供更详细的用户指南和技术文档，增强用户体验，减少上手时间。

---

*发布日期：2026-03-30*
