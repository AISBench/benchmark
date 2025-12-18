# 贡献指南

感谢您对 AISBench 评测工具的关注！我们欢迎所有形式的贡献，包括但不限于代码、文档、问题报告和建议。

## 目录

- [任务看板](#任务看板)
- [许可证](#许可证)
- [开发](#开发)
- [代码检查（Linting）](#代码检查linting)
- [文档](#文档)
- [测试](#测试)
- [问题（Issues）](#问题issues)
- [拉取请求（Pull Requests）与代码审查（Code Reviews）](#拉取请求pull-requests与代码审查code-reviews)
- [DCO 和 Signed-off-by](#dco-和-signed-off-by)
- [PR 标题和分类](#pr-标题和分类)
- [代码质量](#代码质量)
- [添加或更改组件](#添加或更改组件)
- [重大变更须知](#重大变更须知)
- [对审查的期望](#对审查的期望)
- [致谢](#致谢)

## 任务看板

我们使用 GitHub Issues 来跟踪任务和问题。在开始贡献之前，建议您：

1. 查看现有的 [Issues](https://github.com/AISBench/benchmark/issues) 以了解当前的工作重点
2. 如果是新功能或重大变更，建议先创建一个 Issue 进行讨论
3. 对于 bug 修复，可以直接提交 Pull Request

## 许可证

AISBench 使用 [Apache License 2.0](../../../LICENSE) 许可证。通过向本项目贡献代码，您同意您的贡献将在相同的许可证下发布。

所有贡献者必须确保：

- 您拥有或有权许可您提交的代码
- 您的贡献不侵犯任何第三方的知识产权
- 您同意将您的贡献授权给项目维护者

## 开发

### 环境设置

1. **Python 版本要求**：仅支持 Python 3.10、3.11 或 3.12

2. **克隆仓库**：

```bash
git clone https://github.com/AISBench/benchmark.git
cd benchmark/
```

3. **创建开发环境**（推荐使用 Conda）：

```bash
conda create --name ais_bench_dev python=3.10 -y
conda activate ais_bench_dev
```

4. **安装开发依赖**：

```bash
# 安装核心依赖
pip3 install -e ./ --use-pep517

# 安装开发依赖（如需要）
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```

### 代码结构

AISBench 采用基于注册器（Registry）的插件化架构：

- **核心插件模块**：位于 `ais_bench/benchmark/` 目录下
  - `models/`：模型运行类
  - `datasets/`：数据集加载器
  - `openicl/icl_evaluator`: ICL评估器
  - `openicl/icl_inferencer`: ICL推理器
  - `calculators/`：性能指标计算器
  - `registry.py`：注册器定义

- **配置文件**：位于 `ais_bench/benchmark/configs/` 目录下
  - `models/`：模型配置
  - `datasets/`：数据集配置
  - `summarizers/`：结果汇总器配置

- **插件系统**：支持通过 entry_point 机制扩展功能，详见 [插件开发指南](../../../plugin_examples/README.md)

### 开发工作流

1. **创建分支**：

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/your-bug-fix
```

2. **进行开发**：编写代码、添加测试、更新文档

3. **提交更改**：

```bash
git add .
git commit -m "feat: 添加新功能描述"
```

4. **推送并创建 Pull Request**

## 代码检查（Linting）

在提交代码之前，请确保代码符合项目的代码规范：

### Python 代码风格

- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 代码风格指南
- 使用 4 个空格进行缩进
- 行长度建议不超过 120 个字符
- 使用有意义的变量和函数名

### 代码格式化工具

建议使用以下工具进行代码格式化：

```bash
# 使用 black 进行代码格式化（如果项目配置了）
black ais_bench/

# 使用 isort 进行导入排序（如果项目配置了）
isort ais_bench/
```

### 静态检查

在提交前运行基本的 Python 语法检查：

```bash
python -m py_compile your_file.py
```

## 文档

### 文档结构

项目文档位于 `docs/` 目录下：

- `source_zh_cn/`：中文文档
- `source_en/`：英文文档

### 文档编写规范

1. **Markdown 格式**：使用 Markdown 编写文档
2. **中文文档**：使用简体中文，语言简洁明了
3. **代码示例**：提供完整可运行的代码示例
4. **链接检查**：确保所有链接有效

### 更新文档

- 添加新功能时，请同步更新相关文档
- 修改 API 时，请更新对应的文档说明
- 添加数据集或模型配置时，请提供 README.md 说明

## 测试

### 测试结构

测试文件位于 `tests/` 目录下：

- `UT/`：单元测试（已集成到CI）

### 运行测试

安装测试相关依赖

```bash
# 运行所有测试
python tests/run_tests.py

# 多进程运行所有测试
python tests/run_tests.py -p 4 # 4个进程

# 运行特定测试文件
pytest tests/UT/test_example.py
```

### 编写测试

- **单元测试**：测试单个函数或类的功能
- **集成测试**：测试多个组件的交互
- **测试命名**：使用 `test_` 前缀命名测试函数
- **测试覆盖**：新功能应包含相应的测试用例，目标覆盖率不低于 80%

## 问题（Issues）

### 报告 Bug

在报告 bug 时，请提供以下信息：

1. **问题描述**：清晰描述问题现象
2. **复现步骤**：提供可复现的步骤
3. **预期行为**：描述期望的行为
4. **实际行为**：描述实际发生的行为
5. **环境信息**：
   - Python 版本
   - 操作系统
   - AISBench 版本
   - 相关依赖版本
6. **日志信息**：提供相关的错误日志或堆栈跟踪

### 功能请求

在提出功能请求时，请说明：

1. **功能描述**：详细描述您希望添加的功能
2. **使用场景**：说明该功能的使用场景和价值
3. **可能的实现方案**：如果有想法，可以提出实现建议

## 拉取请求（Pull Requests）与代码审查（Code Reviews）

### 创建 Pull Request

1. **准备工作**：
   - 确保代码已通过所有测试
   - 确保代码符合代码规范
   - 更新相关文档

2. **PR 描述**：
   - 清晰描述 PR 的目的和变更内容
   - 如果解决了某个 Issue，请使用 `Fixes #123` 格式关联
   - 提供测试结果或截图（如适用）

3. **代码审查**：
   - 保持 PR 规模适中，便于审查
   - 及时响应审查意见
   - 根据反馈进行修改

### 审查流程

1. **自动化检查**：PR 创建后会自动运行 CI 检查
2. **代码审查**：维护者会审查代码
3. **修改反馈**：根据审查意见进行修改
4. **合并**：审查通过后，维护者会合并 PR

## DCO 和 Signed-off-by

AISBench 使用 [Developer Certificate of Origin (DCO)](https://developercertificate.org/) 来确保贡献的合法性。

### 如何签署提交

在提交信息中添加 `Signed-off-by` 行：

```bash
git commit -s -m "feat: 添加新功能"
```

`-s` 参数会自动添加 `Signed-off-by` 行。

### Signed-off-by 格式

```
Signed-off-by: Your Name <your.email@example.com>
```

### 含义

通过添加 `Signed-off-by`，您确认：

1. 贡献是您原创的，或者您有权以开源许可证提交它
2. 您同意贡献在 Apache License 2.0 下发布

## PR 标题和分类

### 标题格式

使用以下格式：

```
[type][scope]<subject>
```

其中：
- `[type]`：提交类型，见下方类型说明
- `[scope]`：影响范围（可选），如 models、datasets、docs 等
- `<subject>`：简短的描述性标题

### 类型（Type）

- `[feat]`：新功能
- `[fix]`：Bug 修复
- `[docs]`：文档更新
- `[style]`：代码格式调整（不影响功能）
- `[refactor]`：代码重构
- `[test]`：测试相关
- `[chore]`：构建过程或辅助工具的变动
- `[perf]`：性能优化

### 示例

```
[feat][models]添加新的模型后端支持
[fix][datasets]修复数据集加载错误
[docs][readme]更新安装说明
[test][calculators]添加性能计算器测试
```

## 代码质量

### 代码审查标准

代码审查时会关注以下方面：

1. **功能正确性**：代码是否实现了预期功能
2. **代码质量**：代码是否清晰、可维护
3. **测试覆盖**：是否有足够的测试覆盖
4. **文档完整性**：是否更新了相关文档
5. **性能影响**：是否对性能有负面影响
6. **向后兼容性**：是否破坏了现有功能

### 最佳实践

1. **单一职责**：每个函数/类应该只做一件事
2. **DRY 原则**：避免重复代码
3. **注释清晰**：为复杂逻辑添加注释
4. **错误处理**：妥善处理异常情况
5. **类型提示**：建议使用类型提示提高代码可读性

## 添加或更改组件

### 添加新的模型后端

参考[支持新模型](./new_model.md)

### 添加新的数据集和精度评估器

参考[支持新的数据集和精度评估器](./new_dataset.md)

### 添加新的推理器

参考[支持新的推理器](./new_inferencer.md)

## 重大变更须知

请尽量保持变更简洁。对于重大的架构性变更（>500 行代码，不包括数据/配置/测试），我们期望您：

1. **提前讨论**：在 GitHub Issues 中创建讨论，说明变更原因和影响范围
2. **向后兼容**：尽量保持向后兼容，或提供迁移指南
3. **文档更新**：更新所有相关文档
4. **测试覆盖**：确保有充分的测试覆盖
5. **版本说明**：在 CHANGELOG 中详细说明变更

### 重大变更示例

- API 接口变更
- 配置文件格式变更
- 默认行为变更
- 移除已弃用的功能

## 对审查的期望

### 作为贡献者

- **耐心等待**：审查可能需要一些时间，请耐心等待
- **积极回应**：及时响应审查意见，进行必要的修改
- **保持礼貌**：在讨论中保持专业和礼貌的态度
- **持续改进**：将审查视为学习机会，不断改进代码质量

### 作为审查者

- **及时响应**：尽快审查 PR，提供建设性反馈
- **友好沟通**：以友好和专业的方式提供反馈
- **解释原因**：解释为什么需要修改，帮助贡献者理解
- **认可贡献**：认可好的贡献，鼓励继续参与

## 致谢

感谢所有为 AISBench 做出贡献的开发者！您的贡献使这个项目变得更好。

### 贡献方式

您可以通过以下方式贡献：

- 🐛 报告 Bug
- 💡 提出功能建议
- 📝 改进文档
- 💻 提交代码
- 🧪 编写测试
- 🔍 代码审查
- 📢 推广项目

### 贡献者名单

所有贡献者都会在项目中被认可。感谢每一位贡献者的努力！

---

如有任何问题，请随时在 [GitHub Issues](https://github.com/AISBench/benchmark/issues) 中提出。
