# 为 AISBench 贡献代码

欢迎向 [AISBench/benchmark](https://github.com/AISBench/benchmark) 提交 Pull Request（PR）。本指南说明提交代码时应遵循的 PR 规范与代码风格。

本文件是贡献规范的统一入口。[中文开发指南](docs/source_zh_cn/develop_guide/contributing.md)和[英文开发指南](docs/source_en/develop_guide/contributing.md)提供本地化说明、环境准备与组件开发资料；其中的 PR 要求、提交规范和质量检查应与本文件保持一致。

## 1. 准备变更

- 所有代码变更通过 PR 提交到 `AISBench/benchmark` 的 `master` 分支。
- 每个 PR 聚焦一个独立问题，避免混入不相关的功能、重构或格式调整。
- PR 的新增行数应控制在 **1000 行以内**。新增行数超过 1000 行会触发 `large_pr` 检查，请拆分为可独立评审的多个 PR。
- 所有 PR 都必须关联 Issue。Bug 修复应先有对应 Issue；新功能或重大变更应先在 Issue 中讨论并达成共识。

### 从 Fork 创建贡献分支

外部开发者请先在 GitHub 上 Fork 仓库，再从上游 `master` 创建分支。以下命令中的 `YOUR_GITHUB_ID` 请替换为你的 GitHub 用户名。

```bash
git clone https://github.com/YOUR_GITHUB_ID/benchmark.git
cd benchmark

git remote add upstream https://github.com/AISBench/benchmark.git
git fetch upstream
git checkout -b feature/add-model-backend upstream/master
```

按变更类型命名分支：

| 变更类型 | 分支命名 |
| --- | --- |
| 新功能 | `feature/<简短功能名>` |
| Bug 修复 | `fix/<简短缺陷名>` |
| 文档更新 | `docs/<简短文档名>` |
| 代码重构 | `refactor/<简短重构名>` |

完成修改与自检后，将贡献分支推送到自己的 Fork，然后向上游仓库发起 PR：

```bash
git push -u origin feature/add-model-backend
```

创建 PR 时，确认目标仓库为 `AISBench/benchmark`，目标分支为 `master`，来源分支为你自己的贡献分支。

## 2. PR 标题与 Commit 规范

### 标题格式

PR 标题和 Commit Message 使用以下格式：

```text
[type][scope]<subject>
```

- `type`：变更类型，按下表选择。
- `scope`：影响模块，可选。不填写时使用 `[type]<subject>`。
- `subject`：简要说明实际变更，不要只写 `update`、`fix` 等笼统描述。`<subject>` 表示占位内容，实际标题不保留尖括号。

| Type | 含义 |
| --- | --- |
| `[feat]` | 新功能 |
| `[fix]` | Bug 修复 |
| `[docs]` | 文档更新 |
| `[style]` | 代码格式调整，不影响功能 |
| `[refactor]` | 代码重构 |
| `[test]` | 测试相关 |
| `[ci/cd]` | CI/CD 相关改动 |
| `[perf]` | 性能优化 |
| `[dependency]` | 依赖更新 |

常用 `scope` 包括：`models`、`datasets`、`inferencer`、`evaluator`、`calculators`、`cli`、`configs`、`plugins`、`docs`、`ci`。

示例：

```text
[feat][models] 添加 MindIE 流式推理后端支持
[fix][datasets] 修复 MBPP pass@k 计算的 task_id 聚合错误
[docs][readme] 更新 Python 版本要求说明
[test][calculators] 补充吞吐率计算器单元测试
[ci/cd][ci] 更新单元测试工作流
[dependency] 更新依赖版本约束
```

一个 Commit 只完成一个逻辑变更。复杂变更应在 Commit 正文中说明动机与取舍。

### DCO 和 Signed-off-by

AISBench 使用 [Developer Certificate of Origin (DCO)](https://developercertificate.org/)。每个 Commit 都必须包含 `Signed-off-by` 行，表示您确认贡献是原创的，或者您有权提交该贡献，并同意按项目的 Apache License 2.0 发布。

使用 `git commit -s` 自动添加签署行：

```bash
git commit -s -m "[docs] 更新贡献指南"
```

`-s` 使用 Git 中配置的姓名与邮箱，生成如下格式：

```text
Signed-off-by: Your Name <your.email@example.com>
```

DCO 签署与 PR 模板中的 CLA 是两项不同要求。添加 `Signed-off-by` 不代表已完成 CLA 签署；CLA 应按仓库提供的流程单独确认。

## 3. 填写 PR 描述

创建 PR 时，请完整填写仓库自动提供的 [PR 模板](https://github.com/AISBench/benchmark/blob/master/.github/pull_request_template.md)，保留模板章节，并如实勾选检查项。

### PR Type

根据实际变更，将相应类型前的 `- [ ]` 改为 `- [x]`：

```markdown
- [ ] Feature（功能新增）
- [ ] Bugfix（Bug 修复）
- [ ] Docs（文档更新）
- [ ] CI/CD（持续集成/持续部署）
- [ ] Refactor（代码重构）
- [ ] Perf（性能优化）
- [ ] Dependency（依赖项更新）
- [ ] Test-Cases（测试用例更新）
- [ ] Other（其他）
```

请保留类型选项的原始文案，自动打标签依赖模板匹配。**标题中的 `[feat]` 等前缀不能代替 PR Type 勾选项。**

选择 CI/CD 或 Dependency 后，如果未出现对应的 `ci/cd` 或 `dependency` 标签，请联系维护者补充标签。

### 关联 Issue

在 `Related Issue` 中填写明确的 Issue 编号，例如：

```text
Fixes #123
```

如果 PR 只是与某个 Issue 相关，尚不能完整解决该问题，请使用：

```text
Relates to #123
```

不要只写问题描述而省略 Issue 编号。

### 其他模板章节

| 章节 | 填写要求 |
| --- | --- |
| Motivation / 变更动机 | 说明要解决的问题及变更目标 |
| Modification / 修改内容 | 概述实现方式、主要修改和影响范围 |
| Associated Test Results / 关联测试结果 | 提供执行命令、测试结果，以及相关 CI 或测试报告链接 |
| BC-breaking / 向后不兼容变更 | 如存在不兼容变更，说明影响及下游迁移方式 |
| Performance degradation / 性能下降 | 如存在性能下降，说明影响及预期改进 |
| Use cases / 使用案例 | 新功能建议提供使用示例，并更新对应文档 |
| Checklist / 检查列表 | 完成对应事项后再勾选，不要预先全部勾选 |
| Collaboration Info / 协作信息 | 填写建议审核人、相关模块负责人及其他协作说明 |

不适用的可选章节可填写“不适用”及简要原因。

涉及 API、配置格式、默认行为变更或移除已弃用功能时，应在编码前讨论，尽量保持兼容。无法兼容时提供迁移指南，并同步更新相关中英双语文档、测试和 CHANGELOG。

模板中的 CLA 检查项应按仓库提供的签署流程完成。如果未找到签署入口，请在 PR 中联系维护者确认，未完成时不要勾选。

## 4. 提交前自检

### 测试与文档

- 新功能应补充对应单元测试，Bug 修复应增加覆盖该缺陷的回归用例。
- `tests/UT/<模块>` 应与 `ais_bench/benchmark/<模块>` 对应。
- 测试函数使用 `test_` 前缀，推荐命名为 `test_<功能>_<条件>()`。
- 测试用例应独立，不依赖执行顺序。外部 API、文件 IO 和网络操作应使用 Mock，用例结束后清理文件、进程与临时数据。
- 新增功能同步更新 `docs/source_zh_cn/` 和 `docs/source_en/` 中的相关文档。API 变更同步更新使用说明；新增模型或数据集配置提供 `README.md`。
- 如变更影响下游项目，应补充对应的兼容性验证结果。

在仓库根目录执行本地单元测试：

```bash
python3 tests/run_tests.py tests/UT -p 16
```

修改 `prefix_cache` 插件时，补充运行插件测试：

```bash
python3 tests/run_tests.py plugins/prefix_cache/tests \
  --source-dirs plugins/prefix_cache/ais_bench_prefix_cache \
  --output test_reports/prefix_cache
```

CI 对master与 `prefix_cache` 插件分别检查以下指标，任一未达标即失败：

| 检查项 | 要求 |
| --- | --- |
| 测试报告 | `test_summary.txt` 必须存在 |
| 失败用例数 | 0 |
| Total coverage | ≥ 80% |
| Line coverage | ≥ 80% |
| Branch coverage | ≥ 60% |

不要通过删除测试用例或扩大覆盖率排除范围来达到阈值。

### PR 质量检查

PR 提交后，`PR Quality Gate` 检查以下内容：

| 失败标签 | 含义 | 处理方式 |
| --- | --- | --- |
| `large_pr` | 新增行数超过 1000 行 | 拆分 PR |
| `no_issue` | 未关联 Issue | 在描述中补充 Issue 引用 |
| `no_review` | 缺少已提交的正式 Review | 邀请审核人提交 Review |
| `review_opened` | 存在未解决的行内检视意见 | 修改并回复，等待意见提出者或维护者确认闭环 |
| `no_UT` | Feature，或新增行数 > 50 的 Bugfix，缺少 `tests/UT/` 下的文件新增或修改 | 补充对应单元测试 |

对于仅属于 Docs、CI/CD、Refactor、Perf、Dependency、Test-Cases、Other 的 PR，以及新增行数 ≤ 50 的 Bugfix PR，质量检查不强制要求 `tests/UT/` 变更。仍应根据实际行为变化补充必要测试，尤其是 Bug 的回归用例。

整改后可推送新 Commit，或在 PR 下评论 `/pr_check` 重新运行质量检查。检查通过后，对应的失败标签会自动移除。

## 5. 配合 Review

- 邀请相关审核人提交正式 Review，普通 PR 评论不能替代正式 Review。
- 逐条回复检视意见，说明修改结果或提供理由。
- 行内意见应由提出者或维护者确认并 Resolve，请勿自行 Resolve 他人提出的意见。
- 更新自己的贡献分支即可更新原 PR，无需因每次修改重新创建 PR。
- 保持贡献分支与上游 `master` 同步，并解决合并冲突。
- 查看 PR 的 Checks，修复单元测试、覆盖率和适用的冒烟测试失败。
- 来自 Fork 的冒烟测试可能需要维护者批准后才执行。等待审批时可在 PR 中联系维护者。

PR 合入前，必须通过质量检查及适用的 CI 检查，获得至少 1 位维护者批准，并闭环所有行内意见。

提交前可使用以下清单：

- [ ] PR 聚焦一个独立变更，新增行数不超过 1000 行。
- [ ] PR 标题及 Commit Message 符合规定格式。
- [ ] 每个 Commit 都有 `Signed-off-by`。
- [ ] PR Type 已正确勾选，已关联 Issue。
- [ ] 已完整说明动机、修改内容及测试结果。
- [ ] 已执行格式与语法自检，补充必要测试并更新相关文档。
- [ ] 已说明兼容性、性能或下游影响（如适用）。

## 6. 代码风格

### Python 排版与命名

- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/)，每级缩进使用 **4 个空格**。
- 项目建议每行不超过 **120 个字符**，较长表达式应合理换行。
- 函数、变量使用小写加下划线，例如 `calculate_score`；类名使用大驼峰，例如 `ScoreCalculator`；常量使用全大写加下划线，例如 `MAX_RETRIES`。
- 保持单一职责，避免重复代码。复杂逻辑使用注释说明原因与处理方式。
- 修改已有代码时，保持所在模块的风格一致，避免引入与本次变更无关的全文件格式调整。

### Docstring 与类型注解

公共类与函数必须编写 docstring（文档字符串），说明用途。参数、返回值、异常或使用限制等内容应根据接口复杂程度补充。

建议为函数参数与返回值添加类型注解，例如：

```python
def calculate_total_tokens(token_counts: list[int]) -> int:
    """Calculate the total number of tokens across requests.

    Args:
        token_counts: Token count for each request.

    Returns:
        The sum of the token counts.
    """
    return sum(token_counts)
```

上例中，三引号包围的内容是 docstring，`list[int]` 和 `-> int` 分别说明参数类型与返回值类型。

### 日志、异常与组件注册

- 统一使用 `AISLogger` 输出日志。
- 异常统一使用 `ais_bench.benchmark.utils.logging.exceptions` 中的自定义异常类。
- 新增组件必须通过对应注册器接入：

| 组件 | 注册器 |
| --- | --- |
| 模型 | `@MODELS.register_module()` |
| 数据集加载 | `@LOAD_DATASET.register_module()` |
| 评估器 | `@ICL_EVALUATORS.register_module()` |
| 推理器 | `@ICL_INFERENCERS.register_module()` |

### 格式与语法自检

使用 Black、isort 或仓库提供的 lint 工具检查本次修改的文件。以下命令中的路径请替换为实际文件：

```bash
black path/to/changed_file.py
isort path/to/changed_file.py
python -m py_compile path/to/changed_file.py
```

格式检查通过后，请再次检查 diff，确认没有提交不相关的修改。
