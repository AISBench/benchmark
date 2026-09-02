Thanks for your contribution; we appreciate it a lot. The following instructions will make your pull request healthier and help you get feedback more easily. If you do not understand some items, don't worry, just make the pull request and seek help from maintainers.
感谢您的贡献，我们非常重视。以下说明将使您的拉取请求更健康，更易于获得反馈。如果您不理解某些项目，请不要担心，只需提交拉取请求并从维护人员那里寻求帮助即可。

**PR Type / PR类型**
- [ ] Feature（功能新增）
- [ ] Bugfix（Bug 修复）
- [ ] Docs（文档更新）
- [ ] CI/CD（持续集成/持续部署）
- [ ] Refactor（代码重构）
- [ ] Perf（性能优化）
- [ ] Dependency（依赖项更新）
- [ ] Test-Cases（测试用例更新）
- [ ] Other（其他）

**Related Issue | 关联 Issue**
Fixes #(issue ID / issue 编号) / Relates to #(issue ID / issue 编号)

## 🔍 Motivation / 变更动机

Please describe the motivation of this PR and the goal you want to achieve through this PR.
请描述您的拉取请求的动机和您希望通过此拉取请求实现的目标。

## 📝 Modification / 修改内容

Please briefly describe what modification is made in this PR.
请简要描述此拉取请求中进行的修改。

## 📐 Associated Test Results / 关联测试结果

Please provide links to the related test results, such as CI pipelines, test reports, etc.
请提供相关测试结果的链接，例如 CI 管道、测试报告等。

## ⚠️ BC-breaking (Optional) / 向后不兼容变更（可选）

Does the modification introduce changes that break the backward compatibility of the downstream repositories? If so, please describe how it breaks the compatibility and how the downstream projects should modify their code to keep compatibility with this PR.
是否引入了会破坏下游存储库向后兼容性的更改？如果是，请描述它如何破坏兼容性，以及下游项目应该如何修改其代码以保持与此 PR 的兼容性。

## ⚠️ Performance degradation (Optional) / 性能下降（可选）

If the modification introduces performance degradation, please describe the impact of the performance degradation and the expected performance improvement.
如果引入了性能下降，请描述性能下降的影响和预期的性能改进。

## 🌟 Use cases (Optional) / 使用案例（可选）

If this PR introduces a new feature, it is better to list some use cases here and update the documentation.
如果此拉取请求引入了新功能，最好在此处列出一些用例并更新文档。

## ✅ Checklist / 检查列表

**Before PR**:

- [ ] Pre-commit or other linting tools are used to fix the potential lint issues. / 使用预提交或其他 linting 工具来修复潜在的 lint 问题。
- [ ] Bug fixes are fully covered by unit tests, the case that causes the bug should be added in the unit tests. / 修复的 Bug 已完全由单元测试覆盖，导致 Bug 的情况应在单元测试中添加。
- [ ] The modification is covered by complete unit tests. If not, please add more unit tests to ensure the correctness. / 此拉取请求中的修改已完全由单元测试覆盖。如果不是，请添加更多单元测试以确保正确性。
- [ ] All relevant documentation (API docs, docstrings, example tutorials) has been updated to reflect these changes. / 所有相关文档（API 文档、文档字符串、示例教程）已更新以反映这些更改。

**After PR**:

- [ ] If the modification has potential influence on downstream or other related projects, this PR should be tested with those projects. / 如果此拉取请求对下游或其他相关项目有潜在影响，应在那些项目中测试此 PR。
- [ ] CLA has been signed and all committers have signed the CLA in this PR. / CLA 已签署，且本 PR 中的所有提交者均已签署 CLA。

## 👥 Collaboration Info / 协作信息
- Suggested Reviewers / 建议审核人: @xxx
- Relevant Module Owners / 相关模块负责人: @xxx
- Other Collaboration Notes / 其他协作说明：

## 🌟 Useful CI Command / 实用的CI命令
|   Command / 命令   |   Introduction / 介绍  |
| ---- | ----- |
|`/gemini review`| Performs a code review for the current pull request in its current state by Gemini. / 对当前拉取请求在当前状态下由 Gemini 执行代码审核。 |
|`/gemini summary`| Provides a summary of the current pull request in its current state by Gemini. / 对当前拉取请求在当前状态下由 Gemini 提供摘要。 |
|`/gemini help`| Displays a list of available commands of Gemini. / 显示 Gemini 可用命令的列表。 |
|`/readthedocs build`| Triggers a build of the documentation for the current pull request in its current state by Read the Docs. / 触发当前拉取请求在当前状态下由 Read the Docs 构建文档。 |
|`/pr_check`| 手动重新触发 PR 合入质量检查工作流（PR Quality Check），并在 PR 上评论即可生效。 / Manually re-runs the PR Quality Check workflow by commenting on the pull request. |
