# 修改文档结构，使datasets内容在Get Started部分更突出

## 目标
将datasets内容添加到Get Started部分，使其更突出，同时保持文档的完整性和一致性。

## 实施步骤

### 1. 修改英文文档 (`docs/source_en/index.rst`)
- 在Get Started部分添加datasets条目，指向`base_tutorials/all_params/datasets`
- 更新推荐上手路径部分，将datasets添加到学习顺序中

### 2. 修改中文文档 (`docs/source_zh_cn/index.rst`)
- 在"开始你的第一步"部分添加datasets条目，指向`base_tutorials/all_params/datasets`
- 更新推荐上手路径部分，将datasets添加到学习顺序中

### 3. 保持文档完整性
- 保留`base_tutorials/all_params/index.rst`中的datasets条目，确保用户可以从详细参数说明部分访问到datasets内容
- 确保引用路径正确，使用相对路径指向datasets.md文件

## 预期效果
- datasets内容在Get Started部分突出显示，方便用户快速找到
- 保持文档结构的完整性和一致性
- 英文和中文文档同步更新，确保用户体验一致