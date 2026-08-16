# 评估：将datasets.md移动到get_started目录下的合理性

## 结论：建议合理

### 理由分析

1. **文档内容匹配度**
   - `datasets.md` 的内容是"Dataset Preparation Guide"（数据集准备指南）
   - 这是一个入门性质的文档，帮助用户了解如何准备数据集
   - 与`get_started/install.md`（安装指南）和`get_started/quick_start.md`（快速入门）在性质上相似，都属于入门级别的文档

2. **文档结构清晰度**
   - 将"数据集准备"放在`get_started/`目录下，更符合用户的认知路径：安装 → 快速入门 → 数据集准备 → 运行评测
   - 避免与`base_tutorials/all_params/`下的其他技术参数文档（如`models.md`、`mode.md`）混淆

3. **用户体验**
   - 用户在开始使用AISBench时，更容易在Get Started部分找到数据集准备的相关信息
   - 与当前在index.rst的Get Started部分突出显示datasets的修改意图一致

### 需要修改的内容

1. **移动文件**
   - 将 `docs/source_en/get_started/datasets.md` 移动到 `docs/source_en/get_started/datasets.md`
   - 将 `docs/source_zh_cn/get_started/datasets.md` 移动到 `docs/source_zh_cn/get_started/datasets.md`

2. **更新引用路径**
   - 更新 `docs/source_en/index.rst` 中的引用路径：从 `base_tutorials/all_params/datasets` 改为 `get_started/datasets`
   - 更新 `docs/source_zh_cn/index.rst` 中的引用路径

3. **更新文档内链接**（如果存在内部链接）
   - 检查datasets.md中是否有指向其他文档的相对链接，如果有需要更新

### 注意事项

- 需要同时修改英文和中文文档，确保一致性
- 移动文件后需要重新构建文档以验证修改正确