---
name: "model-researcher"
description: "快速检索新出大模型的官方发布信息，包括Hugging Face主页、GitHub主页、论文、技术报告和博客等。当用户需要了解某个新模型的全面官方信息时调用。"
---

# 模型研究助手 (Model Researcher)

## 功能

该技能可以帮助用户快速、全面地检索新出大模型的官方发布信息，包括：

- Hugging Face 模型主页
- GitHub 代码仓库
- 相关论文（arXiv等）
- 技术报告
- 官方博客文章
- 其他官方发布的资源

## 使用方法

当用户需要了解某个新发布的大模型的官方信息时，调用此技能。例如：

- "检索最新发布的GPT-4o模型的官方信息"
- "查找Meta的Llama 3模型的Hugging Face页面和论文"
- "搜索Anthropic的Claude 3模型的GitHub仓库和技术报告"

## 工作流程

1. **收集模型名称**：确认用户要查询的具体模型名称
2. **多源搜索**：
   - 在Hugging Face上搜索模型页面
   - 在GitHub上搜索官方代码仓库
   - 在arXiv和其他学术平台上搜索相关论文
   - 在官方网站和博客上搜索技术报告和发布信息
3. **信息整合**：将所有找到的官方信息整理成结构化的结果
4. **结果呈现**：以清晰的格式展示所有检索到的信息

## 示例

### 输入输出示例

**输入**：
"检索Google Gemini 1.5模型的官方信息"

**输出**：
```
# Google Gemini 1.5 官方信息

## Hugging Face 主页
- https://huggingface.co/google/gemini-1.5-flash

## GitHub 仓库
- https://github.com/google/gemini-api

## 论文
- Gemini 1.5: Unlocking multimodal understanding across millions of tokens
  https://arxiv.org/abs/2403.05530

## 技术报告
- Google AI Blog: Introducing Gemini 1.5
  https://blog.google/technology/ai/google-gemini-next-generation-model/

## 其他资源
- 官方文档: https://ai.google.dev/gemini-api/docs
```

## 注意事项

- 该技能专注于官方发布的信息，确保信息的准确性和权威性
- 对于较新的模型，某些资源可能尚未发布，会明确标注
- 搜索结果会按照相关性和可靠性排序，优先展示官方来源
