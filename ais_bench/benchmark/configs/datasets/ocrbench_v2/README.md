# OCRBench_v2
中文 | [English](README_en.md)

## 数据集简介
OCRBench_v2 是一个大规模双语文本中心基准数据集，用于评估大型多模态模型（LMMs）在复杂 OCR 任务中的性能。该数据集包含 10,000 条经过人工验证的问答对，涵盖 31 种不同的场景（如街景、收据、公式、图表等），通过 23 个任务全面评估模型在文本识别、文本定位、手写内容提取和逻辑推理等方面的能力。

数据集从 81 个学术数据集中手动筛选数据，并补充私有数据以确保场景的多样性，旨在解决现有基准在任务多样性、上下文复杂性和规模上的不足。

> 🔗 数据集主页链接: [https://arxiv.org/abs/2501.00321](https://arxiv.org/abs/2501.00321)

## 数据集部署
- 可以从 HuggingFace 提供的链接下载数据集文件 🔗: [https://huggingface.co/datasets/QYWH/ocrbench_v2/resolve/main/OCRBench_v2.tsv?download=true](https://huggingface.co/datasets/QYWH/ocrbench_v2/resolve/main/OCRBench_v2.tsv?download=true)
- 数据集文件应为 TSV 格式（`.tsv` 文件），默认文件名为 `OCRBench_v2.tsv`
- 建议部署在 `{工具根路径}/ais_bench/datasets/ocrbench_v2/` 目录下（数据集任务中设置的默认路径），以 linux 上部署为例，具体执行步骤如下：

```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
mkdir -p ocrbench_v2
cd ocrbench_v2
wget https://huggingface.co/datasets/QYWH/ocrbench_v2/resolve/main/OCRBench_v2.tsv?download=true -O OCRBench_v2.tsv
```

- 在 `{工具根路径}/ais_bench/datasets` 目录下执行 `tree ocrbench_v2/` 查看目录结构，若目录结构如下所示，则说明数据集部署成功：
    ```
    ocrbench_v2/
    └── OCRBench_v2.tsv
    ```

⏰ **注意**：数据集运行前请先安装依赖 [ocrbench_v2.txt](../../../../../requirements/datasets/ocrbench_v2.txt)
```shell
# 需要处在最外层benchmark文件夹下，运行下列指令：
pip3 install -r requirements/datasets/ocrbench_v2.txt
```

## 可用数据集任务
| 任务名称 | 简介 | 评估指标 | Few-Shot | Prompt 格式 | 对应源码配置文件路径 |
| --- | --- | --- | --- | --- | --- |
| ocrbench_v2_gen_0_shot_chat | OCRBench_v2 数据集生成式任务，支持多模态输入（图像+文本） | 多种指标（根据任务类型） | 0-shot | 对话格式（多模态） | [ocrbench_v2_gen_0_shot_chat.py](ocrbench_v2_gen_0_shot_chat.py) |

## 支持的任务类型
OCRBench_v2 数据集涵盖以下任务类型：

### 英文任务
- **文本识别**：文本识别、细粒度文本识别、全页 OCR
- **文本检测**：文本定位、带位置的 VQA
- **文本定位**：文本定位
- **关系提取**：关键信息提取、关键信息映射
- **元素解析**：文档解析、图表解析、表格解析、公式识别
- **数学计算**：数学问答、文本计数
- **视觉文本理解**：文档分类、认知 VQA、图表问答
- **知识推理**：推理 VQA、科学问答、APP 代理、ASCII 艺术分类

### 中文任务
- **文本识别**：全页 OCR
- **关系提取**：关键信息提取、手写答案提取
- **元素解析**：文档解析、图表解析、表格解析、公式识别
- **视觉文本理解**：文档分类、认知 VQA、图表问答
- **知识推理**：推理 VQA、科学问答、APP 代理

