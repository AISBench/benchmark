# OneIG-Benchmark

**OneIG-Benchmark** 是一个面向文生图模型的综合评测基准，围绕**对齐性**、**文本渲染**、**推理能力**、**风格表现**、**多样性**五个维度组织评测，覆盖图像生成模型的核心能力评估。官方 Standard 套件包含 **5 个子任务**，支持 **EN（英文）** 和 **ZH（中文）** 两种语言模式。

AISBench **已适配 OneIG-Benchmark**。仓库目录 `ais_bench/configs/oneig_examples/` 下放的是 **独立配置文件** 示例，在 **GPU** 上对**生成图片**做多维度质量测评。OneIG 采用 **eval-only** 模式，不包含图片生成步骤，请先使用待评测模型生成图片后再进行测评。

## 数据集概述

### 背景简介

OneIG-Benchmark 由官方团队开发，旨在从多个细粒度维度全面评估文生图模型的生成质量。官方 GitHub 仓库：[https://github.com/OneIG-Bench/OneIG-Benchmark](https://github.com/OneIG-Bench/OneIG-Benchmark)，数据集地址：[https://huggingface.co/datasets/OneIG-Bench/OneIG-Benchmark](https://huggingface.co/datasets/OneIG-Bench/OneIG-Benchmark)。

### 核心特性

| 特性 | 说明 |
| --- | --- |
| **5 维评测** | 覆盖对齐、文本、推理、风格、多样性五个维度 |
| **双语言支持** | EN（英文）/ ZH（中文）两种模式 |
| **LLM-as-Judge** | Alignment 和 Text 任务使用多模态大模型作为裁判 |
| **ML 模型评测** | Reasoning、Style、Diversity 使用专业小模型评测 |
| **Eval-Only 模式** | 仅评测生成图片，不包含图片生成步骤 |
| **网格切分** | 支持将多图拼接的网格图片自动切分为子图 |
| **精度对齐** | 与官方评测方法精度差异 < 1% |

### 评测架构总览

端到端评测流程分为数据准备和评估两个阶段：

```
数据准备阶段                           评估阶段
┌──────────────────────┐    ┌───────────────────────────────────────────┐
│ OneIG-Bench.csv      │    │               评估阶段                    │
│ (原始数据集)          │    │                                           │
│       ↓              │    │  images/ 目录 (待评测图片)                │
│ prompt 提取           │    │       ↓                                   │
│       ↓              │    │  ┌─────────────┐  ┌─────────────┐        │
│ 文生图模型生成图片     │    │  │ Alignment   │  │ Text        │        │
│       ↓              │    │  │ (LLM-Judge) │  │ (LLM-Judge) │        │
│ images/ 目录          │───▶│  └─────────────┘  └─────────────┘        │
│ (待评测对象)          │    │  ┌─────────────┐  ┌─────────────┐        │
└──────────────────────┘    │  │ Reasoning   │  │ Style       │        │
                            │  │ (LLM2CLIP)  │  │ (CSD+SE)    │        │
                            │  └─────────────┘  └─────────────┘        │
                            │  ┌─────────────┐                         │
                            │  │ Diversity   │  → results/ 目录        │
                            │  │ (DreamSim)  │    (评测结果)            │
                            │  └─────────────┘                         │
                            └───────────────────────────────────────────┘
```

**AISBench 适配架构**（四层分离）：

```
ais_bench/
├── benchmark/                              # 框架层
│   ├── datasets/oneig.py                   # 数据集加载器
│   ├── tasks/oneig/                        # 评测任务包
│   │   ├── __init__.py                     # 模块入口
│   │   ├── oneig_eval.py                   # 评测任务（OneIGEvalTask）
│   │   ├── oneig_eval_utils.py             # 公共工具函数
│   │   ├── oneig_alignment_eval.py         # 对齐评估器
│   │   ├── oneig_text_eval.py              # 文本评估器
│   │   ├── oneig_reasoning_eval.py         # 推理评估器
│   │   ├── oneig_style_eval.py             # 风格评估器
│   │   └── oneig_diversity_eval.py         # 多样性评估器
│   └── summarizers/oneig.py                # 评分汇总器
├── configs/oneig_examples/                 # 用户示例配置
│   └── oneig_full_eval.py                  # 全量评测配置文件
└── docs/
    ├── source_zh_cn/extended_benchmark/lmm_generate/oneig.md   # 中文文档
    └── source_en/extended_benchmark/lmm_generate/oneig.md      # 英文文档
```

## 依赖与环境

### 基础环境

OneIG 评测仅支持 **GPU** 平台。在开始前，请确保已安装 AISBench：

```bash
# 克隆 AISBench 代码
git clone https://github.com/AISBench/benchmark.git
cd benchmark/

# 安装运行依赖
pip install -e ./ --use-pep517
```

### OneIG 官方仓库

OneIG 评测依赖官方仓库的辅助数据和参考嵌入，需提前克隆：

```bash
# 克隆 AISBench 组织下的 OneIG 代码（已修复已知 Bug）
git clone https://github.com/AISBench/OneIG-Benchmark.git
cd OneIG-Benchmark/

# 安装依赖
pip install -r requirements.txt
```

### 模型权重与资源下载

OneIG 评测涉及多种模型权重，分为以下三类：

#### 一、HuggingFace 自动下载（首次运行时自动完成，无需手动操作）

| 模型 | 用途 | HuggingFace 路径 |
| --- | --- | --- |
| Judge 模型 | Alignment / Text | `Qwen/Qwen3-VL-8B-Instruct` |
| LLM2CLIP Clip | Reasoning | `openai/clip-vit-large-patch14-336` |
| LLM2CLIP Vision | Reasoning | `microsoft/LLM2CLIP-Openai-L-14-336` |
| LLM2CLIP LLM | Reasoning | `microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned` |
| SE Encoder | Style | `xingpng/OneIG-StyleEncoder` |

#### 二、DreamSim 权重（Diversity）

首次运行时 `dreamsim` 库会自动从 GitHub Releases 下载权重到 `{ONEIG_ROOT}/models/` 目录。如果网络无法访问 GitHub，需手动下载：

- 下载地址：`https://github.com/ssundaram21/dreamsim/releases/download/v0.2.0-checkpoints/dreamsim_ensemble_checkpoint.zip`
- 解压到 `{ONEIG_ROOT}/models/` 目录
- 解压后应包含：`dino_vitb16_pretrain.pth`、`open_clip_vitb16_pretrain.pth.tar`、`clip_vitb16_pretrain.pth.tar`、`ensemble_lora/`

#### 三、必须手动下载的文件（Style 任务）

| 文件 | 归档路径 | 下载地址 |
| --- | --- | --- |
| CSD 编码器 | `{ONEIG_ROOT}/scripts/style/models/checkpoint.pth` | [Google Drive](https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view) |
| CLIP ViT-L-14 | `{ONEIG_ROOT}/scripts/style/models/ViT-L-14.pt` | [OpenAI Public](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) |

#### 四、随 OneIG 仓库分发的数据文件（`git clone` 后自动获取）

| 文件 | 路径 | 用途 |
| --- | --- | --- |
| 问题依赖数据 | `scripts/alignment/Q_D/*.json` | Alignment 任务的问答依赖关系 |
| 文本内容数据 | `scripts/text/text_content*.csv` | Text 任务的参考文本 |
| 参考答案 | `scripts/reasoning/gt_answer*.json` | Reasoning 任务的参考答案 |
| 风格标签 | `scripts/style/style.csv` | Style 任务的风格标签 |
| CSD 参考嵌入 | `scripts/style/CSD_embed.pt` | Style 任务的 CSD 参考向量 |
| SE 参考嵌入 | `scripts/style/SE_embed.pt` | Style 任务的 SE 参考向量 |

## 快速开始

### 配置修改

编辑配置文件 `ais_bench/configs/oneig_examples/oneig_full_eval.py`，修改以下关键参数：

```python
# OneIG 官方项目绝对路径（需提前克隆）
ONEIG_ROOT = "/path/to/OneIG-Benchmark"

# 语言模式：EN（英文）或 ZH（中文）
MODE = "EN"

# 图片根目录（生成图片存放位置）
IMAGE_DIR = "/path/to/oneig/images"

# 模型名称列表（生成图片的模型名称）
MODEL_NAMES = ["Qwen-Image"]

# 网格配置列表（与 MODEL_NAMES 一一对应，格式：'rows,cols'）
IMAGE_GRIDS = ["2,2"]

# 要执行的任务列表（可自由组合）
TASKS = ['alignment', 'text', 'reasoning', 'style', 'diversity']
```

### 执行评测

```bash
# 全量评测（5 个子任务）
ais_bench ais_bench/configs/oneig_examples/oneig_full_eval.py -m eval
```

### 结果查看

评测完成后，结果输出在 `outputs/default/{timestamp}/` 目录下：

```
outputs/default/{timestamp}/
├── configs/
│   └── {timestamp}.py                    # 评测配置快照
├── logs/
│   └── eval/
│       └── oneig_eval/
│           ├── oneig_alignment.out       # 各任务日志
│           ├── oneig_text.out
│           ├── oneig_reasoning.out
│           ├── oneig_style.out
│           └── oneig_diversity.out
├── results/
│   └── oneig_eval/
│       ├── oneig_alignment.json          # 各任务评测结果（含逐样本详情）
│       ├── oneig_text.json
│       ├── oneig_reasoning.json
│       ├── oneig_style.json
│       └── oneig_diversity.json
└── summary/
    ├── summary_{timestamp}.csv           # 评测总览
    ├── summary_{timestamp}.md
    └── summary_{timestamp}.txt
```

## 配置与输出

### 常用配置项

| 配置项 | 作用 | 必填 |
| --- | --- | --- |
| `ONEIG_ROOT` | OneIG 官方项目绝对路径 | 是 |
| `MODE` | 语言模式：`EN`（英文）或 `ZH`（中文） | 是 |
| `IMAGE_DIR` | 待评测图片根目录 | 是 |
| `MODEL_NAMES` | 生成图片的模型名称列表 | 是 |
| `IMAGE_GRIDS` | 网格配置列表，格式 `'rows,cols'`，与 `MODEL_NAMES` 一一对应 | 是 |
| `TASKS` | 要执行的任务列表，可选值：`alignment`、`text`、`reasoning`、`style`、`diversity` | 是 |
| `JUDGE_MODEL_PATH` | Judge 模型路径（Alignment/Text），默认 `Qwen/Qwen3-VL-8B-Instruct` | 否 |
| `JUDGE_SEED` | Judge 模型随机种子，默认 `42` | 否 |
| `DREAMSIM_CACHE_DIR` | DreamSim 权重缓存目录，默认 `{ONEIG_ROOT}/models` | 否 |

### 预设配置一览

| 配置名 | 说明 | 配置文件 |
| --- | --- | --- |
| oneig_full_eval | 全量评测配置，包含 5 个子任务，支持自由组合 | `ais_bench/configs/oneig_examples/oneig_full_eval.py` |

### 测评结果路径

按子任务写入：

```
{work_dir}/results/oneig_eval/oneig_{task}.json
```

其中 `{task}` 为 `alignment`、`text`、`reasoning`、`style`、`diversity` 之一。

### 输出格式说明

每个子任务的 JSON 结果文件结构如下（以 Alignment 为例）：

```json
{
    "accuracy": 88.44,
    "details": [
        {
            "id": "000",
            "class_item": "anime",
            "score": 0.85,
            "image_path": "/path/to/image.png",
            "grid": "2x2",
            "num_splits": 4,
            "judge_details": [
                {
                    "question_id": "Q1",
                    "question": "...",
                    "judge_prompt": "...",
                    "judge_outputs": [
                        {"grid_idx": 0, "raw_output": "Yes", "parsed_answer": "Yes", "score": 1.0}
                    ],
                    "dependency": [0],
                    "filtered_scores": null
                }
            ]
        }
    ],
    "style_scores": null
}
```

各子任务的 `details` 字段包含不同的中间数据：

| 子任务 | 中间数据字段 | 说明 |
| --- | --- | --- |
| Alignment | `judge_details` | 逐切分图的 Judge 问答详情 |
| Text | `ocr_details` | 逐切分图的 OCR 结果与文本指标（ED/CR/WAC） |
| Reasoning | `similarity_details` | 逐切分图的相似度得分 |
| Style | `encoder_details` | 逐切分图的 CSD/SE 相似度与风格得分 |
| Diversity | `pairwise_distances` | 逐对切分图的 DreamSim 距离 |

## 评测指标体系

### 指标总览

| 子任务 | 主指标 | 辅助指标 | 评测方式 | 评测模型 |
| --- | --- | --- | --- | --- |
| Alignment | `accuracy` | - | LLM-as-Judge | Qwen3-VL-8B-Instruct |
| Text | `accuracy` | `ED`、`CR`、`WAC` | LLM-as-Judge + OCR | Qwen3-VL-8B-Instruct |
| Reasoning | `accuracy` | - | 特征相似度 | LLM2CLIP |
| Style | `accuracy` | - | 特征相似度 | CSD + SE Encoder |
| Diversity | `accuracy` | `oneig_diversity_{class}` | 感知距离 | DreamSim |
| **Total** | `oneig_total` | - | 5 任务平均 | - |

### 各子任务评测逻辑

#### Alignment（对齐评估 — LLM-as-Judge）

**目标**：评估生成图片与提示词的对齐程度。

**流程**：
1. 将网格图片切分为子图
2. 对每个子图，使用 Judge 模型（Qwen3-VL-8B-Instruct）回答 Yes/No 问题
3. 答案为 "Yes" 记 1 分，"No" 记 0 分
4. 取所有子图的平均分作为该样本的得分
5. 全部样本得分取平均 × 100 作为 accuracy

**关键参数**：
- `judge_model_path`：Judge 模型路径
- `judge_seed`：随机种子（默认 42，确保可复现）
- `num_gpus`：支持多 GPU 并行（推荐 4）

#### Text（文本评估 — LLM-as-Judge + OCR）

**目标**：评估生成图片中文本渲染的准确性。

**流程**：
1. 将网格图片切分为子图
2. 使用 Judge 模型对每个子图进行 OCR，提取文本
3. 将提取文本与参考文本对比，计算三个指标：
   - **ED**（Edit Distance）：编辑距离
   - **CR**（Character Ratio）：字符比率
   - **WAC**（Word Accuracy Coincidence）：词准确率
4. 综合 OCR 指标与 Judge 评分得到 accuracy

#### Reasoning（推理评估 — LLM2CLIP）

**目标**：评估生成图片对推理类提示词的理解程度。

**流程**：
1. 将网格图片切分为子图
2. 使用 LLM2CLIP 提取图片特征和参考答案文本特征
3. 计算图片特征与文本特征的余弦相似度
4. 取所有子图的平均相似度作为该样本得分
5. 全部样本得分取平均 × 100 作为 accuracy

**模型组成**：
- CLIP Processor：`openai/clip-vit-large-patch14-336`
- CLIP Model：`microsoft/LLM2CLIP-Openai-L-14-336`
- LLM Model：`microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned`

#### Style（风格评估 — CSD + SE Encoder）

**目标**：评估生成图片的风格表现。

**流程**：
1. 将网格图片切分为子图
2. 使用 CSD（CLIP-Style-Diffusion）编码器提取风格特征
3. 使用 SE（Style Encoder）编码器提取风格特征
4. 分别计算 CSD 特征和 SE 特征与参考风格嵌入的余弦相似度
5. 取两个相似度的平均值作为该子图的风格得分
6. 所有子图取平均，全部样本取平均 × 100 作为 accuracy

**风格类别**（29 种）：abstract_expressionism、art_nouveau、baroque、chinese_ink_painting、cubism、fauvism、impressionism、line_art、minimalism、pointillism、pop_art、rococo、ukiyo-e、clay、crayon、graffiti、lego、comic、pencil_sketch、stone_sculpture、watercolor、celluloid、chibi、cyberpunk、ghibli、impasto、pixar、pixel_art、3d_rendering

#### Diversity（多样性评估 — DreamSim）

**目标**：评估同一模型生成的多张图片之间的多样性。

**流程**：
1. 将网格图片切分为子图
2. 使用 DreamSim 模型计算所有子图两两之间的感知距离
3. 取所有距离对的平均值作为该样本的多样性得分
4. 按 class_item（anime、human、object、text、reasoning）分组统计细粒度指标
5. 全部样本得分取平均 × 100 作为 accuracy

### 评分汇总（oneig_total）

`oneig_total` 为 5 个子任务 accuracy 的简单平均：

```
oneig_total = (alignment + text + reasoning + style + diversity) / 5
```

此外，Diversity 任务额外输出按 class_item 分组的细粒度指标：

| 指标 | 说明 |
| --- | --- |
| `oneig_diversity_anime` | Anime 类别的多样性得分 |
| `oneig_diversity_human` | Portrait 类别的多样性得分 |
| `oneig_diversity_object` | General Object 类别的多样性得分 |
| `oneig_diversity_text` | Text Rendering 类别的多样性得分 |
| `oneig_diversity_reasoning` | Knowledge Reasoning 类别的多样性得分 |

### 评测结果示例

```
dataset             version  metric   mode  oneig_eval
oneig_alignment     a39421   accuracy gen   88.44
oneig_text          a39421   accuracy gen   80.79
oneig_text          a39421   ED        gen   43.32
oneig_text          a39421   CR        gen   0.08
oneig_text          a39421   WAC       gen   0.52
oneig_reasoning     a39421   accuracy gen   29.84
oneig_style         a39421   accuracy gen   35.85
oneig_diversity     a39421   accuracy gen   18.28
oneig_total         -        accuracy gen   50.64
oneig_diversity_anime  -     accuracy gen   9.00
oneig_diversity_human  -     accuracy gen   11.21
oneig_diversity_object -    accuracy gen   13.27
oneig_diversity_text   -     accuracy gen   21.14
oneig_diversity_reasoning - accuracy gen   36.80
```

## 数据格式说明

### 原始数据集格式

OneIG 原始数据集为 CSV 文件（`OneIG-Bench.csv`），每条数据包含以下字段：

```json
{
    "category": "Anime_Stylization",
    "id": "000",
    "prompt_en": "4boys, 5girls, multiple boys, multiple girls, ...",
    "type": "T, P",
    "prompt_length": "long",
    "class": "None"
}
```

| 字段 | 说明 |
| --- | --- |
| `category` | 提示词类别：Anime_Stylization、Portrait、General Object、Text Rendering、Knowledge Reasoning、Multilingualism |
| `id` | 唯一 ID，每个类别独立维护 |
| `prompt_en` | 文生图提示词 |
| `type` | 类型标记：T（Text）、P（Portrait）、NP（Non-Portrait） |
| `prompt_length` | 提示词长度：short、middle、long |
| `class` | 风格类别（可选）：fauvism、watercolor、None |

### 图片目录结构

待评测图片需按以下目录结构组织：

```
IMAGE_DIR/
├── anime/                      # class_item 目录
│   └── {model_name}/           # 模型名称目录
│       ├── 000.png             # 图片文件（文件名前3位为 sample_id）
│       ├── 001.png
│       └── ...
├── human/
│   └── {model_name}/
│       ├── 000.png
│       └── ...
├── object/
│   └── {model_name}/
│       └── ...
├── text/
│   └── {model_name}/
│       └── ...
└── reasoning/
    └── {model_name}/
        └── ...
```

各子任务对应的 class_item 目录：

| 子任务 | EN 模式 | ZH 模式（额外） |
| --- | --- | --- |
| Alignment | anime、human、object | multilingualism |
| Text | text | - |
| Reasoning | reasoning | - |
| Style | anime | - |
| Diversity | anime、human、object、text、reasoning | multilingualism |

### 网格切分说明

OneIG 支持将多张生成图片拼接为网格图进行批量评测。`IMAGE_GRIDS` 配置指定了网格的行列数：

| 网格配置 | 含义 | 切分子图数 |
| --- | --- | --- |
| `"1,2"` | 1 行 2 列 | 2 |
| `"2,2"` | 2 行 2 列 | 4 |
| `"1,4"` | 1 行 4 列 | 4 |
| `"3,3"` | 3 行 3 列 | 9 |

评测时，网格图片会被自动切分为子图，每个子图独立评测后取平均分。

## 示例代码

### 单任务评测

修改配置文件中的 `TASKS` 列表，仅包含需要评测的任务：

```python
# 仅评测 Alignment
TASKS = ['alignment']
```

执行：

```bash
ais_bench ais_bench/configs/oneig_examples/oneig_full_eval.py -m eval
```

### 全量评测

```python
# 评测全部 5 个子任务
TASKS = ['alignment', 'text', 'reasoning', 'style', 'diversity']
```

执行：

```bash
ais_bench ais_bench/configs/oneig_examples/oneig_full_eval.py -m eval
```

### 中文模式评测

```python
ONEIG_ROOT = "/path/to/OneIG-Benchmark"
MODE = "ZH"                                    # 切换为中文模式
IMAGE_DIR = "/path/to/oneig/images_zh"         # 中文提示词生成的图片目录
MODEL_NAMES = ["Qwen-Image"]
IMAGE_GRIDS = ["2,2"]
TASKS = ['alignment', 'text', 'reasoning', 'style', 'diversity']
```

### 多模型对比评测

```python
MODEL_NAMES = ["model_a", "model_b"]
IMAGE_GRIDS = ["2,2", "2,2"]                   # 与 MODEL_NAMES 长度一致
```

