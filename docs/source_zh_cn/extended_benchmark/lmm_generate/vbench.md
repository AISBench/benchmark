# VBench 1.0

**VBench**（VBench: Comprehensive Benchmark Suite for Video Generative Models）是面向视频生成模型的评测基准套件，围绕主体一致性、运动平滑度、时序闪烁、空间关系等与感知相关的指标组织评测（官方 Standard 套件为 **16 个维度**），并提供与各维度匹配的 prompt、流程与校验方式。

AISBench **已适配 VBench 1.0**。仓库目录 `ais_bench/configs/vbench_examples/` 下放的是 **独立配置文件** 示例：在 **GPU** 或 **NPU** 上对已有生成视频做质量/语义类维度测评。**当前 aisbench 不包含多模态视频生成**，请完成视频生成后再跑测评。

## 目录

- [快速开始](#快速开始)
- [运行模式 `-m eval` 说明与限制](#运行模式--m-eval-说明与限制)
- [配置与输出](#配置与输出)
- [评分汇总（Quality / Semantic / Total）](#评分汇总quality--semantic--total)
- [依赖与环境](#依赖与环境)
- [Prompt Suite（官方 prompt 结构）](#prompt-suite官方-prompt-结构)
- [数据集生成](#数据集生成)
- [采样伪代码（参考官方）](#采样伪代码参考官方)
- [格式要求](#格式要求)

## 快速开始

1. **准备视频目录**
   Standard / Custom 均需在对应配置中将 `DATA_PATH` 设为生成视频的根目录（绝对或相对路径）。也可复制配置文件后改 `DATA_PATH`，再执行 `ais_bench <your_config.py> --mode eval`。（视频采样说明参考：[数据集生成](#数据集生成)）

2. **（强烈推荐）预热本地缓存**
   VBench 会加载多种小模型权重；未缓存时可能长时间在线下载甚至失败。**一键下载脚本只读取 shell 里的 `VBENCH_CACHE_DIR`**，与配置文件顶层变量无关——若要和测评共用同一目录，请先在终端 `export VBENCH_CACHE_DIR=...`。细节见下文 [依赖与环境](#依赖与环境) 与 [vbench_cache_dependencies.md](vbench_cache_dependencies.md)。

```bash
# 使用默认缓存目录 ~/.cache/vbench
bash ais_bench/configs/vbench_examples/download_vbench_cache.sh

# 或指定自定义缓存目录
VBENCH_CACHE_DIR=/your/custom/cache/dir \
  bash ais_bench/configs/vbench_examples/download_vbench_cache.sh
```

3. **可选：在配置里指定缓存**
   在示例配置顶层设置 `VBENCH_CACHE_DIR = "/path/to/cache"`（或别名 `vbench_cache_dir`）可在测评子进程导入 vbench **之前**覆盖该进程环境变量；未设置时沿用 shell 中的 `export`。

4. **执行测评（推荐显式 `--mode eval`）**

```bash
# Standard（16 维，官方 Prompt Suite）
ais_bench ais_bench/configs/vbench_examples/eval_vbench_standard.py --mode eval

# Custom（10 维，自定义 prompt）
ais_bench ais_bench/configs/vbench_examples/eval_vbench_custom.py --mode eval
```

（`-m eval` 与 `--mode eval` 等价，与同目录示例配置顶部注释写法一致。）

## 配置与输出

### 常用配置项

| 配置项 / 环境变量 | 作用 |
| --- | --- |
| `DATA_PATH` | 待测评视频根目录（**必填**），见 `ais_bench/configs/vbench_examples/eval_vbench_standard.py`、`ais_bench/configs/vbench_examples/eval_vbench_custom.py` |
| `VBENCH_CACHE_DIR`（环境变量或配置顶层） | 小模型与权重缓存根目录；默认 `~/.cache/vbench` |

### 预设配置一览

| 配置名 | 说明 | 配置文件 |
| --- | --- | --- |
| eval_vbench_standard | 标准 prompt 测评，16 维；需提供视频目录与（可按需指定的）full_info json | `ais_bench/configs/vbench_examples/eval_vbench_standard.py` |
| eval_vbench_custom | 自定义输入（prompt 来自文件或文件名），10 维 | `ais_bench/configs/vbench_examples/eval_vbench_custom.py` |

### 测评结果路径

按维度写入：

```
{work_dir}/results/vbench_eval/vbench_<dim>.json
```

Standard 使用 `VBenchSummarizer` 聚合 Quality、Semantic、Total；Custom 使用 `DefaultSummarizer` 输出各维分数。实现与官方 `cal_final_score.py` 思路一致，见 `ais_bench/benchmark/summarizers/vbench.py`。

## 评分汇总（Quality / Semantic / Total）

### 单维度归一与加权

对每个维度：将原始准确率按 `NORMALIZE_DIC` 中该维度的 **Min、Max** 线性缩放为 \((raw - Min)/(Max - Min)\)，再乘以 **DIM_WEIGHT**。若 \(Max - Min \le 0\)，实现中会退回边界取值。常量与官方对齐，均在 `vbench.py`。**`dynamic_degree` 的 DIM_WEIGHT 为 0.5**，其余参与聚合的维度为 **1**。

### Quality 组（视频生成质量）

对 Quality 组各维度的「单维度加权得分」求和后，除以对应的 **DIM_WEIGHT 之和**（加权平均）。包含：

`subject_consistency`，`background_consistency`，`temporal_flickering`，`motion_smoothness`，`aesthetic_quality`，`imaging_quality`，`dynamic_degree`。

### Semantic 组（内容与语义一致性）

同理；该组各维 DIM_WEIGHT 均为 **1**。包含：

`object_class`，`multiple_objects`，`human_action`，`color`，`spatial_relationship`，`scene`，`appearance_style`，`temporal_style`，`overall_consistency`。

### Total（整体分数）

`Total = (Quality × 4 + Semantic × 1) / 5`（对应代码中 `QUALITY_WEIGHT = 4`、`SEMANTIC_WEIGHT = 1`）。

### 缺失维度与输出目录

某维未出现在结果中时，聚合按 **0** 处理（与 `normalized.get(k, 0)` 一致）。

默认 `work_dir` 为 `outputs/default`，可通过 `--work_dir` 修改。

## 依赖与环境

- 评测逻辑封装在 **`ais_bench/third_party/vbench`**（VBench 1.0 接口）。

#### decord（视频解码）

在 **x86_64** 上一般可直接 `pip install decord`。在 **ARM** 等无预编译 wheel 的环境需源码编译，例如：

```bash
git clone https://github.com/dmlc/decord
cd decord
mkdir build && cd build
cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
make
cd ../python
python3 setup.py install --user
```

#### detectron2 与 GRiT

`object_class`、`multiple_objects`、`color`、`spatial_relationship` 等维依赖 GRiT，继而依赖 detectron2。AISBench **统一**使用仓库内 **`ais_bench/third_party/detectron2`**（GPU/NPU 共用）。在仓库根目录执行可编辑安装：

`pip install -e ais_bench/third_party/detectron2 --no-build-isolation`

#### 昇腾上的 torchvision（可选）

部分 torchvision 算子（如 nms、`roi_align`）在昇腾上可能仅 CPU，测评效率低。若 `torch < 2.7.1`，可参考 [昇腾适配 torchvision](https://gitcode.com/Ascend/vision) 安装对应版本以提速。

#### 各维模型权重

各维度所需模型与权重请按 VBench 官方说明自备。

#### 小模型依赖与本地缓存

CLIP、DINO、MUSIQ、RAFT、ViCLIP、GRiT、Tag2Text、BERT base 等小模型默认落在环境变量 **`VBENCH_CACHE_DIR`**（未设置则用 `~/.cache/vbench`）。

- 配置顶层可设 `VBENCH_CACHE_DIR`，在子进程导入 vbench 前生效。
- **`download_vbench_cache.sh` 只认当前 shell 的 `export`**；要与测评路径一致时请事先导出同名目录。完整清单与手动下载步骤见 [vbench_cache_dependencies.md](vbench_cache_dependencies.md)。
- 脚本会跳过已有文件，可重复执行。
- 若脚本失败或需部分手动下载，请查阅 [vbench_cache_dependencies.md](vbench_cache_dependencies.md)。

## Prompt Suite（官方 prompt 结构）

路径相对于 `ais_bench/third_party/vbench/`：

| 路径 | 说明 |
| --- | --- |
| `prompts/prompts_per_dimension/` | 各测评维度对应的 prompt 文件（约 100 条/维度） |
| `prompts/all_dimension.txt` | 全维度合并列表 |
| `prompts/prompts_per_category/` | 8 类：Animal, Architecture, Food, Human, Lifestyle, Plant, Scenery, Vehicles |
| `prompts/all_category.txt` | 全类别合并 |
| `prompts/metadata/` | `color`、`object_class` 等需语义解析的 metadata |

### 维度与 Prompt Suite 映射（Standard，16 维）

以下为 Standard 模式下 **全部 16 个维度** 与官方 Prompt Suite 文件及条目数的对应关系；评测时由 `VBench_full_info.json` 自动匹配 prompt：

| Dimension | Prompt Suite | Prompt Count |
| :---: | :---: | :---: |
| subject_consistency | subject_consistency | 72 |
| background_consistency | scene | 86 |
| temporal_flickering | temporal_flickering | 75 |
| motion_smoothness | subject_consistency | 72 |
| dynamic_degree | subject_consistency | 72 |
| aesthetic_quality | overall_consistency | 93 |
| imaging_quality | overall_consistency | 93 |
| object_class | object_class | 79 |
| multiple_objects | multiple_objects | 82 |
| human_action | human_action | 100 |
| color | color | 85 |
| spatial_relationship | spatial_relationship | 84 |
| scene | scene | 86 |
| temporal_style | temporal_style | 100 |
| appearance_style | appearance_style | 90 |
| overall_consistency | overall_consistency | 93 |

## 数据集生成

本节面向 **需要按官方方式自行生成评测视频** 的用户（与「仅用已有目录跑测评」的快速开始互不冲突）。

### Standard 数据集（eval_vbench_standard）

- **数据来源**：`ais_bench/third_party/vbench/prompts/` 下的 Prompt Suite。
- **元数据**：需要 `VBench_full_info.json`（默认同上第三方目录中的文件）。
- **采样规模**：每条 prompt 一般 **5** 段视频；**`temporal_flickering`** 需 **25** 段，以便 static filter 后仍有足够样本。
- **随机种子**：建议每条视频不同 seed（如 `index` 或 `seed+index`），兼顾多样性与复现。
- **目录形状**：支持扁平目录或按维度子目录。若你使用「按维度子目录」结构，建议采用与评测读取侧一致的子目录命名（见下）。映射逻辑在 `ais_bench/third_party/vbench/__init__.py` 的 `dim_to_subdir`。

#### Standard 模式DATA_PATH目录格式

```
DATA_PATH/
|-- subject_consistency/        # 同名维度
|-- scene/                      # background_consistency 会优先映射到这里
|-- overall_consistency/        # aesthetic_quality / imaging_quality 会优先映射到这里
|-- object_class/
|-- multiple_objects/
|-- color/
|-- spatial_relationship/
|-- temporal_style/
|-- human_action/
|-- temporal_flickering/
`-- appearance_style/
```

其中这些维度在评测时会优先使用映射后的子目录（若该目录存在）：

- `background_consistency` → `scene/`
- `aesthetic_quality` → `overall_consistency/`
- `imaging_quality` → `overall_consistency/`
- `motion_smoothness` → `subject_consistency/`
- `dynamic_degree` → `subject_consistency/`

文件名推荐为 `{prompt}-{i}.mp4`。若你为 `temporal_flickering` 生成了 `0~24`（用于 static filter），请至少保证 `0~4` 存在；评测侧在构建临时 `full_info` 时会默认查找 `0~4`。

### Custom 数据集（eval_vbench_custom）

- **数据来源**：自定义 prompt 列表或 prompt 文件。
- **维度**：含 `subject_consistency`，`background_consistency`，`aesthetic_quality`，`imaging_quality`，`temporal_style`，`overall_consistency`，`human_action`，`temporal_flickering`，`motion_smoothness`，`dynamic_degree`。**不含**需提供 `auxiliary_info` 的维（如 `object_class`、`color`、`spatial_relationship`）。

## 采样伪代码（参考官方）

下面伪代码与 Standard 的读取侧逻辑对齐：**按维度循环**、将视频保存到对应子目录，并对 `temporal_flickering` 使用更大的采样数。

```python
import os

# 评测读取侧的目录映射（见 ais_bench/third_party/vbench/__init__.py）
dim_to_subdir = {
    "background_consistency": "scene",
    "aesthetic_quality": "overall_consistency",
    "imaging_quality": "overall_consistency",
    "motion_smoothness": "subject_consistency",
    "dynamic_degree": "subject_consistency",
}

dimension_list = [
    "subject_consistency", "background_consistency", "aesthetic_quality", "imaging_quality",
    "object_class", "multiple_objects", "color", "spatial_relationship", "scene",
    "temporal_style", "overall_consistency", "human_action", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "appearance_style",
]

if args.seed is not None:
    torch.manual_seed(args.seed)

for dimension in dimension_list:
    prompt_file = f"ais_bench/third_party/vbench/prompts/prompts_per_dimension/{dimension}.txt"
    with open(prompt_file, "r") as f:
        prompt_list = [line.strip() for line in f if line.strip()]

    n = 25 if dimension == "temporal_flickering" else 5

    subdir = dim_to_subdir.get(dimension, dimension)
    save_dir = os.path.join(args.save_path, subdir)
    os.makedirs(save_dir, exist_ok=True)

    for prompt in prompt_list:
        for index in range(n):
            video = sample_func(prompt, index)
            save_path = os.path.join(save_dir, f"{prompt}-{index}.mp4")
            torchvision.io.write_video(save_path, video, fps=8)
```

`sample_func` 表示将你的生成模型接到 `prompt` 上并得到视频的函数。

## 格式要求

### Standard 模式

- **文件名**：`{prompt}-{i}.mp4`，其中 `{prompt}` 为 `VBench_full_info.json` 中的 `prompt_en`，`i` 为 0～4。
- **扩展名**：`.mp4`，`.gif`，`.jpg`，`.png`。

### Custom 模式（更宽松）

- **方式一**：文件名携带 prompt：`get_prompt_from_filename` 从 `{xxx}.mp4` 或 `{xxx}-0.mp4` 解析出 `xxx`。
- **方式二**：提供 `prompt_file`（JSON：`{video_path: prompt}`），可不遵守文件名约定。
