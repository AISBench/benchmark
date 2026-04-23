# VBench 1.0

AISBench 已适配 VBench 1.0，支持在 **GPU（cuda）** 与 **NPU** 上进行视频/图像质量维度测评。配置位于 `ais_bench/configs/vbench_examples/`，采用**独立配置文件**方式运行。

## 使用方式

### 仅测评（推荐）

在已有一批生成视频目录的前提下，仅运行 VBench 测评：

```bash
# Standard 模式（16 维度，官方 Prompt Suite）
ais_bench ais_bench/configs/vbench_examples/eval_vbench_standard.py

# Custom 模式（10 维度，自定义 prompt）
ais_bench ais_bench/configs/vbench_examples/eval_vbench_custom.py
```

> 建议：首次运行前先按下文说明准备好 VBench 本地缓存依赖，否则运行过程中可能因在线下载模型导致耗时较长或失败。

**注意**：需在配置中指定视频目录：

- 修改 [eval_vbench_standard.py](eval_vbench_standard.py) 或 [eval_vbench_custom.py](eval_vbench_custom.py) 中的 `DATA_PATH` 为生成的视频所在目录（绝对或相对路径）；或
- 复制配置文件，修改 `DATA_PATH` 后通过 `ais_bench <your_config.py>` 运行。

设备会自动检测：NPU 可用则用 NPU，否则用 CUDA。

### 设备配置

- **默认**：设备自动检测——若当前环境 NPU 可用（`torch.npu.is_available()`）则使用 NPU，否则使用 CUDA；
- **强制指定**：若需固定设备，可在配置的 `eval_cfg` 中设置 `device='cuda'` 或 `device='npu'`。

例如在 `eval_vbench_standard.py` 的 `vbench_eval_cfg` 中（不写 `device` 即自动检测）：

```python
vbench_eval_cfg = dict(
    use_vbench_task=True,
    load_ckpt_from_local=True,
    # device 不写则自动检测；可选 device='cuda' 或 device='npu' 强制指定
)
```

### 可用配置

| 配置名 | 说明 | 配置文件 |
|--------|------|----------|
| eval_vbench_standard | VBench 标准 prompt 测评，16 维度，需提供视频目录与（可选）full_info json | [eval_vbench_standard.py](eval_vbench_standard.py) |
| eval_vbench_custom | 自定义输入（prompt 来自文件或文件名），10 维度 | [eval_vbench_custom.py](eval_vbench_custom.py) |

### 结果输出

测评结果按维度写入：

```
{work_dir}/results/vbench_eval/vbench_<dim>.json
```

Standard 模式使用 `VBenchSummarizer` 聚合 Quality、Semantic、Total 分数；Custom 模式使用 `DefaultSummarizer` 输出各维度分数。三者与 VBench 官方 `cal_final_score.py` 思路一致，代码见 `ais_bench/benchmark/summarizers/vbench.py`。

#### Quality / Semantic / Total 计算逻辑

1. **单维度加权得分**：将各维度原始准确率按该维度在 `NORMALIZE_DIC` 中的 **Min、Max** 做线性缩放：\((raw - Min) / (Max - Min)\)，结果乘以该维度的 **DIM_WEIGHT**。若 \(Max - Min \le 0\)，实现中会退化为边界取值。常量 **Min、Max、DIM_WEIGHT** 与官方一致并写在上述 `vbench.py` 中；其中 **`dynamic_degree` 的 DIM_WEIGHT 为 0.5**，其余参与聚合的维度为 **1**。
2. **Quality**（视频生成质量）：对 **Quality 组** 全部维度的「单维度加权得分」求和，再除以这些维度的 **DIM_WEIGHT 之和**（即加权平均）。包含 7 项：`subject_consistency`、`background_consistency`、`temporal_flickering`、`motion_smoothness`、`aesthetic_quality`、`imaging_quality`、`dynamic_degree`。
3. **Semantic**（视频内容语义一致性）：对 **Semantic 组** 同理（该组各维度 DIM_WEIGHT 均为 1）。包含 9 项：`object_class`、`multiple_objects`、`human_action`、`color`、`spatial_relationship`、`scene`、`appearance_style`、`temporal_style`、`overall_consistency`。
4. **Total**（视频整体质量）：`Total = (Quality × 4 + Semantic × 1) / 5`（对应代码中 `QUALITY_WEIGHT = 4`、`SEMANTIC_WEIGHT = 1`）。

若某维度未出现在测评结果中，聚合时该维度按 **0** 计入（与 `normalized.get(k, 0)` 行为一致）。

默认 `work_dir` 为 `outputs/default`，可通过 `--work_dir` 指定。

### 依赖与 VBench 资源

- 测评逻辑使用 `ais_bench/third_party/vbench` 中的 VBench 1.0 接口。
- **detectron2**：部分维度（object_class、multiple_objects、color、spatial_relationship）依赖 GRiT，GRiT 依赖 detectron2。AISBench 统一使用仓库内 **`ais_bench/third_party/detectron2`** 作为唯一 detectron2 来源，GPU 与 NPU 通用。可在当前环境执行可编辑安装：`pip install -e ais_bench/third_party/detectron2 --no-build-isolation`（路径相对于仓库根目录），安装后 GPU/NPU 测评均使用该副本。
- （可选）在昇腾平台上，部分 torchvision 算子（例如 nms 及 roi_align）仅支持 CPU 推理，可能导致推理耗时过长。若当前环境为 `torch<2.7.1`，可参考 https://gitcode.com/Ascend/vision 安装适配昇腾的 torchvision 版本，以提升评测效率。
- 标准模式需 VBench 的 `VBench_full_info.json`，默认查找路径为 `ais_bench/third_party/vbench/VBench_full_info.json`；也可在数据集配置中通过 `full_json_dir` 指定。
- 各维度所需模型/权重需自行准备，并符合 VBench 官方说明。

#### 小模型依赖与本地缓存

VBench 涉及多种第三方模型/权重（例如 CLIP、DINO、MUSIQ、RAFT、ViCLIP、GRiT、Tag2Text、BERT base 等），为了方便统一管理和离线复现，AISBench 约定：

- 所有 VBench 相关小模型默认缓存到环境变量 `VBENCH_CACHE_DIR` 指定的目录；若未设置该变量，则默认使用 `~/.cache/vbench`。
- 也可在 VBench 示例配置**顶层**设置 `VBENCH_CACHE_DIR = "/path/to/cache"`（或别名 `vbench_cache_dir`）。在测评子进程内，若配置了非空值，会在导入 vbench 前覆盖该进程的同名环境变量；未配置时仍使用 shell 中的 `export`。一键下载脚本只认 shell 环境变量，与配置文件无关；需与测评同目录时请事先 `export` 相同路径。详见 [vbench_cache_dependencies.md](vbench_cache_dependencies.md)。
- 推荐在首次运行 VBench 测评前，先在仓库根目录执行一键缓存脚本：

```bash
# 使用默认缓存目录 ~/.cache/vbench
bash ais_bench/configs/vbench_examples/download_vbench_cache.sh

# 或指定自定义缓存目录
VBENCH_CACHE_DIR=/your/custom/cache/dir \
  bash ais_bench/configs/vbench_examples/download_vbench_cache.sh
```

- 脚本会自动跳过已存在的文件，多次执行是安全的。
- 若脚本因网络/权限等原因失败，或需要手动下载某些依赖，请参考同目录下的 [vbench_cache_dependencies.md](vbench_cache_dependencies.md) 获取完整依赖清单和手动下载指南。

### 运行模式 `-m eval` 说明与限制

AISBench 工作流中，`-m eval` 对应阶段为 **JudgeInfer → Eval → AccViz**，**不包含**主流程的 **Infer**（OpenICL 推理）。因此：

- **常规 OpenICL 任务**：`-m eval` 假设此前已通过 `all` / `infer` 或 `--reuse` 得到 `predictions/` 下的推理产物；否则评测阶段通常无法正常消费结果。
- **本目录 VBench 示例**：各维度分数在 **Eval** 阶段由 `VBenchEvalTask` 直接读取视频目录计算，**不依赖**上述 `predictions/*.jsonl`。数据集中若无 `judge_infer_cfg`，**JudgeInfer** 会整段跳过。此时使用 `ais_bench <config.py> -m eval` 与默认 `all` 相比，往往只是少跑一段占位 Infer；若 Infer 阶段在环境中不可用，可优先使用 `-m eval`。
- `-m eval` 仍会执行 **Eval** 与 **AccViz**（汇总）；`-m viz` 仅做汇总，不跑评测。
- CLI 帮助中的「evaluate existing inference results」主要针对通用流程；VBench 读视频算分属于例外，不必先具备 predictions。

## Prompt Suite（官方 prompt 结构）

VBench 提供按维度和按内容类别的 prompt 集合：

| 路径 | 说明 |
|------|------|
| `prompts/prompts_per_dimension/` | 各测评维度对应的 prompt 文件（约 100 条/维度） |
| `prompts/all_dimension.txt` | 全维度合并的 prompt 列表 |
| `prompts/prompts_per_category/` | 8 类内容：Animal, Architecture, Food, Human, Lifestyle, Plant, Scenery, Vehicles |
| `prompts/all_category.txt` | 全类别合并 |
| `prompts/metadata/` | color、object_class 等需语义解析的 metadata |

以上路径均相对于 `ais_bench/third_party/vbench/`。

### 维度与 Prompt Suite 映射

测评时不同维度使用不同的 prompt 文件，VBench 通过 `VBench_full_info.json` 自动匹配：

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

### Standard 数据集（eval_vbench_standard）

- **数据来源**：上述 Prompt Suite，路径 `ais_bench/third_party/vbench/prompts/`。
- **元数据**：需 `VBench_full_info.json`（默认 `ais_bench/third_party/vbench/VBench_full_info.json`）。
- **生成逻辑**：每个 prompt 采样 5 个视频；**temporal_flickering 需 25 个**，以在 static filter 后保持足够覆盖。
- **随机种子**：建议每个视频使用不同 seed（如 `index` 或 `seed+index`），确保多样性且可复现。
- **目录结构**：支持扁平目录或按维度子目录（如 `scene/`、`overall_consistency/` 等），详见 `ais_bench/third_party/vbench/__init__.py` 中 `dim_to_subdir` 映射。

### Custom 数据集（eval_vbench_custom）

- **数据来源**：用户自定义 prompt 列表或 prompt 文件。
- **支持维度**：`subject_consistency`, `background_consistency`, `aesthetic_quality`, `imaging_quality`, `temporal_style`, `overall_consistency`, `human_action`, `temporal_flickering`, `motion_smoothness`, `dynamic_degree`（不含 object_class、color、spatial_relationship 等需 auxiliary_info 的维度）。

## 采样伪代码（参考官方）

**仅测评部分维度**：

```python
dimension_list = ['object_class', 'overall_consistency']
for dimension in dimension_list:
    if args.seed is not None:
        torch.manual_seed(args.seed)
    with open(f'ais_bench/third_party/vbench/prompts/prompts_per_dimension/{dimension}.txt', 'r') as f:
        prompt_list = [line.strip() for line in f if line.strip()]
    n = 25 if dimension == 'temporal_flickering' else 5
    for prompt in prompt_list:
        for index in range(n):
            video = sample_func(prompt, index)
            save_path = f'{args.save_path}/{prompt}-{index}.mp4'
            torchvision.io.write_video(save_path, video, fps=8)
```

**测评全维度**：

```python
if args.seed is not None:
    torch.manual_seed(args.seed)
with open('ais_bench/third_party/vbench/prompts/all_dimension.txt', 'r') as f:
    prompt_list = [line.strip() for line in f if line.strip()]
for prompt in prompt_list:
    for index in range(5):
        video = sample_func(prompt, index)
        save_path = f'{args.save_path}/{prompt}-{index}.mp4'
        torchvision.io.write_video(save_path, video, fps=8)
```

> `sample_func` 为调用多模态生成模型将 `prompt` 转化为视频的函数。

## 推理与测评流程

```bash
# 仅测评（视频已生成）
ais_bench ais_bench/configs/vbench_examples/eval_vbench_standard.py
ais_bench ais_bench/configs/vbench_examples/eval_vbench_custom.py
```

- 修改配置中的 `DATA_PATH` 指定视频目录。
- 可通过 `path=/your/video/dir` 在命令行覆盖配置（若 CLI 支持）。

## 格式要求

### Standard 模式

- **文件名**：`{prompt}-{i}.mp4`，其中 `{prompt}` 为 VBench_full_info.json 中的 prompt_en，`i` 为 0~4。
- **支持扩展名**：`.mp4`, `.gif`, `.jpg`, `.png`。

### Custom 模式（格式更宽松）

- **方式一**：文件名即 prompt，`get_prompt_from_filename` 会从 `{xxx}.mp4` 或 `{xxx}-0.mp4` 提取 `xxx` 作为 prompt。
- **方式二**：提供 `prompt_file`（JSON 格式 `{video_path: prompt}`），无需遵守文件名约定。
