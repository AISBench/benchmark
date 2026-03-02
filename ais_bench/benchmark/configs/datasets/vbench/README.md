# VBench 1.0

AISBench 已适配 VBench 1.0，支持在 **GPU（cuda）** 与 **NPU** 上进行视频/图像质量维度测评，使用方式与原有 `ais_bench --models *** --datasets ***` 一致。

## 使用方式

### 仅测评（推荐）

在已有一批生成视频目录的前提下，仅运行 VBench 测评：

```bash
# 设备会自动检测：NPU 可用则用 NPU，否则用 CUDA
ais_bench --mode eval --models vbench_eval --datasets vbench_standard
```

**注意**：需在配置中指定视频目录：

- 使用自定义配置：复制 `vbench_standard.py`，将其中 `path` 改为你的视频目录（绝对或相对路径），再通过 `--config` 指定该配置；或
- 直接修改 `benchmark/configs/datasets/vbench/vbench_standard.py` 中 `path` 为你的视频目录。

### 设备配置

- **默认**：设备自动检测——若当前环境 NPU 可用（`torch.npu.is_available()`）则使用 NPU，否则使用 CUDA；也可通过环境变量 `VBENCH_DEVICE` 指定。
- **强制指定**：若需固定设备，可在对应数据集配置的 `eval_cfg` 中设置 `device='cuda'` 或 `device='npu'`。

例如在 `vbench_standard.py` 中（不写 `device` 即自动检测）：

```python
vbench_eval_cfg = dict(
    use_vbench_task=True,
    load_ckpt_from_local=True,
    # device 不写则自动检测；可选 device='cuda' 或 device='npu' 强制指定
)
```

### 可用数据集配置

| 配置名 | 说明 | 配置文件 |
|--------|------|----------|
| vbench_standard | VBench 标准 prompt 测评，需提供视频目录与（可选）full_info json | [vbench_standard.py](vbench_standard.py) |
| vbench/vbench_custom | 自定义输入（prompt 来自文件或文件名） | [vbench_custom.py](vbench_custom.py) |

### 结果输出

测评结果写入：

```
{work_dir}/results/vbench_eval/vbench_standard_eval_results.json
```

默认 `work_dir` 为 `outputs/default`，可通过 `--work_dir` 指定。

### 依赖与 VBench 资源

- 测评逻辑使用 `ais_bench/third_party/vbench` 中的 VBench 1.0 接口。
- **detectron2**：部分维度（object_class、multiple_objects、color、spatial_relationship）依赖 GRiT，GRiT 依赖 detectron2。AISBench 统一使用仓库内 **`ais_bench/third_party/detectron2`** 作为唯一 detectron2 来源，GPU 与 NPU 通用。
  - **方式一（推荐）**：运行测评时无需额外操作，`VBenchEvalTask` 会自动将 `third_party` 与 `third_party/detectron2` 加入 `sys.path`，使 `import detectron2` 和 `import vbench` 解析到仓库内副本。
  - **方式二**：若希望全局可用，可在当前环境执行可编辑安装：`pip install -e ais_bench/third_party/detectron2`（路径相对于仓库根目录），安装后 GPU/NPU 测评均使用该副本。
- 标准模式需 VBench 的 `VBench_full_info.json`，默认查找路径为 `third_party/vbench/VBench_full_info.json`；也可在数据集配置中通过 `full_json_dir` 指定。
- 各维度所需模型/权重需自行准备，并符合 VBench 官方说明。

## Prompt Suite（官方 prompt 结构）

VBench 提供按维度和按内容类别的 prompt 集合：

| 路径 | 说明 |
|------|------|
| `prompts/prompts_per_dimension/` | 各测评维度对应的 prompt 文件（约 100 条/维度） |
| `prompts/all_dimension.txt` | 全维度合并的 prompt 列表 |
| `prompts/prompts_per_category/` | 8 类内容：Animal, Architecture, Food, Human, Lifestyle, Plant, Scenery, Vehicles |
| `prompts/all_category.txt` | 全类别合并 |
| `prompts/metadata/` | color、object_class 等需语义解析的 metadata |

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

### Standard 数据集（vbench_standard）

- **数据来源**：上述 Prompt Suite，路径 `ais_bench/third_party/vbench/prompts/`。
- **元数据**：需 `VBench_full_info.json`（默认 `third_party/vbench/VBench_full_info.json`）。
- **生成逻辑**：每个 prompt 采样 5 个视频；**temporal_flickering 需 25 个**，以在 static filter 后保持足够覆盖。
- **随机种子**：建议每个视频使用不同 seed（如 `index` 或 `seed+index`），确保多样性且可复现。
- **目录结构**：支持扁平目录或按维度子目录（如 `scene/`、`overall_consistency/` 等），详见 `third_party/vbench/__init__.py` 中 `dim_to_subdir` 映射。

### Custom 数据集（vbench_custom）

- **数据来源**：用户自定义 prompt 列表或 prompt 文件。
- **支持维度**：`subject_consistency`, `background_consistency`, `aesthetic_quality`, `imaging_quality`, `temporal_style`, `overall_consistency`, `human_action`, `temporal_flickering`, `motion_smoothness`, `dynamic_degree`（不含 object_class、color、spatial_relationship 等需 auxiliary_info 的维度）。

## 采样伪代码（参考官方）

**仅测评部分维度**：

```python
dimension_list = ['object_class', 'overall_consistency']
for dimension in dimension_list:
    if args.seed is not None:
        torch.manual_seed(args.seed)
    with open(f'prompts/prompts_per_dimension/{dimension}.txt', 'r') as f:
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
with open('prompts/all_dimension.txt', 'r') as f:
    prompt_list = [line.strip() for line in f if line.strip()]
for prompt in prompt_list:
    for index in range(5):
        video = sample_func(prompt, index)
        save_path = f'{args.save_path}/{prompt}-{index}.mp4'
        torchvision.io.write_video(save_path, video, fps=8)
```

## 推理与测评流程

```bash
# 仅测评（视频已生成）
ais_bench --mode eval --models vbench_eval --datasets vbench_standard
ais_bench --mode eval --models vbench_eval --datasets vbench_custom
```

- 通过 `--config` 或修改配置中的 `path` 指定视频目录。
- 可通过 `path=/your/video/dir` 在命令行覆盖配置。

## 格式要求

### Standard 模式

- **文件名**：`{prompt}-{i}.mp4`，其中 `{prompt}` 为 VBench_full_info.json 中的 prompt_en，`i` 为 0~4。
- **支持扩展名**：`.mp4`, `.gif`, `.jpg`, `.png`。

### Custom 模式（格式更宽松）

- **方式一**：文件名即 prompt，`get_prompt_from_filename` 会从 `{xxx}.mp4` 或 `{xxx}-0.mp4` 提取 `xxx` 作为 prompt。
- **方式二**：提供 `prompt_file`（JSON 格式 `{video_path: prompt}`），无需遵守文件名约定。
- **结论**：custom 模式下文件名不必叫 "prompt"，只要文件名能正确反映视频内容描述即可；若使用 prompt_file 则完全无文件名格式要求。

## 视频生成工具

用户只需实现 `generate(prompt, index) -> video` 的推理逻辑，工具自动遍历 prompt 列表并保存为 VBench 期望的 mp4 格式。支持官方 Prompt Suite 的全部模式。详见 `ais_bench/third_party/vbench/tools/video_generator.py`。

查看用法说明：

```bash
# 方式一：在仓库根目录执行
PYTHONPATH=ais_bench/third_party python -m vbench.tools

# 方式二：先进入 third_party 目录
cd ais_bench/third_party && python -m vbench.tools
```

**依赖**：保存视频需安装 `opencv-python` 或 `imageio`（推荐 `imageio-ffmpeg` 以支持 mp4）。

**导入说明**：使用 `from vbench.tools...` 时，需确保 `ais_bench/third_party` 在 `PYTHONPATH` 中，或在脚本开头添加 `sys.path.insert(0, 'ais_bench/third_party')`（路径相对于运行目录）。

**模式说明**：`custom` | `standard` | `all_dimension` | `all_category` | `category`。`standard` 会自动使用维度→Prompt Suite 映射；`temporal_flickering` 自动 25 视频/prompt。

**Custom 模式**：

```python
def my_generate(prompt: str, index: int):
    video = model.generate(prompt, seed=index)
    return video  # numpy (T,H,W,C) uint8 或 已保存路径

from vbench.tools.video_generator import run_vbench_generation

run_vbench_generation(
    generate_fn=my_generate,
    output_dir="./my_videos",
    mode="custom",
    prompt_source=["a cat running", "a dog swimming"],  # 或 "prompts.txt"
)
```

**Standard 单维度**（自动映射，如 background_consistency → scene.txt）：

```python
run_vbench_generation(
    generate_fn=my_generate,
    output_dir="./videos",
    mode="standard",
    dimension="overall_consistency",
    seed=42,  # 可选，用于复现
)
```

**全维度**（all_dimension.txt）：

```python
run_vbench_generation(
    generate_fn=my_generate,
    output_dir="./videos",
    mode="all_dimension",
    seed=42,
)
```

**按内容类别**（prompts_per_category）：

```python
run_vbench_generation(
    generate_fn=my_generate,
    output_dir="./videos",
    mode="category",
    category="animal",  # animal, architecture, food, human, lifestyle, plant, scenery, vehicles
)
```

**temporal_flickering**（自动 25 视频/prompt）：

```python
run_vbench_generation(
    generate_fn=my_generate,
    output_dir="./videos",
    mode="standard",
    dimension="temporal_flickering",
)
```

生成完成后运行：`ais_bench --mode eval --models vbench_eval --datasets vbench_standard`，配置 `path=./videos`。

## 与现有流程的兼容

- CLI 不变：仍使用 `--models`、`--datasets`（及可选 `--mode eval`、`--work_dir` 等）。
- 当所选数据集中包含 VBench 数据集（`eval_cfg.use_vbench_task=True` 或 `type=VBenchDataset`）时，Eval 阶段会自动使用 `VBenchEvalTask`，在 GPU 或 NPU 上跑 VBench 1.0 测评。
