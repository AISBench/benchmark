# VBench 1.0

**VBench** (VBench: Comprehensive Benchmark Suite for Video Generative Models) is a benchmark suite for video generative models. It organizes evaluation around perception-related metrics such as subject consistency, motion smoothness, temporal flickering, and spatial relationship (the official Standard suite has **16 dimensions**), and provides matching prompts, pipelines, and validation methods for each dimension.

AISBench has **adapted to VBench 1.0**. The repository directory `ais_bench/configs/vbench_examples/` contains **standalone configuration file** examples for running quality/semantic dimension evaluation on generated videos on **GPU** or **NPU**. **AISBench currently does not include multimodal video generation**, so please generate videos first and then run the evaluation. (For Standard mode, see the [Dataset Generation](#dataset-generation) section.)

## Table of Contents

- [Dependencies and Environment](#dependencies-and-environment)
- [Quick Start](#quick-start)
- [Configuration and Output](#configuration-and-output)
- [Score Aggregation (Quality / Semantic / Total)](#score-aggregation-quality--semantic--total)
- [Prompt Suite (Official Prompt Structure)](#prompt-suite-official-prompt-structure)
- [Dataset Generation](#dataset-generation)
- [Sampling Pseudocode (Reference Official)](#sampling-pseudocode-reference-official)
- [Format Requirements](#format-requirements)

## Dependencies and Environment

#### decord (Video Decoding)

On **x86_64**, `pip install decord` usually works directly. On **ARM** and other environments without prebuilt wheels, build from source, for example:

```bash
git clone https://github.com/dmlc/decord
cd decord
mkdir build && cd build
cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release
make
cd ../python
python3 setup.py install --user
```

#### detectron2 and GRiT

Dimensions like `object_class`, `multiple_objects`, `color`, and `spatial_relationship` depend on GRiT, which in turn depends on detectron2. AISBench **uniformly** uses the in-repo **`ais_bench/third_party/detectron2`** (shared by GPU/NPU). Run an editable install from the repo root:

`pip install -e ais_bench/third_party/detectron2 --no-build-isolation`

#### torchvision on Ascend (Optional)

Some torchvision operators (such as `nms` and `roi_align`) may run only on CPU on Ascend, leading to low evaluation efficiency. If `torch < 2.7.1`, refer to [Ascend torchvision adaptation](https://gitcode.com/Ascend/vision) to install a matching version for speedup.

## Quick Start

1. **Prepare the video directory**
   For both Standard and Custom modes, set `DATA_PATH` in the corresponding configuration to the root directory of the generated videos (absolute or relative path). You can also copy the configuration file, change `DATA_PATH`, and then run `ais_bench <your_config.py> --mode eval`. (See [Dataset Generation](#dataset-generation) for video sampling notes.)

2. **Download third-party dependencies to local cache**
   VBench loads multiple small model weights for video generation quality evaluation. It is recommended to download them in advance. By default, the evaluation will also try to download dependencies automatically, but downloads may fail and break the evaluation. For details, see [`vbench_cache_dependencies.md`](./vbench_cache_dependencies.md).

```bash
# Use default cache directory ~/.cache/vbench
bash ais_bench/configs/vbench_examples/download_vbench_cache.sh

# Or specify a custom cache directory
VBENCH_CACHE_DIR=/your/custom/cache/dir \
  bash ais_bench/configs/vbench_examples/download_vbench_cache.sh
```

3. **Specify the cache in the config**
   Setting `VBENCH_CACHE_DIR = "/path/to/cache"` (or the alias `vbench_cache_dir`) at the top of the example configuration overrides the environment variable in the evaluation subprocess **before** vbench is imported. If not set, the `export VBENCH_CACHE_DIR` from the shell is used.

4. **Run the evaluation (must explicitly specify `--mode eval`)**

```bash
# Standard (16 dimensions, official Prompt Suite)
ais_bench ais_bench/configs/vbench_examples/eval_vbench_standard.py --mode eval --max-num-workers 1

# Custom (10 dimensions, custom prompts)
ais_bench ais_bench/configs/vbench_examples/eval_vbench_custom.py --mode eval --max-num-workers 1
```

**Note**: It is recommended to set `--max-num-workers <num>` to evaluate on multiple devices in parallel for better throughput.

## Configuration and Output

### Common Configuration Items

| Config Item / Environment Variable | Description |
| --- | --- |
| `DATA_PATH` | Root directory of videos to evaluate (**required**). See `ais_bench/configs/vbench_examples/eval_vbench_standard.py` and `ais_bench/configs/vbench_examples/eval_vbench_custom.py` |
| `VBENCH_CACHE_DIR` (env var or top-level config) | Cache root directory for small models and weights; default is `~/.cache/vbench` |

### Preset Configurations

| Config Name | Description | Configuration File |
| --- | --- | --- |
| eval_vbench_standard | Standard prompt evaluation, 16 dimensions; requires the video directory and an (optional) full_info json | `ais_bench/configs/vbench_examples/eval_vbench_standard.py` |
| eval_vbench_custom | Custom input (prompt from file or filename), 10 dimensions | `ais_bench/configs/vbench_examples/eval_vbench_custom.py` |

### Evaluation Result Path

Written per dimension:

```
{work_dir}/results/vbench_eval/vbench_<dim>.json
```

Standard mode uses `VBenchSummarizer` to aggregate Quality, Semantic, and Total; Custom mode uses `DefaultSummarizer` to output per-dimension scores. The implementation follows the official `cal_final_score.py`. See `ais_bench/benchmark/summarizers/vbench.py`.

## Score Aggregation (Quality / Semantic / Total)

### Per-Dimension Normalization and Weighting

For each dimension, the raw accuracy is linearly scaled to \((raw - Min)/(Max - Min)\) using the dimension's **Min** and **Max** in `NORMALIZE_DIC`, then multiplied by **DIM_WEIGHT**. If \(Max - Min \le 0\), the implementation falls back to the boundary value. Constants are aligned with the official ones, all in `vbench.py`. **`dynamic_degree` has DIM_WEIGHT 0.5**, while all other aggregated dimensions are **1**.

### Quality Group (Video Generation Quality)

Sum the per-dimension weighted scores within the Quality group, then divide by the **sum of DIM_WEIGHTs** (weighted average). Includes:

`subject_consistency`, `background_consistency`, `temporal_flickering`, `motion_smoothness`, `aesthetic_quality`, `imaging_quality`, `dynamic_degree`.

### Semantic Group (Content and Semantic Consistency)

Computed similarly; all dimensions in this group have DIM_WEIGHT **1**. Includes:

`object_class`, `multiple_objects`, `human_action`, `color`, `spatial_relationship`, `scene`, `appearance_style`, `temporal_style`, `overall_consistency`.

### Total (Overall Score)

`Total = (Quality × 4 + Semantic × 1) / 5` (corresponds to `QUALITY_WEIGHT = 4` and `SEMANTIC_WEIGHT = 1` in the code).

### Missing Dimensions and Output Directory

When a dimension is missing from the results, aggregation treats it as **0** (consistent with `normalized.get(k, 0)`).

The default `work_dir` is `outputs/default`; use `--work_dir` to change it.

## Prompt Suite (Official Prompt Structure)

Paths are relative to `ais_bench/third_party/vbench/`:

| Path | Description |
| --- | --- |
| `prompts/prompts_per_dimension/` | Prompt files per evaluation dimension (~100 entries/dimension) |
| `prompts/all_dimension.txt` | Combined list across all dimensions |
| `prompts/prompts_per_category/` | 8 categories: Animal, Architecture, Food, Human, Lifestyle, Plant, Scenery, Vehicles |
| `prompts/all_category.txt` | Combined across all categories |
| `prompts/metadata/` | Metadata that requires semantic parsing, such as `color` and `object_class` |

### Dimension and Prompt Suite Mapping (Standard, 16 Dimensions)

The following table shows the mapping of **all 16 dimensions** in Standard mode to the official Prompt Suite files and their entry counts. During evaluation, prompts are matched automatically via `VBench_full_info.json`:

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

## Dataset Generation

This section is for users who **need to generate evaluation videos using the official approach** (and does not conflict with the Quick Start that runs evaluation on an existing directory).

### Standard Dataset (eval_vbench_standard)

- **Data Source**: The Prompt Suite under `ais_bench/third_party/vbench/prompts/`.
- **Metadata**: Requires `VBench_full_info.json` (default file in the third-party directory above).
- **Sampling Scale**: Typically **5** videos per prompt; **`temporal_flickering`** requires **25** videos so that enough samples remain after the static filter.
- **Random Seed**: A different seed per video is recommended (e.g., `index` or `seed+index`) to balance diversity and reproducibility.
- **Directory Shape**: A flat directory or per-dimension subdirectories are both supported. If you use the per-dimension subdirectory layout, it is recommended to use the same subdirectory naming as the evaluation reading side (see below). The mapping logic is in `dim_to_subdir` in `ais_bench/third_party/vbench/__init__.py`.

#### DATA_PATH Directory Layout for Standard Mode

```
DATA_PATH/
|-- subject_consistency/        # same-name dimension
|-- scene/                      # background_consistency is mapped here first
|-- overall_consistency/        # aesthetic_quality / imaging_quality are mapped here first
|-- object_class/
|-- multiple_objects/
|-- color/
|-- spatial_relationship/
|-- temporal_style/
|-- human_action/
|-- temporal_flickering/
`-- appearance_style/
```

Among them, the following dimensions preferentially use the mapped subdirectory during evaluation (if it exists):

- `background_consistency` → `scene/`
- `aesthetic_quality` → `overall_consistency/`
- `imaging_quality` → `overall_consistency/`
- `motion_smoothness` → `subject_consistency/`
- `dynamic_degree` → `subject_consistency/`

Filenames are recommended to follow `{prompt}-{i}.mp4`. If you generated `0~24` for `temporal_flickering` (used by the static filter), at least make sure `0~4` exist; the evaluation side defaults to looking up `0~4` when constructing the temporary `full_info`.

### Custom Dataset (eval_vbench_custom)

- **Data Source**: A custom prompt list or prompt file.
- **Dimensions**: Includes `subject_consistency`, `background_consistency`, `aesthetic_quality`, `imaging_quality`, `temporal_style`, `overall_consistency`, `human_action`, `temporal_flickering`, `motion_smoothness`, `dynamic_degree`. **Excludes** dimensions that require `auxiliary_info` (such as `object_class`, `color`, and `spatial_relationship`).

## Sampling Pseudocode (Reference Official)

The pseudocode below aligns with the Standard reading-side logic: **iterate over dimensions**, save videos under the corresponding subdirectory, and use a larger sample count for `temporal_flickering`.

```python
import os

# Directory mapping on the evaluation reading side (see ais_bench/third_party/vbench/__init__.py)
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

`sample_func` denotes the function that connects your generative model to a `prompt` and produces a video.

## Format Requirements

### Standard Mode

- **Filename**: `{prompt}-{i}.mp4`, where `{prompt}` is `prompt_en` from `VBench_full_info.json` and `i` ranges from 0 to 4.
- **Extensions**: `.mp4`, `.gif`, `.jpg`, `.png`.

### Custom Mode (More Flexible)

- **Method 1**: Embed the prompt in the filename: `get_prompt_from_filename` parses `xxx` from `{xxx}.mp4` or `{xxx}-0.mp4`.
- **Method 2**: Provide a `prompt_file` (JSON: `{video_path: prompt}`); filename conventions can be ignored.
