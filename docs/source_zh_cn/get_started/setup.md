# VABench + AISBench 统一环境（`vabench311_new`）配置记录

> 代码版本要求：以下custom评测、官方格式备份、指标组进度和新配置模板对应后续配套代码，不适用于仅含10e1469的版本。本次提交为文档更新，使用前需同步配套代码。

> 状态：以下是7条视频现场已跑通的专项统一环境记录，保留已知Pillow/fsspec/tqdm依赖冲突，不作为通用默认安装方案。默认依赖一致的方案见[vabench_gpu_environment.md](vabench_gpu_environment.md)，AISBench主环境通过BACKEND_PYTHON调用同一个15项评分Conda。若复用统一环境，必须保存最终freeze、pip check输出及原始评分报告，不能把pip check失败改记为通过。

本文记录 `vabench311_new` 这个 Conda 环境从创建到能够被 AISBench 调度、并跑通 VABench 15 项指标的实际配置过程。

## 环境概览

| 项目 | 值 |
| --- | --- |
| Conda 环境 | `/root/miniconda3/envs/vabench311_new` |
| Python | 3.11.18 |
| AISBench | 3.1.0，editable，commit `10e1469` |
| VABench 源码 | `/workspace/code/VABench-main` |
| 模型缓存 | `/workspace/vabench-cache` |
| 安装/检查审计目录 | `/workspace/vabench-env-audit` |
| 最终 pip 包数量 | `pip freeze` 约 217 行 |

核心原则：

> **先装 VABench 评分后端，再把 AISBench 以 `--no-deps` 装进同一环境，最后回钉 VABench/NISQA 需要的版本。**

之所以不能简单执行一条 `pip install -r`，是因为 VABench/NISQA 对 `Pillow`、`tqdm`、`fsspec`、`torch` 等有硬性版本要求，而 AISBench 的通用 runtime 会升级或替换这些依赖。如果先装 AISBench，15 项指标很可能在 import 或推理阶段失败。

---

## 1. 创建环境

```bash
/root/miniconda3/bin/conda create -n vabench311_new python=3.11 pip -y --offline
```

说明：

- 这是新建的空环境，不是从 `vabench311` clone 的。
- Python 版本为 3.11.18。
- Conda 初始安装 pip 26.2.2，后续 VABench 安装器会将其固定为 pip 25.2。
- 本机 Conda 包缓存已有 Python 3.11，因此使用了 `--offline`；换机器复现时应去掉该参数。

---

## 2. 第一段：安装 VABench 评分后端

当前可用的安装命令（`--video` 可选；不传时检查阶段用 FFmpeg 生成临时测试视频）：

```bash
cd /workspace/code/benchmark

/root/miniconda3/envs/vabench311_new/bin/python3 \
  -m ais_bench.benchmark.vabench.environment install \
  --source-root /workspace/code/VABench-main \
  --cache-dir /workspace/vabench-cache \
  --work-dir /workspace/vabench-env-audit \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple
```

> **关于系统 ffmpeg：** 上面这条命令默认**不会**下载或安装 ffmpeg。它只会通过 `shutil.which` 检查 `ffmpeg`、`ffprobe`、`git`、`gcc`、`g++`、`make` 是否已经存在于系统 PATH 中。如果系统缺少 ffmpeg，需要先手动执行 `apt-get install -y ffmpeg`，或者在安装命令中增加 `--install-system-deps`（仅限 Ubuntu/Debian、root）；后者会通过 apt 安装 `git`、`ffmpeg`、`libsndfile1`、`build-essential`、`ca-certificates`。

### 2.1 执行顺序上的两个关键点

1. **必须在 AISBench 源码根目录执行。**
   此时新环境里还没有安装 AISBench 包，但当前目录在 `sys.path` 中，`python -m ais_bench...` 可以直接从源码加载安装器。

2. **必须在安装 AISBench 本体之前执行。**
   安装器中有 `runtime_guard`，一旦检测到 `ais-bench-benchmark` 已安装，会直接拒绝继续，要求使用独立评分环境。

> 当前环境已经安装了 AISBench，因此现在直接重跑 `environment install` 会被 `runtime_guard` 拒绝。
> 重跑只能用 `environment check`，或者换新环境/先卸载 AISBench。

安装使用的是华为云 PyPI 镜像：

```text
https://repo.huaweicloud.com/repository/pypi/simple
```

此前也测试过默认源、清华源和阿里云源，最终华为云下载最稳定。

### 2.2 安装器执行的 4 个 Stage

| Stage | 实际执行的命令 | 主要下载/安装内容 |
| --- | --- | --- |
| 1/4 | `pip install -r requirements/datasets/vabench-build.txt` | pip 25.2、setuptools 80.9.0、wheel 0.45.1、numpy 1.26.4、cython 3.1.4 |
| 2/4 | `pip install --no-build-isolation -r requirements/datasets/vabench.txt` | torch 2.6.0、torchvision 0.21.0、torchaudio 2.6.0、CUDA 12.4 运行库、transformers 4.57.0、accelerate 1.10.1、qwen-omni-utils 0.0.8、decord 0.6.0、onnxruntime-gpu 1.23.0、insightface 0.7.3、audiobox-aesthetics 0.0.4、laion-clap 1.1.7、msclap 1.3.4、librosa/soundfile、opencv-python-headless 等 |
| 3/4 | `pip install --no-deps --no-build-isolation --force-reinstall -r requirements/datasets/vabench-git.txt` | ImageBind @ `53680b0...`、PyTorchVideo @ `2ac6ed3...`、CLIP @ `d05afc4...` |
| 4/4 | `pip download --only-binary=:all: --no-deps nisqa==2.0.post2`，然后打补丁并 `pip install --no-deps --force-reinstall` | NISQA 2.0.post2 原始 wheel + 仅修改元数据的本地 wheel |

### 2.3 为什么 NISQA 需要单独打补丁

NISQA 上游 wheel 的 `METADATA` 把依赖写死为：

- `torch==2.2.1`
- `torchvision==0.17.1`
- `scipy==1.12.0`
- `scikit-learn==1.4.1.post1`
- `matplotlib==3.8.3`

如果直接带依赖安装，会把 VABench 的 torch 2.6.0 降级。安装器的处理方式是：

1. 下载官方 `nisqa==2.0.post2` wheel，校验原始 SHA256：
   `386205dc930aa98114a689f301112852dfe661d81eb4e6e630675ae9acd2c593`
2. 仅把这 5 个 `==` 改成 `>=`；
3. 重新生成 `RECORD`，包代码、权重、版本完全不变；
4. 生成 patched wheel，SHA256：
   `4c215a7e698c0bfa02daa22d20ec814c3a0764b5201168321aa981358782e275`
5. 使用 `--no-deps --force-reinstall` 安装 patched wheel。

补丁详情记录在：

```text
/workspace/vabench-env-audit/patched/nisqa_patch.json
```

其中 `package_code_unchanged: true`。

### 2.4 Git 包固定到具体 commit

以下三个包不是按 PyPI 版本安装，而是从 GitHub 固定 commit 构建：

| 包 | Commit |
| --- | --- |
| ImageBind | `53680b02d7e37b19b124fa37bae4b6c98c38f5be` |
| PyTorchVideo | `2ac6ed361309cb25144636ad82dd8bd6a4970e0c` |
| CLIP | `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |

安装参数：

```bash
pip install --no-deps --no-build-isolation --force-reinstall -r vabench-git.txt
```

- `--no-deps`：避免它们的依赖再次解析、改动 torch 等版本。
- `--no-build-isolation`：复用当前环境固定的 setuptools/wheel。
- `--force-reinstall`：即使环境里已有同名 PyPI 包，也强制换成指定 commit。

### 2.5 外部源码与模型缓存

安装器还会检查或补齐以下资源：

- `catalog(source_root, 'standard')`：验证固定版本的 VABench 778 题标准数据。
- DNSMOS：
  `third_party/DNS-Challenge/DNSMOS/dnsmos_local.py`
  本次已存在，保留原文件，SHA256 为
  `1ab566afe006daab32ac7073296a5d0ef99f8b82f91c7266f3ccf26113d7a28b`。
  只有缺失时才从微软固定提交下载。
- Synchformer：
  缓存中存在 `cache/synchformer/Synchformer` 时，软链到 `third_party/Synchformer`。
- InsightFace `buffalo_l`：
  如果只有备用目录，补 `models/buffalo_l` 软链。
- 模型权重不会由 pip 下载。
  `/workspace/vabench-cache` 需要提前准备官方 VABench 缓存、`sfd_face.pth`、`buffalo_l`、Synchformer 等。

### 2.6 第一段安装后的检查

检查结果文件：

```text
/workspace/vabench-env-audit/environment_check.json
```

结果：

- `passed: true`
- `pip_check: true`
- `missing_resources: []`
- 15 个指标入口全部 import 成功
- CUDA、decord 实际解码、ONNX providers、ImageBind、Qwen classes、Synchformer lazy import 全部通过
- `pip check` 仅接受 `decord 0.6.0` 的平台标签警告
- `model_inference_verified: false`

`model_inference_verified: false` 表示检查只验证了导入、CUDA 和解码探针，不等于完整 15 项推理。此时 VABench 评分后端本身是干净且可用的。

---

## 3. 第二段：把 AISBench 装进同一环境

AISBench 源码已在 `/workspace/code/benchmark`，无需重新 clone。安装分三步。

### 3.1 生成去掉 Pillow 固定版本的 runtime 清单

从 `requirements/runtime.txt` 生成 `/workspace/aisbench_runtime_unified.txt`：

```bash
cd /workspace/code/benchmark

sed -e 's/^gradio-client$/gradio-client==1.14.0/' \
    -e 's/^openai$/openai==1.109.1/' \
    requirements/runtime.txt \
  | grep -viE '^\s*Pillow\s*==' \
  > /workspace/aisbench_runtime_unified.txt

printf 'anyio==4.10.0\n' >> /workspace/aisbench_runtime_unified.txt
```

关键修改：

- **删除 `Pillow==11.2.1`**，保留 VABench/NISQA 需要的 `Pillow 10.2.0`。
- 固定 `gradio-client==1.14.0`、`openai==1.109.1`、`anyio==4.10.0`。
  这些版本是先用 `pip install --dry-run --report` 和 wheel `METADATA` 检查依赖关系后确定的，避免 resolver 选到与现有 `httpx`/`anyio` 不兼容的新版本。
- 保留 `setuptools<=80.10.2`，实际仍为 80.9.0。

### 3.2 安装 AISBench runtime 依赖

```bash
/root/miniconda3/envs/vabench311_new/bin/python3 -m pip install \
  -r /workspace/aisbench_runtime_unified.txt \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple
```

新增的主要是 AISBench 调度所需：

- `mmengine-lite==0.10.7`
- `datasets==3.6.0`
- `evaluate`、`absl-py`、`addict`、`aiohttp`
- `openai==1.109.1`、`gradio-client==1.14.0`、`httpx`、`anyio==4.10.0`
- `nltk`、`rouge`、`rouge_chinese`、`rouge_score`、`sacrebleu`
- `tree-sitter==0.21.3`、`tree_sitter_languages`
- `tiktoken`、`pyext==0.5`、`fuzzywuzzy`、`python-Levenshtein`
- `plotly`、`seaborn`、`orjson`、`janus`、`jsonlines` 等

这一步会让 pip 把 `tqdm` 升到 4.70.1、`fsspec` 升到 2025.3.0，与 NISQA/VABench 的固定版本冲突；下一步会重新钉回。

### 3.3 安装 AISBench 本体：不解析依赖

```bash
/root/miniconda3/envs/vabench311_new/bin/python3 -m pip install \
  -e /workspace/code/benchmark \
  --no-deps \
  --no-build-isolation
```

- `--no-deps` 是关键：不让 pip 重新解析 AISBench 的 `install_requires`，否则它会按元数据把 Pillow 拉回 11.2.1，还可能改动 torch/transformers。
- `--no-build-isolation` 让 editable 安装复用当前环境的 setuptools/wheel。
- 安装后 `ais-bench-benchmark==3.1.0` 指向 `/workspace/code/benchmark`，`ais_bench` 命令可用。

### 3.4 重新钉回 fsspec / tqdm

```bash
/root/miniconda3/envs/vabench311_new/bin/python3 -m pip install \
  "fsspec==2025.9.0" "tqdm==4.66.2" \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple
```

- `tqdm==4.66.2`：NISQA 的精确要求。
- `fsspec==2025.9.0`：VABench 固定环境中的版本。
- 代价是 `datasets 3.6.0` 会抱怨 `tqdm>=4.66.3` 和 `fsspec<=2025.3.0`；但实际 VABench 调度路径不依赖这两个新版本，`datasets.Dataset.from_list` 的冒烟测试也通过了。

### 3.5 导入验证

```bash
/root/miniconda3/envs/vabench311_new/bin/python3 -c "
import ais_bench
from ais_bench.benchmark.tasks.vabench_eval import VABenchEvalTask
from ais_bench.benchmark.datasets.vabench import VABenchDataset
print('ok')
"
```

确认 AISBench 本体、VABench task、VABench dataset 都能解析到。

---

## 4. 最终关键版本

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.18 |
| pip / setuptools / wheel | 25.2 / 80.9.0 / 0.45.1 |
| NumPy | 1.26.4 |
| torch / torchvision / torchaudio | 2.6.0 / 0.21.0 / 2.6.0（CUDA 12.4 运行库） |
| transformers / accelerate | 4.57.0 / 1.10.1 |
| qwen-omni-utils | 0.0.8 |
| decord | 0.6.0 |
| onnxruntime-gpu | 1.23.0 |
| insightface | 0.7.3 |
| audiobox-aesthetics / laion-clap / msclap | 0.0.4 / 1.1.7 / 1.3.4 |
| NISQA | 2.0.post2（仅元数据补丁） |
| ImageBind | 0.1.0 @ `53680b0...` |
| PyTorchVideo | 0.1.5 @ `2ac6ed3...` |
| CLIP | 1.0 @ `d05afc4...` |
| Pillow | 10.2.0 |
| AISBench | 3.1.0 editable @ `10e1469` |
| mmengine-lite / datasets | 0.10.7 / 3.6.0 |
| openai / gradio-client / anyio | 1.109.1 / 1.14.0 / 4.10.0 |
| tqdm / fsspec | 4.66.2 / 2025.9.0 |

---

## 5. 已知 pip 冲突与为什么还能运行

最终环境执行 `pip check` 仍会报 4 条：

```text
ais-bench-benchmark 3.1.0 has requirement Pillow==11.2.1, but you have pillow 10.2.0.
datasets 3.6.0 has requirement fsspec[http]<=2025.3.0,>=2023.1.0, but you have fsspec 2025.9.0.
datasets 3.6.0 has requirement tqdm>=4.66.3, but you have tqdm 4.66.2.
decord 0.6.0 is not supported on this platform
```

这些是统一环境的刻意取舍：

- **Pillow 10.2.0**：NISQA/VABench 需要；AISBench 元数据要求 11.2.1。VABench 调度路径实际跑通。
- **fsspec 2025.9.0**：VABench freeze 版本；`datasets` 希望 `<=2025.3.0`，但 `Dataset.from_list` 冒烟测试通过。
- **tqdm 4.66.2**：NISQA 精确要求；`datasets` 希望 `>=4.66.3`，但 VABench 任务未受影响。
- **decord 0.6.0 平台标签**：环境检查接受该警告，因为实际 decord 解码 probe 通过。

安装 AISBench 后重新运行环境检查，结果文件：

```text
/workspace/vabench-env-audit/post_aisbench_unified/environment_check.json
```

结果：

- `passed: false`
- `versions`、`git`、`nisqa`、`probes` 全部通过
- `missing_resources: []`
- 只有 `pip_check: false`

也就是说，**失败仅来自 pip 元数据一致性检查，不是运行时导入/CUDA/解码失败**。

如果要求 `pip check` 全绿，应该按文档使用独立评分环境：AISBench 主环境通过 `BACKEND_PYTHON` 调用 `vabench311`，不要把两套完整依赖装进同一个环境。

---

## 6. 运行时如何接入

环境变量：

```bash
export VABENCH_ROOT=/workspace/code/VABench-main
export VABENCH_CACHE_DIR=/workspace/vabench-cache
export CUDA_VISIBLE_DEVICES=6,7
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

AISBench 配置中的关键值：

```python
DEVICE = 'cuda:0'
BACKEND_PYTHON = '/root/miniconda3/envs/vabench311_new/bin/python3'
NUM_DEVICES = 2
SOURCE_ROOT = '/workspace/code/VABench-main'
CACHE_DIR = '/workspace/vabench-cache'
```

要点：

- 物理卡 6/7 在子进程中是逻辑 `cuda:0/1`。
- `first_dnsmos` 通过 `device_overrides` 强制 CPU，其余指标使用 GPU。
- `VABENCH_CACHE_DIR` 要在导入官方模块前设置，部分路径在 import 时即确定。
- `prepare` 生成 prepared WAV/JSON/manifest，`run` 再调用 `BACKEND_PYTHON` 启动评分进程。
- 模型权重不在 pip 里，必须提前放在 `/workspace/vabench-cache`。

后续已用这个环境跑完 7 个视频的 15 项指标，AISBench 报告成功生成。实际运行验证比 `pip check` 更能说明该路径可用。

---

## 7. 审计文件

完整记录：

| 文件 | 内容 |
| --- | --- |
| `/workspace/vabench-env-audit/install.json` | 安装状态、requirement SHA256、NISQA/DNSMOS 来源 |
| `/workspace/vabench-env-audit/installed.freeze.txt` | VABench 安装后的 freeze（164 行） |
| `/workspace/vabench-env-audit/installed.freeze.unified.txt` | 统一环境最终 freeze（217 行） |
| `/workspace/vabench-env-audit/environment_check.json` | VABench 单独检查结果 |
| `/workspace/vabench-env-audit/post_aisbench_unified/environment_check.json` | 装入 AISBench 后的检查结果 |
| `/workspace/vabench-env-audit/patched/nisqa_patch.json` | NISQA 元数据补丁详情 |
| `/workspace/vabench-env-audit-install.log` | VABench 依赖安装日志 |
| `/tmp/aisbench_runtime_install.log` | AISBench runtime 安装日志 |
| `/tmp/aisbench_editable_install.log` | AISBench editable 安装日志 |

---

## 8. 复现检查清单

1. 确认系统工具有 `ffmpeg`、`ffprobe`、`git`、`gcc`、`g++`、`make`；缺失时用 apt 或 `--install-system-deps` 安装。
2. 准备 `/workspace/vabench-cache` 模型缓存，以及 DNSMOS、Synchformer、`sfd_face.pth`、`buffalo_l`。
3. 创建 Python 3.11 Conda 环境。
4. **先**在 AISBench 源码根目录运行 `environment install`，安装 VABench 4 个 stage。
5. 运行 `environment check`，确认 15 项 import、CUDA、decord、ONNX 全部通过。
6. 生成去掉 Pillow 固定版本的 `aisbench_runtime_unified.txt`。
7. 安装 AISBench runtime 依赖。
8. 用 `pip install -e ... --no-deps --no-build-isolation` 安装 AISBench 本体。
9. 重新钉回 `fsspec==2025.9.0` 和 `tqdm==4.66.2`。
10. 验证 `VABenchEvalTask`/`VABenchDataset` 导入。
11. 接受已知 `pip check` 冲突，或改用独立评分环境。
12. 设置 `BACKEND_PYTHON`、`SOURCE_ROOT`、`CACHE_DIR`，执行 prepare/run；具体命令见第 9 节。

---

## 9. 运行与测试命令

前面的 `environment check` 只验证依赖、CUDA、decord、ONNX 等，不会真正跑 15 项评分。实际运行评测使用以下命令。

### 9.1 快速环境诊断（不跑模型）

```bash
cd /workspace/code/benchmark

/root/miniconda3/envs/vabench311_new/bin/python3 \
  -m ais_bench.benchmark.vabench.cli doctor \
  --source-root /workspace/code/VABench-main \
  --cache-dir /workspace/vabench-cache
```

输出会列出 `ffmpeg`/`ffprobe`、torch/CUDA、关键模块和缺失资源，适合正式跑之前确认环境。

### 9.2 准备自定义子集（仅首次）

如果 prepared 目录还不存在：

```bash
cd /workspace/code/benchmark

/root/miniconda3/envs/vabench311_new/bin/python3 \
  -m ais_bench.benchmark.vabench.cli prepare \
  --source-root /workspace/code/VABench-main \
  --subset custom \
  --videos /workspace/video \
  --output /workspace/prepared_all_official_v2
```

注意：`prepare` 不覆盖已有输出。`/workspace/prepared_all_official_v2/manifest.json` 已存在时不要重跑，否则会报错；已有 prepared 目录可直接跳到 9.3/9.4。9.4 配置中的 `PREPARED_DIR` 必须与这里一致。

### 9.3 单样本冒烟：只跑 CLAP

用途：验证 AISBench → 后端 Python → VABench worker 的调度链路。

```bash
cd /workspace

WORKDIR=/workspace/vabench_runs/aisbench/smoke_clap_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=6 \
/root/miniconda3/envs/vabench311_new/bin/ais_bench \
  /workspace/code/benchmark/ais_bench/configs/vabench_examples/eval_vabench_smoke_clap.py \
  -m eval \
  --work-dir "$WORKDIR"
```

配置要点：

- `PREPARED_DIR=/workspace/prepared_kitten_gpu`
- `dimensions=['second_clap']`
- `NUM_DEVICES=1`

### 9.4 全量 7 视频 / 15 项指标

`--work-dir` 必须是新目录；已有 `report.json` 时会拒绝运行。

```bash
cd /workspace

WORKDIR=/workspace/vabench_runs/aisbench/custom_all_7_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=6,7 \
/root/miniconda3/envs/vabench311_new/bin/ais_bench \
  /workspace/code/benchmark/ais_bench/configs/vabench_examples/eval_vabench_custom.py \
  -m eval \
  --work-dir "$WORKDIR"

echo "$WORKDIR"
```

配置要点：

- `PREPARED_DIR=/workspace/prepared_all_official_v2`
- `BACKEND_PYTHON=/root/miniconda3/envs/vabench311_new/bin/python3`
- `DEVICE='cuda:0'`、`NUM_DEVICES=2`
- 物理卡 6/7 对应子进程逻辑 `cuda:0/1`
- `first_dnsmos` 强制 CPU，其余指标使用 GPU
- 7 个 MLLM 指标在同一个 Qwen 子进程中复用

链路与评分状态应分别检查（以下不是官方对齐通过的判据）：

- 检查外层命令退出码和AISBench内部任务状态；外层返回0不能替代检查评分报告；
- 报告文件存在；
- `total.valid_dimensions == 15`；
- `total.missing_dimensions == []`；
- `total.incomplete_dimensions` 可能包含 `second_desync`、`second_lsa`，这是已记录的评分失败/省略，说明链路可执行但应评覆盖不完整；VABenchEvalTask会按不完整处理并保留报告；
- 自定义子集的 `total.status` 为 `preview`，不是 VABench 官方榜单总分。

`second_clap`长音频可能触发随机裁剪。本地官方缓存配置为`rand_trunc`而非`fusion`，应核对服务器的实际配置和哈希。逐指标原始均分相对误差<1%（近零时绝对差≤1e-6）的验收标准不变；需先控制相同输入、顺序及随机状态/采样片段。由于两条路径可能消耗不同数量的随机数，仅在程序开头设同一个seed未必足够。随机性未受控时标记待对齐，不能直接把验收改成多次均值；多次运行只作为附加稳定性分析。

报告路径示例（实际以 `$WORKDIR` 为准）：

```text
$WORKDIR/<AISBench运行时间戳>/results/custom_all_7/vabench_t2av_artifacts/report.json
```

同一 artifacts 目录下会自动生成官方格式备份：

```text
official_eval_results/<category>/<category>_<时间>_eval_results.{json,csv}
official_eval_results/<category>/<category>_<时间>_full_info.json
```

### 9.5 只做 dry-run（不跑模型）

```bash
cd /workspace

CUDA_VISIBLE_DEVICES=6,7 \
/root/miniconda3/envs/vabench311_new/bin/ais_bench \
  /workspace/code/benchmark/ais_bench/configs/vabench_examples/eval_vabench_custom.py \
  -m eval --dry-run
```

用于检查配置、数据集路径和任务拆分，不会执行评分。

### 9.6 命令行直接跑 VABench（可选）

如果不想走 AISBench eval 配置，也可以直接使用 VABench CLI：

```bash
/root/miniconda3/envs/vabench311_new/bin/python3 \
  -m ais_bench.benchmark.vabench.cli run \
  --source-root /workspace/code/VABench-main \
  --cache-dir /workspace/vabench-cache \
  --prepared /workspace/prepared_all_official_v2 \
  --device cuda:0 \
  --python /root/miniconda3/envs/vabench311_new/bin/python3 \
  --output /workspace/vabench_runs/aisbench/cli_custom_all_7_$(date +%Y%m%d_%H%M%S)
```

该路径会按 15 项指标运行，输出 `report.json`、CSV 和 `official_eval_results/` 官方格式备份；适合调试后端，不经过 AISBench 的任务看板/分片调度。

---

## 10. 一句话总结

这套安装顺序在记录的7条视频路径上已验证可执行，仍保留依赖冲突；不保证其他AISBench功能或重新解析依赖后的兼容性。应优先按独立评分环境方案复现，统一环境仅在锁定现场版本且接受已知范围时使用。






### 固定现场审计信息

在已跑通的统一Conda中执行并保存（这不会消除已知冲突）：

```bash
python3 -m pip freeze --all > /workspace/vabench-env-audit/installed.freeze.unified.txt
python3 -m pip check > /workspace/vabench-env-audit/pip-check.unified.txt 2>&1
```

进度看板单位为“指标组”：8个专家组加1个共享Qwen组，共9组；不再把组比例换算成已完成视频数。配置模板已随仓库交付，运行前填写模板中的路径与GPU卡数。
