# VABench 音视频联合评测快速上手

VABench 评估同一生成视频的音频质量、跨模态一致性、同步性和内容表现。本指南接入**已有带音轨 MP4**，由 AISBench 调度外部 VABench 计算 15 项指标，保存原始分数，并展示指标均分、归一化分和总分。视频生成 API 接入及 A3/NPU 验证属于后续阶段。

VABench 与 [VBench 视频评测](vbench.md) 使用不同的数据和指标，不能共用权重清单或总分规则。本文的归一化和总分是项目约定的 AISBench 展示协议，**不是 VABench 官方总分**。

已有 GPU 评分环境的推荐顺序：环境检查 → 获取题目 → 准备对应视频和 WAV → 运行 15 项评分 → 查看结果。第 3 节为固定 78 题输入准备，第 4 节为执行命令，第 6 节为原生官方对照。

需要从新环境逐步操作时，参阅[手动四步评测指南](../../get_started/vabench_get_start.md)，包含源码补齐、权重下载、统一Conda环境、主命令及原生官方对照。Dataset、Task和汇总实现已分别包化，两个原生操作脚本位于tools；公开类导入和主CLI命令保持不变。

## 1. 准备源码、环境与权重

当前操作入口统一为 `ais_bench.tools.vabench`，早期顶层兼容包已移除；安装和评测请使用下文命令。

以下命令在 Linux GPU 容器中执行，AISBench 源码目录为 `/workspace/code/benchmark`。评分环境使用 Conda Python 3.11。固定外部 VABench 版本：

```bash
git clone https://github.com/tanABCC/VABench.git /workspace/code/VABench-main
git -C /workspace/code/VABench-main checkout 12b9617a900a9eca1691da6b7115ce3d58f53a16

source /root/miniconda3/etc/profile.d/conda.sh
conda create -n vabench311 python=3.11 pip -y
conda activate vabench311
cd /workspace/code/benchmark
export VABENCH_ROOT=/workspace/code/VABench-main
export VABENCH_CACHE_DIR=/workspace/vabench-cache
export CUDA_VISIBLE_DEVICES=6,7
```

已有源码或环境可直接复用，跳过对应的 `git clone`、`conda create` 和依赖安装。先激活实际评分环境、进入 AISBench 根目录并设置下列环境变量；目录和卡号按实际机器修改。物理卡 6、7 在进程内是逻辑卡 0、1；必须保证它们有足够空闲显存。`max_memory` 是模型分配上限，不表示预留了这些显存。

### 已有评分环境：先检查新入口

在源码根目录执行，确保当前导入的是更新后的本地代码：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vabench311
cd /workspace/code/benchmark
export VABENCH_ROOT=/workspace/code/VABench-main
export VABENCH_CACHE_DIR=/workspace/vabench-cache
export CUDA_VISIBLE_DEVICES=6,7

python3 -m ais_bench.tools.vabench.cli --help
python3 -m ais_bench.tools.vabench.environment check \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --work-dir /workspace/vabench-env-audit
```

若实际环境名为 `vabench311_new`，激活该环境，并将后续 `--python` 同步改为它的 `bin/python3`。检查失败时按审计报告处理，不以已运行过模型代替检查，也不通过重复安装覆盖现有环境。

此次删除兼容目录只改变命令和导入入口，15 项实现、权重缓存及准备协议保持不变。使用当前协议生成的准备目录可直接复用；所有脚本应使用新的 `ais_bench.tools.vabench` 操作命令。

### 新环境：安装依赖

先按 [GPU 环境指南](../../get_started/vabench_gpu_environment.md) 下载模型缓存及额外人脸权重，再安装依赖。工具不会通过 pip 下载全部模型权重：

```bash
python3 -m ais_bench.tools.vabench.environment install \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --work-dir /workspace/vabench-env-audit \
  --install-system-deps \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple

python3 -m ais_bench.tools.vabench.environment check \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --work-dir /workspace/vabench-env-audit
```

`--install-system-deps` 适用于使用 apt 的 root 容器，安装 Git、FFmpeg、编译工具等；已有系统依赖时省略。`--index-url` 只指定 PyPI 软件源，可以省略以使用 pip 默认配置，不能代理 GitHub。必须检查命令返回值和审计报告。

安装入口处理固定版本的 PyPI/Git 包及 NISQA 元数据补丁。不能用一次 `pip install -r requirement.txt` 替代全部准备步骤。评分环境可与 AISBench 主环境分离；不要默认把两套依赖强行安装在一起。确需统一环境时参考已运行的 [统一环境记录](../../get_started/setup.md)，并如实保存 `pip check` 冲突。

## 2. 取得题目并准备生成视频

官方 `json_files/json_text/*.json` 共 848 条；本项目只使用与 `mapping/final_idx_to_prompt.csv` 精确匹配的 **778 条**。JSON 是题目和评分元数据，不包含待评测的生成视频。视频需要由被测模型生成或提供已有产物。

```bash
python3 -m ais_bench.tools.vabench.cli catalog \
  --source-root "$VABENCH_ROOT" --subset standard \
  --output /workspace/t2av_778_prompts.json
```

- 标准集：778 题，使用 `--subset standard`。
- 项目选定的 10% 子集：78 题，获取和评测见第 3 节。

生成时使用清单中 `entries[].metadata.prompt_en` 的完整文本。`idx` 只用于保存文件名，例如 `kitten curled up meowing.mp4`，不能用短标签替代完整 Prompt。每题保留一个模型的一份带音轨 MP4，可以平铺或按类别分目录；文件名中的空格合法，shell 参数须加引号。

本地 H3 请求中出现的 10 秒、768P、16:9 是该批次生成配置，不是官方评测器统一强制的输入规格。比较模型时应固定并记录生成参数，检查每项模型对实际媒体长度等条件的要求。

### 778 题完整集的输入准备

完整集视频全部就位后执行：

```bash
python3 -m ais_bench.tools.vabench.cli prepare \
  --source-root "$VABENCH_ROOT" --subset standard \
  --videos /workspace/generated_standard --output /workspace/prepared_standard
```

完整集继续第 4 节，将运行命令的 `--prepared` 改为 `/workspace/prepared_standard`、`--output` 改为新的完整集结果目录。评测固定 78 题时执行第 3 节，不使用这条完整集准备命令。

## 3. 获取并准备已选定的 10% 子集（78 题）

### 清单来源与抽样规则

用户通过 DeepSeek 分析形成的 `VABench/build_mini10pct.py`，其产物为 `VABench/mini10pct`。本次核查将其中选定的 **78 个 idx 与类别**固化到 AISBench 的 `ais_bench/configs/vabench_examples/mini10pct.json`；同步本分支后即可获取，不需要重新抽样或访问生成服务。它是项目选定集合，不是上游官方发布的子集。

| 项目 | 核验结果 |
| --- | --- |
| 抽样基数 | 可映射的 778 题，不是原始 848 条 |
| 选中数量 | `round(778×10%)=78`，实际占 10.03% |
| 类别覆盖 | 24/24 类，每类 2–4 题 |
| easy / hard | 40/398、38/380 |
| 类别名额 | easy 10 类各 4；hard 14 类先各 2，剩余 10 个名额分给最大的 10 个 hard 类 |
| 类内选择 | 对 `prompt_en + prompt_vision + prompt_audio` 做 TF-IDF，从 medoid 开始选择最远点，idx SHA256 解决同分；没有随机种子参数 |

该算法衡量词面多样性，不保证语义或全部指标适用性；题目覆盖不等于所有生成视频都可获得 LSA 等有效分数。清单保留上游提交、原始 778 清单、选择脚本及输出数据清单的 SHA256，便于审计。

原脚本写死了 `/mnt/c/需求/vabench/...`，末尾还复制 `h3.py`、`batch_run.py`、`prepare_dataset.py`、`.env.example`。这些辅助文件在当前本地源码根目录缺失，因此它不能原样作为通用 Linux 获取命令；`--force` 还会删除已有请求清单。**复用固化清单即可，不必重跑或覆盖原产物**。本次在临时目录重算的 78 题与现有清单相同；原产物没有改写。

### 导出完整 Prompt

在 AISBench 根目录执行以下脚本，校验固定清单并从官方源码导出对应元数据。它不调用任何生成 API：

```bash
cd /workspace/code/benchmark
python3 - <<'PY'
import os
from ais_bench.benchmark.utils.vabench.data import catalog, read_json, write_json

selected = read_json('ais_bench/configs/vabench_examples/mini10pct.json')
standard = {e['idx']: e for e in catalog(os.environ['VABENCH_ROOT'], 'standard')}
ids = [e['idx'] for e in selected['entries']]
assert len(ids) == len(set(ids)) == selected['count'] == 78
assert len({e['category'] for e in selected['entries']}) == 24
assert all(standard[e['idx']]['category'] == e['category'] for e in selected['entries'])
write_json('/workspace/mini10pct_prompts.json', {
    'subset': selected['name'], 'count': 78,
    'upstream_commit': selected['upstream_commit'],
    'entries': [standard[idx] for idx in ids],
})
print('已导出 78 题：/workspace/mini10pct_prompts.json')
PY
```

从导出的 `entries[].metadata.prompt_en` 生成视频，保存为 `<idx>.mp4`。同步/复制原有产物时只需这 78 题对应的 MP4，不要把任务状态文件、下载链接或 API Key 当成评测数据。本次本地 `mini10pct/output` 实际只有 **5 个 MP4**，另外 73 题尚无视频，不能宣称已经具备完整 10% 评测输入。

### 按固定 78 题准备输入

视频齐全后，显式传入全部 78 题到准备函数。这样缺少任一视频都会报错，避免普通 `--subset custom` 根据已有文件自动缩成 5 题：

```bash
python3 - <<'PY'
from ais_bench.benchmark.utils.vabench.data import prepare, read_json

prompts = read_json('/workspace/mini10pct_prompts.json')
assert prompts['count'] == len(prompts['entries']) == 78
prepare(prompts['entries'], '/workspace/generated_mini10pct',
        '/workspace/prepared_mini10pct', subset='custom')
print('已准备固定 78 题')
PY
```

准备完成后继续第 4 节执行评分，最后按第 5 节查看结果。

当前报告 scope 为 `custom`，应与固化清单、导出的 Prompt 文件一起归档，并核对 `subset_size=78`。也可以将该准备目录填入主命令配置。若只想先测已有 5 条，可按 [自定义子集说明](../../get_started/vabench_t2av.md) 操作，但报告必须标为“自定义 5 条冒烟”，不能当作 78 题结果。

## 4. 用 AISBench 启动评测

先按第 3 节导出固定 78 题并准备全部对应视频，生成 `/workspace/prepared_mini10pct`。激活实际评分 Conda 后，使用简化命令执行全部 15 项评分：

```bash
cd /workspace/code/benchmark
export CUDA_VISIBLE_DEVICES=6,7

python3 -m ais_bench.tools.vabench.cli run \
  --source-root /workspace/code/VABench-main \
  --cache-dir /workspace/vabench-cache \
  --prepared /workspace/prepared_mini10pct \
  --output /workspace/vabench_runs/mini10pct_simple
```

目录和卡号按实际机器调整，`--output` 使用新的评分目录。省略参数的实际行为为：

| 省略的参数 | 默认行为 |
| --- | --- |
| `--python` | 使用运行本命令的 Python，因此必须先激活评分 Conda |
| `--device` | 请求逻辑 `cuda:0`；可见卡为 6、7 时对应物理卡 6，CUDA 不可用时会报错 |
| `--options` | 使用代码及官方模型默认参数；DNSMOS 使用 CPU，Qwen 默认 `device_map="auto"`，每个评分子进程超时 24 小时 |

Qwen 自动分配可使用多张可见 GPU，因此默认 `cuda:0` 不等于限制所有模型只用一张卡。省略 `--options` 不会自动读取外部 VABench 的 `config.yaml`，其 `max_memory` 等额外参数不会自动继承。

需要指定另一个评分环境、设备或评分参数时，可使用以下完整命令，并按需修改选项；`options.json` 为可选配置文件：

```bash
python3 -m ais_bench.tools.vabench.cli run \
  --source-root /workspace/code/VABench-main \
  --cache-dir /workspace/vabench-cache \
  --prepared /workspace/prepared_mini10pct \
  --python /root/miniconda3/envs/vabench311/bin/python3 \
  --device cuda:0 \
  --options ais_bench/configs/vabench_examples/options.json \
  --output /workspace/vabench_runs/mini10pct_configured
```

`prepare` 检查缺失、重复文件及音视频流，从第一条音轨抽取同源 PCM16 WAV，不主动裁剪、混声道或重采样，也不重编码 MP4。报告保存媒体哈希，原始 MP4 必须保留；移动机器后重新准备。准备目录和完整评分目录应使用新路径。当前协议生成的准备目录可复用。若更早清单版本的目录报协议不一致，应换新目录重新准备；不能直接修改 manifest 哈希绕过校验。

默认计算全部 15 项。专家指标依次启动评分子进程，7 个 MLLM 指标在同一进程中复用 Qwen2.5-Omni-7B，无须部署 HTTP 服务。DNSMOS 显式使用 CPU ONNX。请求设备和观测到的模型设备分别记录。

只有一条小猫视频时，可先验证部分链路：

```bash
python3 -m ais_bench.tools.vabench.cli prepare \
  --source-root "$VABENCH_ROOT" --subset smoke \
  --sample-id 'kitten curled up meowing' \
  --videos /workspace/generated --output /workspace/prepared_kitten
python3 -m ais_bench.tools.vabench.cli run \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --prepared /workspace/prepared_kitten --device cuda:0 \
  --python /root/miniconda3/envs/vabench311/bin/python3 \
  --dimensions second_clap --output /workspace/vabench_runs/kitten_clap
```

单指标成功不会生成 15 项总分。约 4 秒的小猫视频可能使 Synchformer 返回异常兜底值，无人脸时 LSA 可能省略结果；执行结束不等于 15 项均有效。

### 使用 AISBench 主命令

在主环境中安装 AISBench，编辑 `ais_bench/configs/vabench_examples/eval_vabench.py`：

| 配置 | 含义 |
| --- | --- |
| `PREPARED_DIR` | 上一步的准备目录 |
| `SOURCE_ROOT`、`CACHE_DIR` | 固定官方源码、模型缓存 |
| `BACKEND_PYTHON` | 评分 Conda 的 `bin/python3` 绝对路径 |
| `DEVICE` | 评分进程中的逻辑设备，例如 `cuda:0` |
| `NUM_DEVICES` | LocalRunner 为该任务保留的卡数，按实际分配填写 |

```bash
ais_bench ais_bench/configs/vabench_examples/eval_vabench.py -m eval
```

`max_num_workers` 控制独立任务数，不表示同一 VABench 数据集的视频并发。当前接入没有实现视频分片并行、结果自动合并或断点续评。最新看板按完成指标组数/总组数显示，实际视频数另存为 sample_count，并非逐视频实时完成数或有效覆盖率。

`--num-prompts N` 会截取 prepared manifest 的前 N 条，等价于 `reader_cfg.test_range='[:N]'`；报告 scope 仍是 `custom`，不会变成完整集。manifest 按类别排序，前 N 条不一定有人脸，`second_lsa` 可能因此 missing。默认完成度策略将纯 missing 视为跳过，不再让任务以非零码退出；但报告仍将其标为 `incomplete`。若 15 项均在请求范围内，缺失项按 0 计入，显示预览总分；只评测部分指标时不生成 15 项总分。`failed`、`backend_error`、`not_evaluated` 及未解释的样本缺口仍会导致任务失败。需要非前缀样本时，在配置中显式设置 `reader_cfg=dict(test_range='[4:5]')`。

CLI 的结果目录就是 `--output`，例如 `/workspace/vabench_runs/mini10pct_simple`。主命令的输出由运行器工作目录控制，以任务日志给出的结果位置为准。

## 5. 查看 15 项分数和总分

结果位于 `--output` 指定目录：

```bash
cat /workspace/vabench_runs/mini10pct_simple/summary.md
```

| 文件 | 内容 |
| --- | --- |
| `summary.md` | 可阅读的 15 项指标及总分 |
| `metrics.csv`、`total.csv` | 原始均分、展示分、有效数、失败/缺失数、覆盖率和总分状态 |
| `samples.csv`、`scenes.csv` | 逐题状态及各场景均分 |
| `report.json` | 原始记录、汇总、输入/源码/权重/参数指纹及设备证据 |
| `official_eval_results/<category>/` | 官方格式的原始 JSON、CSV、full_info 备份 |
| `job_*/process.log`、`job_*/<dimension>/<category>/backend.log` | 子进程及评分日志 |

对每个指标，先将所有成功视频的原始分等权平均，再按下表转换为 0–100 展示分，并限制在该范围。原始值和超界信息保留；不会先裁剪每条分数再平均。

| 指标 | 原始方向 | 展示公式，m 为有效视频原始均分 |
| --- | --- | --- |
| `first_dnsmos`、`first_nisqa` | 越高越好 | `(m-1)/4*100` |
| `first_audiobox` | 官方 `(CE+CU+PQ-PC)/4` 越高越好 | `(m+1.75)/9*100` |
| `second_viclip`、`second_clap`、`second_imagebind` | 余弦相似度越高越好 | `(m+1)/2*100` |
| `second_desync` | 绝对偏移秒数越小越好 | `(1-m/2)*100` |
| `second_lsa` | 同步置信度越高越好 | `m/10*100` |
| `third_alignment`、`third_audio_reality`、`third_visual_reality`、`third_expressiveness`、`third_artistry` | 1–5 越高越好 | `(m-1)/4*100` |
| `fourth_qa_audio`、`fourth_qa_vision` | 官方正向极性比例越高越好 | `m*100` |

MOS 的 1/5、AudioBox 的 −1.75/7.25、LSA 的 0/10 是固定展示锚点，尤其 LSA 的 10 不宣称为模型硬上限。

总分为 15 项展示分等权平均，每项占 1/15，分母固定为 15：

- 15 项均在请求范围内但任一指标无有效均分：缺失项按 0 计入，显示预览总分；`report.json` 和 `total.csv` 记录 `zero_filled_dimensions`。
- 15 项都有分，但存在失败、缺失、LSA 省略或未解决的后端错误：展示预览总分，状态不完整。
- 15 项均有分且应评样本覆盖率 100%、无已记录后端错误：展示当前范围的正式总分。
- 只评测部分指标（例如 smoke 只跑 `second_clap`）时不生成 15 项总分，最终摘要不显示 total 行。

LSA 沿用官方过滤逻辑，仅有效记录参与均值；无脸、零置信度或内部异常可能被省略，不能可靠区分原因，因此展示有效数/输入数及省略数，不伪造零分。省略数是 missing 的子集，不能相加重复计算。自定义 78 题的完整总分仅代表该子集，并非 778 标准集成绩或官方对齐证明。

## 6. 官方原生评测与 AISBench 对照

本节结合 [官方测试操作记录](../../get_started/vabench_official_test.md)，以固定源码的 `evaluate.py` 实际参数为准。

| 项目 | 官方 `evaluate.py` | AISBench 接入 |
| --- | --- | --- |
| 输入 | `data/video/<类别>/<idx>.mp4`、`audio/<类别>/<idx>.wav`、`json/<类别>.json` | 准备目录 manifest、同源 WAV 与实际题目元数据 |
| 执行 | 按场景、指标顺序调用，异常可能中断 | 独立子进程隔离专家，Qwen 组复用，保存失败和部分结果 |
| 输出 | 每场景原始 `*_eval_results.json/csv` 和 `*_full_info.json` | 原始备份，加逐样本状态、均分、展示分、总分 |
| 总分 | 该源码不提供本文约定的 15 项归一化总分 | 使用第 5 节项目协议 |
| 适用用途 | 独立官方基线 | 编排、追踪、汇总及对齐验证 |

### 为官方 CLI 准备相同输入

可以复用第 3 节已经核验的 78 题媒体准备结果，生成官方需要的目录布局。下面只创建新目录和软链、复制 JSON，不修改官方评分源码，也不重新抽取音频。`prepared` 改为本次待对齐的准备目录：

```bash
cd /workspace/code/benchmark
python3 - <<'PY'
import json
import os
import shutil
from pathlib import Path
from ais_bench.benchmark.utils.vabench.data import verify_prepared

prepared = Path('/workspace/prepared_mini10pct')
root = Path('/workspace/official_mini10pct')
source = Path(os.environ['VABENCH_ROOT'])
manifest, _ = verify_prepared(prepared)
root.mkdir(parents=True, exist_ok=False)
for row in manifest['records']:
    for media, suffix in [('video', '.mp4'), ('audio', '.wav')]:
        target = root / 'data' / media / row['category'] / (row['idx'] + suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(Path(row[media + '_path']).resolve())
shutil.copytree(prepared / 'json', root / 'data/json')
(root / 'mapping_text').mkdir()
(root / 'mapping_text/final_idx_to_prompt.csv').symlink_to(
    source / 'mapping/final_idx_to_prompt.csv')
(root / 'input_manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(root)
PY
```

官方实现从**运行目录**读取 `mapping_text/final_idx_to_prompt.csv`，因此要进入上述目录。缓存变量必须在 Python 导入官方模块前设置。使用相同评分 Conda 执行：

```bash
conda activate vabench311
export VABENCH_CACHE_DIR=/workspace/vabench-cache
export CUDA_VISIBLE_DEVICES=6,7
cd /workspace/official_mini10pct
python3 - <<'PY'
import os
from pathlib import Path
import yaml

source = Path(os.environ['VABENCH_ROOT']) / 'vabench/config.yaml'
cfg = yaml.safe_load(source.read_text(encoding='utf-8'))
# 与本文 AISBench options 的 device_map:auto 一致；只写运行目录配置副本。
cfg['params'].pop('max_memory', None)
Path('config_aligned.yaml').write_text(
    yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
PY
set -o pipefail
python3 -u /workspace/code/VABench-main/evaluate.py \
  --data_dir /workspace/official_mini10pct/data \
  --config /workspace/official_mini10pct/config_aligned.yaml \
  --output_dir /workspace/official_mini10pct/results \
  2>&1 | tee /workspace/official_mini10pct/official.log
```

官方原配置含全部 15 项及 `device_map: auto`、逻辑卡 0/1 的 `max_memory`。本指南的 AISBench 简化命令不传 `--options`，Qwen 加载器默认使用 `device_map: auto`；上面生成的官方运行配置副本因此去掉显存上限，使实际加载策略一致，官方源码文件不变。请确认实际可用显存适合自动分配。若要沿用双卡 80GB 上限，须另行统一两边配置，并验证参数经过子进程 JSON 传递后仍保留 GPU 编号语义；当前入口没有数字键转换，不要直接假设 JSON 的字符串键等价于 Python 整数 GPU 编号。

检查每个场景是否都有 15 项原始结果、样本数是否一致及日志是否有异常。此版本官方入口捕获 fatal error 后未显式返回非零状态，不能单靠 shell 退出码判成功。CSV 的 `-1` 表示未取得该项成绩，不能参与均分。

将官方 `results/<category>/*_eval_results.json` 的逐题原始记录，与 AISBench 的 `official_eval_results` 逐项核对，再比较同一有效样本集合的原始均分。官方数值与剔除失败后的有效均分可能不同，需结合日志核验异常和 LSA 省略集合。不得拿归一化总分代替逐指标官方对齐。

验收口径：原始均分的相对误差严格 `<1%`；官方均分绝对值 `≤1e-6` 时，改为绝对差 `≤1e-6`。输入、权重、源码、预处理、参数和成功/失败/省略集合必须一致。CLAP 长音频存在随机截断，应核对实际缓存预处理配置并控制随机状态/裁剪条件；无法控制时保留波动证据，不能自动放宽上述验收口径。

AISBench 的 `--mode official` 是带资源注入和路径修正的官方类参考路径，**不等于独立运行上述官方 CLI**。`cli compare` 读取两份 AISBench `report.json`，用于适配器与参考路径对照；`cli compare-official` 可直接读取原生 `evaluate.py` 输出和 AISBench `official_eval_results`，按类别、指标、样本对齐并比较 `video_results`，打印缺失项和数值差异。后者仍需逐项核对输入、权重、源码和随机状态；CLAP 随机截断产生的差异不能据此宣称运行失败或对齐完成。更详细限制见 [接入说明](../../get_started/vabench_t2av.md)。

## 7. 测试与设计资料

- [VABench 需求与设计](../../design/vabench-benchmark-design.md)：范围、架构、目录分析、评分协议、验收和测试设计。
- UT 命令见设计文档第 4 节，源码内开发说明位于 `tests/UT/vabench/README.md`；测试包含子进程覆盖率收集，模型使用模拟实现，不能替代 GPU 评分验收。
- [官方源码](https://github.com/tanABCC/VABench) 与 [模型缓存](https://huggingface.co/datasets/tanABBCC/VABENCH_CACHE_DIR)：评分算法、题目和缓存来源。
