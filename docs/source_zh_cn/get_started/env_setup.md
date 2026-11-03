# VABench + AISBench（`vabench311_new`）精简步骤

> 代码版本要求：下列custom命令和配置模板需要后续配套代码。本次仅提交文档；只拉取本次文档提交或10e1469代码不能获得这些新增功能。

> 状态：以下是7条视频现场已跑通的专项统一环境记录，保留已知Pillow/fsspec/tqdm依赖冲突，不作为通用默认安装方案。默认依赖一致的方案见[vabench_gpu_environment.md](vabench_gpu_environment.md)，AISBench主环境通过BACKEND_PYTHON调用同一个15项评分Conda。若复用统一环境，必须保存最终freeze、pip check输出及原始评分报告，不能把pip check失败改记为通过。

> 只保留依赖下载和执行命令。环境已经存在时，从第 5 步开始。

## 1. 创建 Conda 环境

```bash
/root/miniconda3/bin/conda create -n vabench311_new python=3.11 pip -y
```

## 2. 安装系统依赖

```bash
apt-get update
apt-get install -y git ffmpeg libsndfile1 build-essential ca-certificates
```

## 3. 下载并安装 VABench 依赖

```bash
cd /workspace/code/benchmark

/root/miniconda3/envs/vabench311_new/bin/python3 \
  -m ais_bench.tools.vabench.environment install \
  --source-root /workspace/code/VABench-main \
  --cache-dir /workspace/vabench-cache \
  --work-dir /workspace/vabench-env-audit \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple
```

## 4. 下载并安装 AISBench 依赖

```bash
cd /workspace/code/benchmark

sed -e 's/^gradio-client$/gradio-client==1.14.0/' \
    -e 's/^openai$/openai==1.109.1/' \
    requirements/runtime.txt \
  | grep -viE '^\s*Pillow\s*==' \
  > /workspace/aisbench_runtime_unified.txt

printf 'anyio==4.10.0\n' >> /workspace/aisbench_runtime_unified.txt

/root/miniconda3/envs/vabench311_new/bin/python3 -m pip install \
  -r /workspace/aisbench_runtime_unified.txt \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple

/root/miniconda3/envs/vabench311_new/bin/python3 -m pip install \
  -e /workspace/code/benchmark \
  --no-deps \
  --no-build-isolation

/root/miniconda3/envs/vabench311_new/bin/python3 -m pip install \
  "fsspec==2025.9.0" "tqdm==4.66.2" \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple
```

## 5. 准备评测数据

已有 `/workspace/prepared_all_official_v2/manifest.json` 时跳过本步。评测配置 `eval_vabench_custom_all.py` 的 `PREPARED_DIR` 当前指向该目录，两边必须一致。

```bash
cd /workspace/code/benchmark

/root/miniconda3/envs/vabench311_new/bin/python3 \
  -m ais_bench.tools.vabench.cli prepare \
  --source-root /workspace/code/VABench-main \
  --subset custom \
  --videos /workspace/video \
  --output /workspace/prepared_all_official_v2
```

## 6. 执行评测

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

## 7. 查看结果

使用第 6 步最后打印的 `WORKDIR`；如果换了 shell，取最新目录：

```bash
WORKDIR=$(ls -dt /workspace/vabench_runs/aisbench/custom_all_7_* | head -1)
REPORT=$(find "$WORKDIR" -path '*/vabench_t2av_artifacts/report.json' | sort | tail -1)

/root/miniconda3/envs/vabench311_new/bin/python3 - "$REPORT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
print(json.dumps(report['total'], ensure_ascii=False, indent=2))
PY
```

官方格式备份位于：

```text
<artifacts>/official_eval_results/<category>/
  <category>_<时间>_eval_results.json
  <category>_<时间>_eval_results.csv
  <category>_<时间>_full_info.json
```

```bash
find "$WORKDIR" -path '*/official_eval_results/*/*_eval_results.csv'
```

`second_clap` 对超过 10 秒的音频使用官方随机 fusion 截断，重复运行会有波动；短于 10 秒的样本（如小猫）稳定。不要按逐样本完全相等验收，必要时多次运行取均值。

### 固定现场审计信息

在已跑通的统一Conda中执行并保存（这不会消除已知冲突）：

```bash
python3 -m pip freeze --all > /workspace/vabench-env-audit/installed.freeze.unified.txt
python3 -m pip check > /workspace/vabench-env-audit/pip-check.unified.txt 2>&1
```

进度看板单位为“指标组”：8个专家组加1个共享Qwen组，共9组；不再把组比例换算成已完成视频数。配置模板已随仓库交付，运行前填写模板中的路径与GPU卡数。
