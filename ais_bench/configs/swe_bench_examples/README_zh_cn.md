# SWEbench 使用指南

SWE-bench是一个基准测试，用于评估大语言模型在从GitHub收集的现实世界软件问题上的表现。给定一个代码库和一个问题，语言模型的任务是生成一个补丁来解决所描述的问题。

## 1. 功能概览

当前在 `ais_bench` 已接入以下 SWEbench 能力：

- 数据集：`full`、`verified`、`lite`、`multilingual`
- 任务：
  - `infer`：调用 `mini-swe-agent` 生成补丁（`model_patch`）
  - `eval`：调用 SWE-bench harness 执行评测并统计 resolved
- 结果汇总：输出 `accuracy`、`submitted_accuracy`、`resolved_instances` 等关键指标

本目录已提供 4 个示例配置：

- `mini_swe_agent_swe_bench_lite.py`
- `mini_swe_agent_swe_bench_verified.py`
- `mini_swe_agent_swe_bench_full.py`
- `mini_swe_agent_swe_bench_multilingual.py`

## 2. 前置依赖

运行前请确保以下依赖可用：

1) 安装 `mini-swe-agent`（infer 依赖）

```bash
pip install mini-swe-agent
```

2) 安装 SWE-bench harness（eval 依赖）

```bash
git clone https://github.com/SWE-bench/SWE-bench.git
cd SWE-bench
pip install -e .
cd -
```

3) Docker 可用（infer/eval 都依赖容器环境）

```bash
docker --version
docker ps
```

## 3. 最小配置（先跑通再调优）

建议从 `mini_swe_agent_swe_bench_lite.py` 开始，仅改 `models[0]` 里的三个字段：

- `model`：模型名（必填）
- `url`：模型服务地址（OpenAI 兼容 API）
- `api_key`：服务密钥（本地服务可用 `EMPTY`）

示例（本地 vLLM 场景）：

```python
models = [
    dict(
        attr="local",
        abbr="swebench",
        type="LiteLLMChat",
        model="qwen3",
        api_key="EMPTY",
        url="http://127.0.0.1:8000/v1",
        batch_size=1,
        generation_kwargs=dict(),
    )
]
```

### 数据集路径说明

示例配置默认 `path=""`，表示优先从 Hugging Face 在线加载。

- 你可以保持 `path=""` 直接在线拉取
- 若离线使用，把 `path` 改为本地 parquet 文件或目录（支持 `data/<split>-*.parquet`）

### 首跑建议

- 数据集先用 `lite`
- `batch_size=1`
- `step_limit=200`（示例默认值，先不改）

## 4. 运行命令

在仓库根目录执行（`config` 为配置文件路径）：

```bash
ais_bench ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_lite.py
```

上面是完整流程（`all`），也可以分步执行：

```bash
# 只做推理，生成 predictions
ais_bench ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_lite.py -m infer

# 基于现有 predictions 做评测
ais_bench ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_lite.py -m eval
```

### 断点续跑

使用 `--reuse` 跳过已完成实例，适合中断后续跑：

```bash
ais_bench ais_bench/configs/swe_bench_examples/mini_swe_agent_swe_bench_lite.py -m infer --reuse
```

## 5. 输出结果怎么看

默认输出目录为 `outputs/default/<timestamp>/`，重点关注：

- 推理结果：
  - `predictions/swebench/swebench_*.json`
  - 每个 `instance_id` 下包含 `model_patch`
- 评测结果：
  - `results/swebench/swebench_*.json`
  - 关键字段：
    - `accuracy`：`resolved_instances / total_instances`
    - `submitted_accuracy`：`resolved_instances / submitted_instances`
    - `resolved_instances` / `unresolved_instances` / `error_instances`
    - `harness_exit_code`：harness 退出码

## 6. 常见问题与排障（SWEB 错误码）

以下错误码来自 `SWEB_CODES`，可结合全量 FAQ 查看：

- 中文 FAQ：`docs/source_zh_cn/faqs/error_codes.md`

### 1) `SWEB-DEPENDENCY-001`：缺少 mini-swe-agent

- 现象：infer 启动时报依赖导入失败
- 原因：未安装 `mini-swe-agent`
- 处理：执行 `pip install mini-swe-agent`

### 2) `SWEB-DEPENDENCY-002`：缺少 SWE-bench harness

- 现象：eval 阶段报 harness import error
- 原因：未安装 SWE-bench 项目或未在当前环境可见
- 处理：按“前置依赖”安装 SWE-bench，并确认当前 Python 环境一致

### 3) `SWEB-PARAM-001`：模型配置为空

- 现象：提示 model 未配置
- 原因：`models[0]['model']` 为空或仅空白字符
- 处理：配置 `model/url/api_key`，至少保证 `model` 非空

### 4) `SWEB-DATA-002` / `SWEB-FILE-003`：数据集加载失败

- 现象：在线加载失败或本地 parquet 未找到
- 原因：
  - 在线模式：网络或 Hugging Face 访问异常
  - 本地模式：`path` 不存在，或目录结构不匹配 split parquet 规则
- 处理：
  - 在线失败时可改为本地 parquet
  - 本地路径需满足：`<root>/data/test-*.parquet` 或 `<root>/test-*.parquet`

### 5) `SWEB-FILE-001`：找不到 predictions 文件

- 现象：`-m eval` 时提示 predictions 不存在
- 原因：未先执行 infer，或 work_dir/reuse 指向不一致
- 处理：先执行 `-m infer`，并确认 eval 与 infer 使用同一份配置/输出目录

### 6) `SWEB-RUNTIME-001` / `SWEB-RUNTIME-002`：容器或 harness 运行失败

- 现象：Docker 镜像拉取失败、评测执行异常
- 原因：镜像不可用、网络异常、容器运行环境不满足
- 处理：
  - 先确认 `docker ps` 正常
  - 检查镜像是否可拉取
  - 失败后可用 `--reuse` 重试，避免重复计算已完成实例

## 7. 进阶建议（可选）

- 初次调试优先 `lite`，确认流程稳定后切 `verified/full`
- 需要减少空补丁时，优先优化模型能力与 agent 提示模板
- 评测时关注 `empty_patch_instances`、`error_instances`，它们通常比 `accuracy` 更能定位首轮问题

