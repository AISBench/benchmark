# SWE-Bench Pro 使用指南

SWE-Bench Pro 是一个用于评估大语言模型在长时域软件工程任务上表现的基准测试。给定一个代码库和一个问题，语言模型的任务是生成一个补丁来解决所描述的问题。

> **注意**：由于官方提供的 Docker 镜像均为 x86 架构，SWE-bench Pro 目前仅支持在 x86 环境上评测，暂不支持 ARM 环境。

## 1. 功能概览

当前在 `ais_bench` 已接入以下 SWE-Bench Pro 能力：

- 数据集：`full`、`mini`
- 任务：
  - `infer`：调用 `mini-swe-agent` 生成补丁（`model_patch`）
  - `eval`：调用 SWE-bench Pro harness 执行评测并统计 resolved
- 结果汇总：输出 `accuracy`、`eval_resolved_instances_num` 等关键指标

`ais_bench/configs/swe_bench_pro_examples/`目录已提供以下示例配置：

- `mini_swe_agent_swe_bench_pro_mini.py`：SWE-bench Pro Mini，适合先跑通流程/快速迭代。
- `mini_swe_agent_swe_bench_pro_full.py`：SWE-bench Pro Full，完整测试集。

## 2. 运行环境安装

### 2.1 源码安装
1. 确认docker版本满足要求，执行`docker version`，确保docker版本为20.10.0及以上， docker API版本为 1.42及以上
2. 参考[工具安装&卸载](../../get_started/install.html)源码安装AISBench测评工具
3. 安装 `mini-swe-agent`（infer 依赖）

> **注意**：SWE-Bench Pro 官方组织 scaleapi 对 mini-swe-agent 做了适配，需从 scaleapi 的仓库下载适配版本。

```bash
git clone https://github.com/scaleapi/mini-swe-agent.git
cd mini-swe-agent
pip install -e .
cd -
```

4. 安装 `SWE-Bench_Pro`（infer 和 eval 依赖）

```bash
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git
cd SWE-bench_Pro-os
pip install -r requirements.txt
cd -
```

5. SWEBP 脚本与 Docker 目录配置

SWE-Bench Pro 评测**必须**指定以下两个路径，无默认处理动作：

- `swebp_scripts_dir`：SWE-bench Pro 官方仓库的 `run_scripts` 目录绝对路径
- `swebp_docker_dir`：SWE-bench Pro 官方仓库的 `dockerfiles` 目录绝对路径

```python
SWEBP_SCRIPT_PATH_ABS = "{your_work_dir}/SWE-bench_Pro-os/run_scripts"  # 必须指定
SWEBP_DOCKER_PATH_ABS = "{your_work_dir}/SWE-bench_Pro-os/dockerfiles"  # 必须指定
```

> **注意**：需提前克隆 SWE-bench Pro 官方仓库：`git clone https://github.com/scaleapi/SWE-bench_Pro-os.git`



### 2.2 一键准备方案（推荐）
如果你不想源码安装依赖，或者docker版本较低（docker version < 20.10.0），推荐使用 **AISBench Agent Runtime 一键准备方案**：

```bash
# 1. 物理机上准备 mini 数据集（已有可跳过）
#    SWE-bench Pro 的 mini 数据集**必须**本地准备，无在线版
mkdir -p /data/datasets/swebench_pro
# 下载地址：https://modelers.cn/datasets/AISBench/SWE-Bench_Pro_mini
# 优先 parquet 格式；下载后解压到 /data/datasets/swebench_pro/ 即可

# 2. 物理机上一键起 runtime 容器（自动选 DinD/Socket 模式，自动挂载数据集，自动把 case 镜像 tar 拷进容器内部 docker load 完）
#    在线场景：省略 --runtime-tar，runtime 镜像自动从 ghcr.io 拉取最新
#    离线场景：通过 --runtime-tar 跳过外网拉取，可以从最新的release信息中提前获取
curl -fsSL https://aisbench.obs.cn-north-4.myhuaweicloud.com/agent/ais_bench_agent_bootstrap.sh \
    | bash -s -- \
        --runtime-tar /path/to/agent_runtime_image_v3.1-20260701-master-ubuntu24.04-py312-<arch>.tar.gz \
        --datasets /data/datasets/swebench_pro \
        --container-name ais_bench_agent
# --runtime-tar （可选）提前准备的测评镜像，不传则自动拉取最新
# --datasets 指向本地数据集目录，容器内会自动创建同名目录挂载数据集
# --container-name 指向的容器名需唯一，否则会覆盖旧容器

# 3. 进入容器
docker exec -it ais_bench_agent bash

# 4. 验证 runtime 就绪
ais_bench_agent_doctor.sh swebench_pro

# 5. 激活 swebench_pro venv（scaleapi fork 的 mini-swe-agent）
agent_env swebench_pro

```
> 该一键准备方案的方案原理与脚本实现和更详细的介绍见 [`docker/agent_runtime/`](https://github.com/AISBench/benchmark/tree/master/docker/agent_runtime/README.md)。

## 3. 最小配置（先跑通再调优）

建议从 `mini_swe_agent_swe_bench_pro_mini.py` 开始，仅改 `models[0]` 里的三个字段：

- `model`：模型名（必填）
- `url`：模型服务地址（OpenAI 兼容 API）
- `api_key`：服务密钥（本地服务可用 `EMPTY`）

示例（本地 vLLM 场景）：

```python
models = [
    dict(
        attr="local",
        abbr="swebench_pro_mini_module",
        type="LiteLLMChat",
        model="qwen3",
        api_key="EMPTY",
        url="http://127.0.0.1:8000/v1",
        batch_size=4,
        generation_kwargs=dict(),
    )
]
```

### 数据集路径说明

不同数据集的加载方式有所区别：

- **full 数据集**：支持从 Hugging Face 在线获取和本地加载两种方式
  - 在线加载：保持 `path=""`
  - 本地加载：将 `path` 指向本地 parquet 文件或目录

- **mini 数据集**：**必须提前在本地准备好**，无法在线获取
  - 下载地址：`https://modelers.cn/datasets/AISBench/SWE-Bench_Pro_mini`
  - 优先推荐的数据格式为 parquet
  - 将 `path` 指向本地下载的 parquet 文件或目录

### 首跑建议

- 数据集先用 `mini`
- `batch_size=4`（每道题会创建一个容器，过大的 batch_size 容易导致宿主机 OOM）
- `step_limit=250`（示例默认值，先不改）

## 4. 运行命令

在仓库根目录执行（`config` 为配置文件路径）：

```bash
ais_bench ais_bench/configs/swe_bench_pro_examples/mini_swe_agent_swe_bench_pro_mini.py
```

上面是完整流程（`all`），也可以分步执行：

```bash
# 只做推理，生成 predictions
ais_bench ais_bench/configs/swe_bench_pro_examples/mini_swe_agent_swe_bench_pro_mini.py -m infer

# 基于现有 predictions 做评测
ais_bench ais_bench/configs/swe_bench_pro_examples/mini_swe_agent_swe_bench_pro_mini.py -m eval
```

### 断点续跑

使用 `--reuse` 跳过已完成实例，适合中断后续跑：

```bash
ais_bench ais_bench/configs/swe_bench_pro_examples/mini_swe_agent_swe_bench_pro_mini.py -m infer --reuse
```

## 5. 输出结果怎么看

默认输出目录为 `outputs/default/<timestamp>/`，目录结构如下：

### 推理结果（predictions）

```
├── predictions
│   └── swebench_pro_mini_model
│       ├── swebench_pro_mini_data
│       │   ├── exit_statuses.yaml     # agent全量题目退出状态统计
│       │   │
│       │   ├── instance_gravitational__teleport-xxx    # 题目xxx目录，记录推理运行中信息
│       │   │   ├── instance_gravitational__teleport-xxx.config.yaml
│       │   │   ├── instance_gravitational__teleport-xxx.debug.log
│       │   │   ├── instance_gravitational__teleport-xxx.info.log
│       │   │   ├── instance_gravitational__teleport-xxx.pred
│       │   │   └── instance_gravitational__teleport-xxx.traj.json
│       │   │
│       │   └── instance_qutebrowser__qutebrowser-yyy   # 题目yyy目录，记录推理运行中信息
│       │       ├── instance_qutebrowser__qutebrowser-yyy.config.yaml
│       │       ├── instance_qutebrowser__qutebrowser-yyy.debug.log
│       │       ├── instance_qutebrowser__qutebrowser-yyy.info.log
│       │       ├── instance_qutebrowser__qutebrowser-yyy.pred
│       │       └── instance_qutebrowser__qutebrowser-yyy.traj.json
│       │
│       └── swebench_pro_mini_data.json    # 推理最终结果
```

### 评测结果（results）

```
├── results
│   ├── swebench_pro_mini_model
│   │   ├── instance_gravitational__teleport-xxx      # 题目xxx目录，记录评测运行中信息
│   │   │   ├── swebench_pro_mini_data_entryscript.sh
│   │   │   ├── swebench_pro_mini_data_output.json
│   │   │   ├── swebench_pro_mini_data_patch.diff
│   │   │   ├── swebench_pro_mini_data_stderr.log
│   │   │   ├── swebench_pro_mini_data_stdout.log
│   │   │   └── workspace
│   │   │
│   │   └── instance_qutebrowser__qutebrowser-yyy      # 题目yyy目录，记录评测运行中信息
│   │       ├── swebench_pro_mini_data_entryscript.sh
│   │       ├── swebench_pro_mini_data_output.json
│   │       ├── swebench_pro_mini_data_patch.diff
│   │       ├── swebench_pro_mini_data_stderr.log
│   │       ├── swebench_pro_mini_data_stdout.log
│   │       └── workspace
│   │
│   └── swebench_pro_mini_model_swebench_pro_mini_data_report.json   # 评测最终结果
```

### 评测报告关键字段说明

```json
{
  "total_instances_num": 2,   // 数据集元素实例数量
  "total_prediction_num": 2,  // 推理结果数量
  "build_patch_instances_num": 2,   // 成功在约束轮次内完成推理并生成patch的实例数量
  "empty_patch_instances_num": 0,   // 未在约束轮次内完成推理，生成patch为空的实例数量
  "eval_resolved_instances_num": 1,  // 评测结果为"已解决原始问题"的实例数量
  "eval_unresolved_instances_num": 1,  // 评测结果为"未解决原始问题"的实例数量
  "empty_patch_instances_ids": [],   // 未在约束轮次内完成推理的实例ID
  "unresolved_instances_ids": [
    "instance_gravitational__teleport-xxx"   // 评测结果为"未解决原始问题"的实例ID
  ],
  "accuracy": 50.0   // 最终评测精度
}
```

## 6. 常见问题与排障（SWEBP 错误码）

以下错误码来自 `SWEBP_CODES`，可结合全量 FAQ 查看：

- FAQ：`docs/source_zh_cn/faqs/error_codes.md`

### 1) `SWEBP-DEPENDENCY-001`：缺少 mini-swe-agent

- 现象：infer 启动时报依赖导入失败
- 原因：未安装 `mini-swe-agent`
- 处理：按照“前置依赖”章节中的说明，从 scaleapi 的仓库克隆并安装适配版本的 mini-swe-agent

### 2) `SWEBP-PARAM-001`：模型配置为空

- 现象：提示 model 未配置
- 原因：`models[0]['model']` 为空或仅空白字符
- 处理：配置 `model/url/api_key`，至少保证 `model` 非空

### 3) `SWEBP-PARAM-002`：数据集名称非法

- 现象：提示数据集名称不受支持
- 原因：`name` 不在 `full`、`mini` 支持范围内
- 处理：将数据集 `name` 修正为受支持值

### 4) `SWEBP-DATA-001`：数据集加载失败

- 现象：在线加载失败或本地文件未找到
- 原因：
  - 在线模式：网络或 Hugging Face 访问异常（仅 full 数据集支持）
  - 本地模式：`path` 不存在，或文件格式不支持
- 处理：
  - full 数据集在线失败时可改为本地 parquet 文件
  - mini 数据集需确保已从 `https://modelers.cn/datasets/AISBench/SWE-Bench_Pro_mini` 下载

### 5) `SWEBP-FILE-001`：找不到 predictions 文件

- 现象：`-m eval` 时提示 predictions 不存在
- 原因：未先执行 infer，或 work_dir/reuse 指向不一致
- 处理：先执行 `-m infer`，并确认 eval 与 infer 使用同一份配置/输出目录

### 6) `SWEBP-RUNTIME-001` / `SWEBP-RUNTIME-002`：容器或 harness 运行失败

- 现象：Docker 镜像拉取失败、评测执行异常
- 原因：镜像不可用、网络异常、容器运行环境不满足、宿主机内存不足导致 OOM
- 处理：
  - 先确认 `docker ps` 正常
  - 检查镜像是否可拉取，确保可以从 DockerHub 拉取镜像
  - 若宿主机内存不足，降低 `batch_size`（建议不超过 4）
  - 失败后可用 `--reuse` 重试，避免重复计算已完成实例

## 7. 进阶建议（可选）

- 初次调试优先 `mini`，确认流程稳定后切 `full`
- 需要减少空补丁时，优先优化模型能力与 agent 提示模板
- 评测时关注 `empty_patch_instances_ids` 和 `unresolved_instances_ids`，它们通常比 `accuracy` 更能定位首轮问题
- SWE-Bench Pro 使用 Docker 镜像执行评测，建议确保网络稳定以加快镜像拉取速度
- 注意控制 `batch_size`，避免宿主机因内存不足导致容器退出
