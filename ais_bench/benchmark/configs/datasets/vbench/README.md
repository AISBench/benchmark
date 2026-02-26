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

## 与现有流程的兼容

- CLI 不变：仍使用 `--models`、`--datasets`（及可选 `--mode eval`、`--work_dir` 等）。
- 当所选数据集中包含 VBench 数据集（`eval_cfg.use_vbench_task=True` 或 `type=VBenchDataset`）时，Eval 阶段会自动使用 `VBenchEvalTask`，在 GPU 或 NPU 上跑 VBench 1.0 测评。
