# 工具安装&卸载
## 🔧 工具安装
✅ 环境要求

**Python 版本**：仅支持 Python **3.10**、 **3.11** 或 **3.12**

不支持 Python 3.9 及以下版本，也不兼容 Python 3.13 及以上版本

**推荐使用 Conda 管理环境**，以避免依赖冲突
```shell
conda create --name ais_bench python=3.10 -y
conda activate ais_bench
```

📦 安装方式-源码安装（首选）

AISBench 当前推荐使用源码安装方式，以便获得更好的自定义配置文件使用体验，请确保安装环境联网：
```shell
git clone https://github.com/AISBench/benchmark.git
cd benchmark/
pip3 install -e ./ --use-pep517
```
该命令会自动安装核心依赖。
执行`ais_bench -h`，如果打印出AISBench评测工具的所有命令行的帮助信息，说明安装成功

⚙️ 服务化框架支持（可选）

若需评估服务化模型（如 vLLM、Triton 等），需额外安装相关依赖：
```shell
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```

⚙️ Huggingface多模态模型/vllm多模态离线推理支持（可选）

```shell
pip3 install -r requirements/hf_vl_dependency.txt
```

🔗 Berkeley Function Calling Leaderboard (BFCL) 测评支持

```shell
pip3 install -r requirements/datasets/bfcl_dependencies.txt --no-deps
```

**重要提示**：由于 `bfcl_eval` 会自动安装 `pathlib` 库，而 Python 3.5+ 环境已内置该库，为避免版本冲突，请务必使用 `--no-deps` 参数跳过额外依赖的自动安装。

🔗 OCRBench_v2数据集测评支持（可选）

```shell
pip3 install -r requirements/datasets/ocrbench_v2.txt
```

📦 安装方式-一键安装（备选）

AISBench 也提供了一键安装方式，适用于基于预置配置文件的快速体验和评估场景，请确保安装环境联网。
- 基本功能的安装命令如下：
```shell
pip3 install ais_bench_benchmark
```
- 全量功能的安装命令如下：
```shell
pip3 install ais_bench_benchmark[full]
```

## ❌ 工具卸载
如需卸载 AISBench Benchmark，可执行以下命令：
```shell
pip3 uninstall ais_bench_benchmark
```