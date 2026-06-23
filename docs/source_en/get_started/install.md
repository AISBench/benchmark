# Tool Installation & Uninstallation
## 🔧 Tool Installation
✅ Environment Requirements

**Python Version**: Only Python **3.10**, **3.11**, or **3.12** is supported.

Python 3.9 and lower versions are not supported, nor are Python 3.13 and higher versions compatible.

**It is recommended to use Conda for environment management** to avoid dependency conflicts.
```shell
conda create --name ais_bench python=3.10 -y
conda activate ais_bench
```

📦 Installation Method - Source Code Installation (Preferred)

Currently, AISBench recommends the source code installation method for a better custom configuration file experience. Ensure the installation environment has internet access:
```shell
git clone https://github.com/AISBench/benchmark.git
cd benchmark/
pip3 install -e ./ --use-pep517
```
This command will automatically install core dependencies.
Execute `ais_bench -h`. If all command-line help information for the AISBench evaluation tool is printed, the installation is successful.

⚙️ Service-Oriented Framework Support (Optional)

If you need to evaluate service-oriented models (such as vLLM, Triton, etc.), you need to install additional relevant dependencies:
```shell
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```

⚙️ Huggingface Multi-modal Model / vLLM Multi-modal Offline Inference Support (Optional)

```shell
pip3 install -r requirements/hf_vl_dependency.txt
```

🔗 Berkeley Function Calling Leaderboard (BFCL) Evaluation Support

```shell
pip3 install -r requirements/datasets/bfcl_dependencies.txt --no-deps
```

**Important Note**: Since `bfcl_eval` will automatically install the `pathlib` library, and the Python 3.5+ environment already has this library built-in, be sure to use the `--no-deps` parameter to skip the automatic installation of additional dependencies and avoid version conflicts.

🔗 OCRBench_v2 Dataset Evaluation Support (Optional)

```shell
pip3 install -r requirements/datasets/ocrbench_v2.txt
```

📦 Installation Method - One-Click Install (Alternative)

AISBench also provides a one-click installation method, suitable for quick experience and evaluation scenarios based on preset configuration files. Ensure the installation environment has internet access.
- Basic functionality installation command:
```shell
pip3 install ais_bench_benchmark
```
- Full functionality installation command:
```shell
pip3 install ais_bench_benchmark[full]
```

## ❌ Tool Uninstallation
If you need to uninstall AISBench Benchmark, you can execute the following command:
```shell
pip3 uninstall ais_bench_benchmark
```