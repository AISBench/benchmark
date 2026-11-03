# VABench GPU评分环境：Conda Python 3.11

> 版本说明：10e1469已包含依赖安装工具；本文链接的统一环境记录及新增custom评测命令需要后续配套代码。本次提交仅更新文档，不包含这些代码改动。

本指南依据用户在H200容器中的最终统一环境记录（`过程/10.txt`），使用外部固定版本VABench，不修改其评分源码。15个指标共用一个评分Conda。AISBench主环境通过`BACKEND_PYTHON`或`--python`调用该Conda，无须把GPU评分依赖加入所有数据集的公共runtime。

> 如果需要把AISBench和VABench装进同一个Conda环境，使用已跑通的[vabench311_new统一环境指南](setup.md)和[精简步骤](env_setup.md)；本文保留独立评分环境方案及其依赖审计细节。

这里交付依赖声明、安装与环境检查；本地Windows上的测试不代表H200真实评分或官方<1%对齐已通过。记录里的4秒小猫仍可能导致desync返回异常兜底1.0，LSA无脸仍会省略；依赖安装不能解决这两项输入限制。只有完成逐指标结果检查，才能认定有15项有效成绩。

## 1. 依赖文件与安装边界

与`requirements/datasets/geometry3k.txt`一样，将数据集专用包放在可选依赖目录，不加入`setup.py`的通用安装依赖。

| 文件 | 用途 |
| --- | --- |
| `requirements/datasets/vabench.txt` | GPU评分PyPI固定清单，来源为官方requirement及10.txt的兼容性调整；包含Torch和NVIDIA运行库 |
| `requirements/datasets/vabench-build.txt` | 固定pip25.2、setuptools80.9.0、wheel0.45.1及NumPy/Cython，先安装以构建InsightFace/Git包 |
| `requirements/datasets/vabench-git.txt` | 固定ImageBind、PyTorchVideo、CLIP的完整提交，只在普通依赖之后以`--no-deps`安装 |
| `requirements/vabench.txt` | 旧路径兼容入口，转向新的PyPI清单；不是完整安装入口 |

不能只执行一次`pip install -r vabench.txt`就宣称15项依赖已完整：NISQA需要本地元数据补丁，Git包必须单独安装，外部DNSMOS/Synchformer代码和模型缓存也不属于PyPI依赖。

具体调整：

- Torch2.6.0／torchvision0.21.0／torchaudio2.6.0，保留官方CUDA12.4运行库固定版本。不把驱动显示的CUDA13.2当成必须安装的PyTorch CUDA版本。
- 只保留`opencv-python-headless==4.9.0.80`，移除另一套`opencv-python`，与NumPy1.26.4配套。
- NISQA2.0.post2仅把Torch、torchvision、SciPy、scikit-learn、Matplotlib的五个`==`改为原版本的`>=`，其余依赖及所有包代码不变。RECORD重新生成，原始wheel SHA256必须匹配PyPI官方文件。
- 保留Pillow10.2.0等其他NISQA约束；AISBench主runtime固定Pillow11.2.1，不能默认将两份完整依赖清单装在同一个环境。安装工具检测到已安装AISBench时拒绝改写它。
- ImageBind固定`53680b02d7e37b19b124fa37bae4b6c98c38f5be`，PyTorchVideo固定用户最终验证的`2ac6ed361309cb25144636ad82dd8bd6a4970e0c`。后者不能只凭显示版本0.1.5判断已安装正确，检查工具核对`direct_url.json`中的实际提交。
- 用户流程未固定CLIP，本交付将其固定为核查时的官方提交`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`，须在H200回归中一起记录。setuptools固定80.9.0保留`pkg_resources`，而不是以后随`<82`漂移。
- `certifi==2025.10.5`及构建工具固定值是此次新增的可复现性约束，不声称这些具体版本已出现在用户服务器freeze中。安装后保存实际`installed.freeze.txt`用于校准。

## 2. 系统工具与Conda

Linux x86_64、NVIDIA GPU容器，已有驱动可见性与完整模型缓存。下面路径按用户容器示例填写，已有环境可直接复用。

“完整FFmpeg”指提供`ffmpeg`、`ffprobe`并具备所需音视频处理能力的发行包；当前解码预检还使用lavfi生成小视频。编译工具主要为gcc、g++、make及C/C++开发头文件，用于构建InsightFace等源码扩展。无需用户自行编译FFmpeg。

Ubuntu/Debian容器可手动执行以下apt步骤，或在第4节安装命令中增加`--install-system-deps`由工具自动下载并安装这些系统包。自动模式要求容器内root，仅使用现有apt源，不修改源配置，也不下载编译FFmpeg源码。

```bash
apt-get update
apt-get install -y git ffmpeg libsndfile1 build-essential ca-certificates
nvidia-smi
ffmpeg -version
ffprobe -version

source /root/miniconda3/etc/profile.d/conda.sh
conda create -n vabench311 python=3.11 pip -y
conda activate vabench311
python3 --version
python3 -m pip --version
```

若已有`vabench311`，跳过创建步骤。不要卸载系统Python或在评分Conda中执行`pip install -e .`安装完整AISBench；AISBench在其主环境中安装即可。评分工具从源码目录运行，模块入口本身不要求mmengine等主环境依赖。

仅当机器使用自有CA时，沿用现有证书配置；不要把这组示例路径强加给其他服务器：

```bash
export PIP_CERT=/root/ca-bundle.crt
export SSL_CERT_FILE=/root/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/root/ca-bundle.crt
export GIT_SSL_CAINFO=/root/ca-bundle.crt
```

## 3. 源码与权重

已有固定版本源码和缓存时，设置对应实际路径，不重复克隆：

```bash
export VABENCH_ROOT=/workspace/code/VABench-main
export VABENCH_CACHE_DIR=/workspace/vabench-cache
export CUDA_VISIBLE_DEVICES=6,7
```

若重新获取源码：

```bash
git clone https://github.com/tanABCC/VABench.git /workspace/code/VABench-main
git -C /workspace/code/VABench-main checkout 12b9617a900a9eca1691da6b7115ce3d58f53a16
```

模型缓存需事先下载或复制，pip不会下载评分模型权重。官方资源为[固定版本模型缓存](https://huggingface.co/datasets/tanABBCC/VABENCH_CACHE_DIR/tree/aa0c3da973d86f2994e9c8860f28785ce0cc6152)。另外补齐`checkpoints/auxiliary/sfd_face.pth`及InsightFace的`models/buffalo_l`目录；来源见[VABench评测指南](vabench_t2av.md)。

安装工具仅在目标缺失时下载固定微软DNSMOS脚本、链接缓存附带的Synchformer源码，并在发现备用buffalo_l目录时补软链。既有文件不覆盖；它们的哈希及后续评分来源指纹仍须与官方基线核对。Synchformer缓存源码不存在时会在预检中报缺失，不擅自换成不匹配权重的最新版。

**先设置缓存环境变量，再导入官方模块**；部分路径在导入时即确定。

## 4. 预览与完整安装

在AISBench源码根目录，例如`/workspace/code/benchmark`，使用评分Conda的Python执行。`plan`仅打印参数列表，不联网、不安装、不生成输出目录：

```bash
cd /workspace/code/benchmark
python3 -m ais_bench.benchmark.vabench.environment plan \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --work-dir /workspace/vabench-env-audit
```

完整安装（普通依赖解析不绕过，只有固定Git包与已修补NISQA采用`--no-deps`）：

```bash
python3 -m ais_bench.benchmark.vabench.environment install \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --work-dir /workspace/vabench-env-audit \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple
```

省略`--index-url`即可沿用pip配置或默认源。PyPI镜像不会代理GitHub克隆。若失败，查看终端及`install.json`；脚本失败返回非零，不把中断后的缓存文件当成已安装成功。可重新运行，固定Git包与补丁NISQA会重装，主依赖继续由pip按清单处理。

若希望同一条命令连系统工具也自动准备，在Ubuntu/Debian容器内以root执行：

```bash
python3 -m ais_bench.benchmark.vabench.environment install \
  --install-system-deps \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --work-dir /workspace/vabench-env-audit \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple
```

它先执行`apt-get update`与系统包安装，再执行pip阶段。默认不带该参数时不运行apt；`plan --install-system-deps`只预览，不安装。`--index-url`仅用于pip，系统包仍从容器现有apt源下载。检查模式不能触发系统安装。

安装包含数GB的Torch/NVIDIA包，完整模型权重另计。若只需事先下载依赖，可在同一Linux/Python3.11目标环境中使用声明清单：

```bash
python3 -m pip download --no-build-isolation \
  -r requirements/datasets/vabench-build.txt \
  -r requirements/datasets/vabench.txt --dest /workspace/vabench-packages
python3 -m pip wheel --no-deps --no-build-isolation \
  -r requirements/datasets/vabench-git.txt --wheel-dir /workspace/vabench-packages
```

这两步应在构建工具已经安装后执行，下载结果中可能含源码包；将`/workspace/vabench-env-audit/original`和`patched`也保留。当前完整安装工具仍按在线固定提交执行，尚未交付一键离线安装；不把上述目录误称为已验证的离线镜像。

## 5. 环境检查与审计结果

安装自动执行检查，也可对用户已准备好的环境单独运行：

```bash
python3 -m ais_bench.benchmark.vabench.environment check \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --work-dir /workspace/vabench-env-audit \
  --video /workspace/video/kittencurledupmeowing.mp4
```

检查固定PyPI版本、Git提交、NISQA五项元数据补丁、15个官方入口导入、DNSMOS外部脚本、ImageBind/PyTorchVideo兼容导入、微型CUDA与torchvision NMS、decord实际解码、系统工具、关键资源及Qwen分片。检查阶段离线，不调用评分模型推理，不更改源码。未提供`--video`时，用FFmpeg生成极小的临时测试视频；完整FFmpeg须支持lavfi。

`pip check`中只有精确的decord0.6.0平台标签提示，且实际解码通过，才作为已记录例外接受。其他冲突或解码失败仍令检查失败，不一律忽略decord错误。

```text
/workspace/vabench-env-audit/
  install.json                 # 安装状态、清单哈希及外部脚本来源
  installed.freeze.txt         # 实际安装版本及Git引用
  original/nisqa-*.whl         # 原始PyPI wheel，不覆盖
  patched/nisqa-*.whl          # 仅改五项依赖元数据的本地wheel
  patched/nisqa_patch.json     # 原始/补丁哈希、修改字段、代码不变说明
  environment_check.json      # 每项检查、输出日志及总状态
```

`environment_check.json`中的`model_inference_verified=false`表示检查通过也不等于15项评分通过；此标记不会被导入成功或微型CUDA测试改成true。

## 6. AISBench调度该环境

回到已安装AISBench的主环境，编辑`ais_bench/configs/vabench_examples/eval_vabench.py`：

```python3
DEVICE = 'cuda:0'
BACKEND_PYTHON = '/root/miniconda3/envs/vabench311/bin/python3'
NUM_DEVICES = 2
```

还需填写`PREPARED_DIR`、`SOURCE_ROOT`和`CACHE_DIR`。可见物理卡6/7在子进程中是逻辑0/1；具体卡可见性也受LocalRunner调度影响，应保留实际运行日志。

独立CLI等价运行：

```bash
python3 -m ais_bench.benchmark.vabench.cli run \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --prepared /workspace/prepared_mini --device cuda:0 \
  --python /root/miniconda3/envs/vabench311/bin/python3 \
  --options ais_bench/configs/vabench_examples/options.json \
  --output /workspace/vabench_runs/gpu_mini
```

install仅安装依赖，不产生评分；需执行prepare和run。run已展示15项原始均分、固定归一化分和正式/预览/缺分总分，并记录LSA省略缺口；断点补跑和更细的场景归因仍待交付。详见[VABench评测指南](vabench_t2av.md)。H200验收需使用同一批输入与官方基线逐项核对原始结果及状态；均值相对误差<1%，官方均值绝对值≤1e-6时采用绝对差≤1e-6。`second_clap`长音频可能触发随机裁剪；本地固定缓存配置为`rand_trunc`，服务器须核对实际配置。原始均分相对误差<1%的验收标准保持不变。对照前应保证同一输入、顺序和随机状态/采样片段；未控制随机性时应标记待对齐，不能自动用多次均值替代验收。重复运行只作为补充稳定性分析。

依据：[NISQA官方分发元数据](https://pypi.org/pypi/nisqa/2.0.post2/json)、[固定ImageBind依赖](https://github.com/facebookresearch/ImageBind/blob/53680b02d7e37b19b124fa37bae4b6c98c38f5be/requirements.txt)、[固定PyTorchVideo解码接口](https://github.com/facebookresearch/pytorchvideo/blob/2ac6ed361309cb25144636ad82dd8bd6a4970e0c/pytorchvideo/data/encoded_video.py)、[固定CLIP源码](https://github.com/openai/CLIP/tree/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6)。
