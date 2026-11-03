# VABench T2AV 评测需求与设计

本文沿用 Geometry3K 需求设计的“背景—方案—使用—测试”结构，描述当前 AISBench 对外部 VABench 的接入。核查基线为 AISBench `2d71f1e`、官方 VABench `12b9617a900a9eca1691da6b7115ce3d58f53a16`；本次补充测试、设计和快速上手，并参照 VBench 完成自有适配代码分层。当前本地测试不能代替 H200 评分及官方精度对齐验收。

## 1. 背景描述

### 1.1 项目背景

原始需求《支持 VABench 音视频生成评测能力》《支持音视频同出模型精度测评》《构建 VABench Mini Benchmark 快速评测集》《测评工具支持音视同出模型精度测评》提出：在 AISBench 中评测文本生成音视频模型，覆盖数据加载、生成、产物收集、专家/MLLM 评分及结果保存；支持完整集及快速子集，在 GPU 和昇腾环境验证，并与官方结果对照。MiniMax H3 等模型是被测生成模型，Qwen2.5-Omni 是评分模型，两者职责不同。

结合已在 H200 容器 Conda 环境运行的现场过程，本阶段优先复用已生成的音视频，打通可追溯的 15 项评分；保留外部官方源码，不复制维护一套评分算法。

### 1.2 目标与需求追踪

| 编号 | 需求 | 当前设计/交付 | 验收边界 |
| --- | --- | --- | --- |
| R1 | T2AV 数据加载 | 固定 JSON/mapping 指纹，采用精确匹配的 778 题 | 不将原始 848 条全部视作可映射题目 |
| R2 | 生成及产物收集 | 接收已有带音轨 MP4，按 idx 查找并抽取同源 WAV | API 提交、轮询、下载后续接入，不自动产生生成费用 |
| R3 | 专家与 MLLM 全量评测 | 独立子进程调用官方类/compute 函数，8 项专家＋7 项 MLLM | 真实 GPU 运行与逐指标有效性需服务器验证 |
| R4 | 官方口径与落盘 | 保留原始记录及官方格式备份，单独生成有效均分、归一化分和总分 | 项目展示总分不称为官方 VABench 总分 |
| R5 | 快速子集 | 固化用户已选定的 10% 集合，共78题 | 显式固定题目及输入数，不能因缺视频缩小分母 |
| R6 | GPU / 昇腾 | 当前 H200/GPU 优先，依赖脚本面向 Linux x86_64 Conda 3.11 | 已有 NPU 迁移分支不表示 A3 已验证，NPU 验收后续进行 |
| R7 | 官方对齐 | 相同输入/源码/权重/参数及样本状态，逐指标原始均分误差 <1% | 接口复用、mock 通过或参考模式 PASS 均不能替代原生 CLI 对齐 |
| R8 | 工程质量 | 全部接入实现与兼容层纳入 UT 覆盖率，包含真实子进程 | 语句、分支及逐模块合并覆盖率门槛均为 80% |

### 1.3 范围与限制

本期支持 T2AV 已有音视频的准备、评测调度、状态追踪、原始结果落盘及统计展示。不包含 FL2VA/R2VA 生成、视频生成 API、自动分片并发、断点续评或自动导入原生官方结果。`partial_report.json` 用于审计进度，不是续跑检查点。

生成模型无需通过 AISBench 文本推理接口服务化，评分 Qwen 直接由后端进程加载。不得通过补静音、拉伸视频、改 Prompt 或替换模型算法把无效输入强行变成有效分。

### 1.4 整体架构

```text
已有模型生成 MP4 + 固定题目/映射
              ↓
data：题目校验 → 媒体探测 → WAV → prepared manifest / JSON
              ↓
AISBench Dataset → VABenchEvalTask → engine
                                  ↓
              专家独立子进程 / Qwen 七指标共享子进程
                                  ↓
             外部固定 VABench + 本地模型缓存 + 评分 Conda
                                  ↓
         原始返回值 / 日志 / 错误归因 / 设备观测 / 来源指纹
                                  ↓
       results + scoring → 官方格式备份、逐样本/场景/指标/总分
                                  ↓
                         VABenchSummarizer
```

AISBench 控制任务生命周期及报告；外部 VABench 控制评分算法和模型预处理。运行时路径、资源注入与设备观测属于工程适配，原始值始终可追溯。

### 1.5 完整数据流

1. `catalog` 校验固定官方数据哈希，按 JSON 中完整 `prompt_en` 精确关联 TSV mapping 的 idx。
2. 根据标准集、单题 smoke 或显式 custom（含固定78题） 确定应评清单。
3. `prepare` 查找每题唯一 MP4，ffprobe 检查音视频流，FFmpeg 抽取第一条音轨为 PCM16 WAV。
4. 保存元数据 JSON、媒体哈希、输入与元数据签名；运行前 `verify_prepared` 重验，防止产物被替换。
5. `engine` 生成 worker 请求；逐组预检资源，启动指定 Conda 的 Python 子进程，持续保存部分报告。
6. `worker` 导入官方代码，调用 `compute_<dimension>` 或官方类参考路径，记录原始结果与后端日志。
7. `results` 对齐应评样本，剔除已识别失败，保留省略和原始数值；`scoring` 按固定协议归一化和聚合。
8. 写出 JSON/CSV/Markdown、官方格式备份；主流程汇总器展示指标与总分。原生官方 CLI 另行运行作为独立基线。

## 2. 方案设计

### 2.1 目录结构与放置合理性

参照当前仓库 VBench 的 Dataset/Task/Summarizer 分工，VABench 已完成分层迁移：

```text
ais_bench/
  benchmark/
    datasets/vabench.py              # 数据集注册与准备清单加载
    tasks/vabench_eval.py            # 框架任务入口，与 VBenchEvalTask 同层
    tasks/vabench/
      engine.py                     # 评分组编排、子进程生命周期
      worker.py                     # 外部官方 compute/类调用、错误与设备观测
      resources.py                  # 运行资源检查、权重与源码指纹
    summarizers/
      vabench.py                    # 主框架汇总展示
      vabench_scoring.py            # 固定归一化、15项等权总分
      vabench_results.py            # 逐样本状态、统计、导出、比较
    utils/vabench/
      __init__.py                   # 共享指标常量，无重依赖导入
      data.py                       # 跨层共用的题目、媒体、JSON/哈希协议
      protocol.json                 # 固定778题协议
  tools/vabench/
    cli.py                          # catalog/prepare/run/compare等操作入口
    environment.py                  # 依赖安装、检查和审计
  configs/vabench_examples/          # 用户配置、78题清单、运行示例
requirements/datasets/vabench*.txt   # 可选依赖声明
```

VBench 的数据集入口位于 `datasets/vbench.py`，任务位于 `tasks/vbench_eval.py`，归一化和总分位于 `summarizers/vbench.py`，所带评分代码位于 `ais_bench/third_party/vbench`。VABench 采用相同的职责边界，但官方评分源码按已确认方案继续由外部 `--source-root` 指定，不另复制到 `third_party`。

原 `benchmark/vabench` 将数据协议、调度、安装和汇总放在顶层同一个领域包中，功能可用，但职责不够清晰。迁移后，业务运行时集中在任务层，统计计算集中在汇总层，操作工具独立放入 `tools`；`utils/vabench` 仅容纳跨层共享且无框架/模型重依赖的协议与文件处理，不把整个评测运行时搬进通用工具层。

保留 `datasets/vabench.py`、`tasks/vabench_eval.py`、`summarizers/vabench.py` 原有类路径，现有 AISBench 配置无需改类名。命令统一使用 `python3 -m ais_bench.tools.vabench.cli` 和 `python3 -m ais_bench.tools.vabench.environment`。顶层兼容目录已删除，旧模块命令和 helper 导入不再提供；代码和测试直接引用分层实现。

迁移同时更新 worker 启动路径、包根与依赖清单定位、protocol 打包及覆盖率范围。实际实现指纹覆盖 Dataset、Task、运行时、共享协议、汇总及工具各层；旧报告的实现指纹会与新版本不同，应保留版本并重新建立对照，不能为迁移而关闭来源检查。目录重构不改变评分规则。当前规划已清理退役子集元数据，数据协议哈希随之更新，旧准备目录应重新 prepare；不自动改写原始媒体或已有清单。

CLI、环境命令及 worker 可在未安装 mmengine/datasets/torch/transformers 的环境中导入；这些模型依赖仍在实际评分时加载。该隔离边界有独立子进程 UT 验证，不通过将整套框架装入评分环境来掩盖导入问题。

### 2.2 数据加载与固定子集

数据来自固定外部源码的 `json_files/json_text/*.json` 和 `mapping/final_idx_to_prompt.csv`。后者是 Tab 分隔，Prompt 中的转义按 JSON 字符串解码。只接受精确匹配，不做模糊匹配或自动修名。

原始 JSON 848 条，当前可映射标准集为 778 条，指纹与 idx 固定在 `protocol.json`。`catalog` 返回 `idx`、`category`、`metadata`；metadata 保留官方 `prompt_en`、分模态提示词、维度和 QA 等辅助信息。

| scope | 数量 | 规则 |
| --- | --- | --- |
| `standard` | 778 | 固定可映射题目 |
| `smoke` | 1 | 显式选取 778 中一个 idx |
| `custom` | 非空子集 | 所有题目仍须属于固定 778，不自动忽略未知视频 |

10% 集合以 `ais_bench/configs/vabench_examples/mini10pct.json` 固定 78 个 idx 和类别，附来源哈希。来自用户现有 `build_mini10pct.py` 产物，24 类全覆盖、easy 40、hard 38，每类 2–4 题；类内使用 TF-IDF、多样性选择及 SHA256 同分排序。它是词面多样性的工程抽样，不宣称语义代表性已统计验证，也不是官方 split。

78 题使用显式清单传给 `prepare(..., subset='custom')`，缺少视频时拒绝准备。不能直接对仅有 5 个视频的目录执行 custom 后称为“完整 78 题”。原始脚本依赖 WSL 路径及缺失辅助文件，获取流程复用固化清单，不要求重跑生成任务。

### 2.3 媒体准备与落盘

输入文件以 `<idx>.mp4` 命名，可以包含空格。一个 idx 必须对应唯一文件；缺失、重复、无音轨、无视频流或时长非正时失败。对标准/显式集合，清单之外的媒体不改变应评分母；普通 custom 会读取目录中全部 MP4，并拒绝未知 idx。

抽取命令为 `ffmpeg -i <MP4> -map 0:a:0 -vn -c:a pcm_s16le <WAV>`，不主动重采样、裁剪或混声道；官方各模型再执行自身预处理。manifest 记录实际音视频路径、哈希、时长、短视频标记、FFmpeg 版本和协议签名。原始 MP4 不复制、不重编码，因此目录迁移后应重新准备。

JSON 与 manifest 使用临时文件＋原子替换，避免写到一半被识别为完成。已存在完成 manifest 的准备目录拒绝覆盖。数据集 loader 只消费准备结果，不在多个 task 中重复执行媒体准备。

### 2.4 依赖声明和 Conda 环境

参考 `requirements/datasets/geometry3k.txt` 的可选依赖模式，避免让所有 AISBench 用户下载多模态 GPU 评分包：

| 清单/资源 | 管理方式 |
| --- | --- |
| `requirements/datasets/vabench.txt` | 固定 GPU PyPI 依赖 |
| `vabench-build.txt` | 固定构建工具、NumPy/Cython，先准备源码扩展构建环境 |
| `vabench-git.txt` | CLIP、ImageBind、PyTorchVideo 固定完整 Git 提交，分阶段安装 |
| `requirements/vabench.txt` | 旧路径兼容，不是完整安装入口 |
| NISQA 2.0.post2 | 核验原始 wheel 哈希，仅放宽五处过严元数据等号，不改模型实现 |
| DNSMOS / Synchformer | 外部源码/缓存代码，安装工具补齐或检查，不属于普通 pip 权重 |
| 评分权重、人脸权重 | 单独下载/复制及预检，pip 依赖声明不能替代 |
| Git、FFmpeg/ffprobe、C/C++ 编译工具等 | 系统包；root apt 容器可由安装工具准备 |

采用用户最终运行过程支持的 Torch 2.6.0、torchvision 0.21.0、torchaudio 2.6.0，保留 CUDA 12.4 运行库。驱动显示 CUDA 13.2 不要求把 PyTorch 升为该版本。仅保留一套 OpenCV headless，与 NumPy 1.26.4 配套。

NISQA 元数据补丁只涉及 Torch、torchvision、SciPy、scikit-learn、Matplotlib 五个版本声明，重新生成 RECORD 并保存补丁证据。固定 Git 包与补丁 wheel 采用受控 `--no-deps`，其余解析和最终 `pip check` 不绕过。安装输出记录计划、实际 freeze、Git 来源和环境检查；失败返回非零。

默认建议主环境和评分 Conda 分离，通过 `BACKEND_PYTHON` 指定后端。现场统一 Conda 已有成功运行记录，但 Pillow/fsspec/tqdm 等存在已知声明冲突，必须如实保留，不能宣称为通用无冲突环境。详细步骤见 [环境指南](../get_started/vabench_gpu_environment.md) 和 [现场统一环境](../get_started/setup.md)。

### 2.5 评分调度与设备

`VABenchEvalTask` 负责主框架适配，`engine.run` 负责实际编排。8 个专家指标各一组独立进程，7 个 MLLM 指标组成一个共享 Qwen 进程，共 9 组，默认顺序执行。组内按场景处理，避免将同时加载所有评分模型作为默认前提。

评分子进程设置外部源码路径和本地缓存，在导入前配置离线环境。adapter 模式直接调用 `compute_<dimension>`；official 模式调用带 mapping 路径修正和本地资源注入的官方类。后者是参考路径，不能替代原生 `evaluate.py`。

设备分三层记录：请求值、初始化/工厂参数和实际可观测的 Torch 设备或 ONNX provider。未观测到设备时留空，不由请求值推断已在 GPU/NPU 上运行。DNSMOS 默认 CPU；视频解码及文件处理可使用 CPU。GPU 多卡可见性由用户/Runner 控制，卡号按子进程逻辑编号解释。

看板按完成组数/组总数折算到样本数，只是阶段进度。`LocalRunner.max_num_workers` 允许独立任务并发，不会自动把同一集合切成多个视频并发评分。未来并发需要额外定义显存预算、Qwen 复用边界、分片固定分母、失败补跑与全局逐样本合并协议。

### 2.6 原始评分、异常和适用性

| 类别 | 指标 | 官方主评分 |
| --- | --- | --- |
| 音频质量 | `first_dnsmos`、`first_nisqa` | MOS 预测 |
| 音频审美 | `first_audiobox` | `(CE+CU+PQ-PC)/4` |
| 跨模态一致性 | `second_viclip`、`second_clap`、`second_imagebind` | 官方余弦相似度主分，ImageBind 保留 sim_av |
| 同步 | `second_desync` | 预测绝对时间偏移，越小越好 |
| 唇形同步 | `second_lsa` | 同步置信度，越高越好 |
| MLLM 直接评分 | 5 个 `third_*` | 官方提示词和解析得到的 1–5 分 |
| MLLM QA | 2 个 `fourth_qa_*` | 官方正向极性比例，不改成简单 yes 占比 |

逐题结果包括 `status`、有效 `score`、`original_score` 和原始 `raw`。缺失、模型异常、可识别解析失败或非法数值不伪装成正常零分。仅有效记录参与均值；原始数值与官方格式备份仍保留，便于审计模型内部兜底。

对 CLAP/Desync/ImageBind 等错误，结合具名日志逐样本归因。若所有错误都能映射，则保留同场景其他有效分；全局错误、临时路径或无法归因的 traceback 使该场景保守标记未通过。未捕获到的官方内部错误仍有残余风险，不能宣称已完整恢复所有模型失败语义。

LSA 以源码行为为准，不再建立人工语义 N/A 清单。无脸、零置信度或异常均可能不返回记录，故统一展示有效数/输入数与省略数；不据此猜测原因。省略造成覆盖缺口；全无有效 LSA 分时不生成总分。

### 2.7 指标均分、固定归一化与总分

每项先计算全部成功视频原始分的算术平均 m；各视频等权，不把不同规模场景的均分等权平均。另行展示各场景统计。归一化后限制为 0–100，保留超界数量及原始均分，不改写原始结果。

| 指标 | 固定公式 |
| --- | --- |
| DNSMOS、NISQA、5 个 third 指标 | `(m−1)/4×100` |
| AudioBox | `(m+1.75)/9×100` |
| ViCLIP、CLAP、ImageBind | `(m+1)/2×100` |
| Desync | `(1−m/2)×100` |
| LSA | `m/10×100` |
| 2 个 QA 指标 | `m×100` |

协议版本为 `aisbench-vabench-display-v1`。MOS、AudioBox 和 LSA 使用固定展示锚点；LSA 的 10 只是项目锚点，不表示模型硬上限。

总分始终使用 15 项等权平均，权重不因缺项而变化：缺任何有效均分则不可用；各项有分但仍有缺口或已记录错误则为预览；15 项均有分且全部应评记录成功则为当前范围正式总分。未选择指标显示未评测。正式状态不等于已完成官方精度验收。

### 2.8 原始输出、日志和来源

`report.json` 保存逐题记录、15 项及场景统计、总分状态；`samples.csv`、`metrics.csv`、`scenes.csv`、`total.csv` 和 `summary.md` 用于查看。`official_eval_results/<category>` 保留官方结构 JSON/CSV/full_info；CSV 的缺失哨兵 `-1` 不能作为有效成绩计算 AISBench 均值。

来源包含输入/协议、官方源码、适配器、权重和参数签名，后端 Python/包版本及设备观测。worker 的 request/response/process.log、按指标/场景 backend.log 和 partial 报告保留。源码指纹用于检测内容变化，不把包版本字符串当作完整来源证明。

`summarize` 可以对已有 AISBench 报告离线重汇总，保存源文件哈希和未重跑标志；它不能恢复原报告没有保存的 MLLM 完整生成文本，也不是新的 GPU 验证。

### 2.9 官方对齐规则

以原生官方 `evaluate.py` 为独立基线，准备同一批视频、同源 WAV、相同题目元数据及 mapping，使用相同权重和实际评分参数。比较逐题样本/失败/LSA 省略集合后，再比较各项原始均分：

- 官方均分绝对值 `>1e-6`：`abs(candidate-reference)/abs(reference) <0.01`。
- 官方均分绝对值 `≤1e-6`：`abs(candidate-reference) ≤1e-6`。
- 非 LSA 的执行失败不能仅凭两边失败状态相同就通过；LSA 省略集合一致且至少有有效分，才可继续比较。

当前 `compare` 读取 AISBench 的两份报告，可对比 adapter 和官方类参考路径；原生官方结果自动导入及完整溯源仍未实现。原始备份可用于与原生结果人工/脚本逐题核对。

CLAP 长音频存在随机裁剪。本地缓存预处理配置为 `rand_trunc`，不能只按 Transformers 构造函数默认值断言使用 fusion。必须记录实际预处理配置及随机状态、控制裁剪条件，保留重复实验波动证据；不能自行将 <1% 改成多轮均值或更宽容差。

## 3. 使用说明

### 3.1 前置条件

Linux x86_64 GPU 容器、可用 GPU 驱动与空闲显存、Conda Python 3.11、固定官方源码、完整模型缓存、系统 FFmpeg/ffprobe 和输入视频。依赖脚本的镜像源可选，GitHub 和模型资源下载单独处理。

### 3.2 命令与配置入口

完整可执行步骤见 [VABench 快速上手](../extended_benchmark/lmm_generate/vabench.md)：环境安装/检查、题目导出、78题准备、15 项运行、主命令配置、官方 CLI 对照、固定 78 题获取与准备。

主要入口为 `python3 -m ais_bench.tools.vabench.cli` 的 `catalog`、`prepare`、`doctor`、`run`、`summarize`、`compare`。环境入口为 `python3 -m ais_bench.tools.vabench.environment` 的 `plan`、`install`、`check`。AISBench 主命令使用 `ais_bench/configs/vabench_examples/eval_vabench.py -m eval`。

### 3.3 输出验收

同时保留环境审计、实际 freeze、准备 manifest、原始 MP4/WAV、评分全目录、独立官方输出和固定子集清单。检查 15 项的有效数、失败数、覆盖率和原始日志；不能只截取总分或“运行结束”。

78 题结果必须核对输入数为 78；只有 5 个视频时明确标注自定义 5 条冒烟。标准集验收使用固定 778 题。小猫单题仅适合链路验证，不能证明所有同步/唇形指标适用。

## 4. 测试设计

### 4.1 测试结构

`tests/UT/vabench` 覆盖分层后的 Dataset、Task、运行时、共享协议、汇总和工具，共 14 个 Python 模块（12 个含可执行语句，另 2 个仅为包文档）。使用真实临时文件、FFmpeg 媒体和评分子进程协议；评分模型、硬件接口及 pip/apt 安装边界采用替身，避免在 UT 中下载大模型、实际安装软件或要求 GPU。

### 4.2 关键用例

| 范围 | 主要检查 |
| --- | --- |
| 数据 | 778题协议、78题固定清单、单题调试、custom 未知/重复/缺失、元数据和媒体被改写、实际 WAV 抽取 |
| 调度 | 子进程请求/响应、超时、缺权重、进程异常、组进度回调异常不改变评分 |
| 评分与报告 | 15 项固定公式、边界/超界、裁剪顺序、逐视频权重、失败与 LSA 省略、缺项/预览/正式状态 |
| 原始备份 | 官方 JSON/CSV/full_info 格式、缺失哨兵、Windows 安全时间戳 |
| 错误归因 | 具名失败、混合日志、未归因 traceback、合法零分、解析/OOM 兜底 |
| 设备 | CPU/CUDA/NPU 初始化成功/不可用分支、工厂配置与观测异常隔离 |
| 环境 | 安装计划、模拟阶段失败、源码/资源检查、NISQA 元数据、Git 来源、decord 与 FFmpeg 预检 |
| 框架入口 | 真实 Dataset 加载、Task 收尾、Summarizer 文件输出、CLI 与离线重汇总 |
| 分层与入口 | 直接导入实际实现、新命令与worker实际启动、无旧包、无重依赖导入、依赖路径、protocol 打包及各实现层指纹 |

### 4.3 覆盖率运行与门槛

```bash
cd /workspace/code/benchmark
python3 -m pip install -r tests/UT/vabench/requirements.txt
python3 tests/UT/vabench/run_coverage.py
```

测试专用 `sitecustomize.py` 根据 `COVERAGE_PROCESS_START` 启动父/子进程测量，并合并本次数据；正常产品运行不启用此钩子。运行器先清理本范围旧测量，生成 HTML/JSON/XML，任一条件失败即返回非零：UT 不通过、总语句覆盖率 <80%、总分支覆盖率 <80%、任一模块语句＋分支合并覆盖率 <80%。不通过排除环境、CLI 或异常分支提高数字。

2026-11-03 本地测量：最初基线 81 项测试、合并覆盖率 77.97%；补边界测试后 112 passed。分层迁移、当前子集规划清理及旧兼容入口移除后，**154 passed，语句 95.77%，分支 89.70%，合并 94.01%**；全部统计模块超过 80%，最低 91.18%。报告在 `test_reports/vabench/htmlcov/index.html`、`coverage.json`、`coverage.xml`。这是 VABench 接入范围的覆盖率，不是整个 AISBench 仓库覆盖率，也不证明真实评分模型精度。

### 仓库 CI 与专项覆盖率

专项报告仅统计 VABench 接入，不能代替整个仓库的 PR CI。PR #636 的旧 CI 报告全仓 79.95%，按普通 pytest-cov 收集复现时发现 worker 执行路径原先主要在子进程中覆盖，父进程报告仅58.22%。新增17项自包含测试后，父进程worker覆盖率提升到95.96%，VABench范围合并93.91%，新增覆盖121条语句；没有修改全仓统计范围或排除配置。关键执行及数据协议测试不依赖外部VABench源码、模型权重或FFmpeg，原集成测试继续保留。完整仓库是否达到80%，仍以修复后的同一PR CI为准。

### 4.4 GPU 集成验收与后续工作

H200 上使用记录过的 Conda 环境和全部权重，先做单题链路，再运行固定子集和标准集；逐指标检查失败、LSA 省略、随机采样、媒体长度及原生官方误差。只有满足第 2.9 节条件才能声明官方对齐。后续按独立工作项推进 A3/NPU、生成 API、原生官方结果导入、失败补跑、视频分片并发。目录分层及旧兼容入口移除均已完成。
