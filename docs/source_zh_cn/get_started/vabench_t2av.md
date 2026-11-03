# VABench T2AV评测接入与GPU运行指南

> 版本说明：本文新增的custom子集、官方格式备份、指标组进度及配置模板需要配套的新代码，不适用于仅含10e1469的版本。本次提交仅更新文档；使用新增功能前须同步配套代码。

本期先在H200/GPU接入外部VABench的15个指标，A3/NPU留后续。旧版接入骨架保留；当前代码已具备原始评分调用与落盘，已补固定归一化和总分展示；细化场景错误归因与补跑仍按需求规划继续实现，不宣称已完成GPU官方对齐。

依赖准备以[统一Conda Python3.11安装与预检指南](vabench_gpu_environment.md)为准；已实际跑通的统一环境记录和精简步骤见[vabench311_new统一环境](setup.md)、[精简步骤](env_setup.md)。`requirements/vabench.txt`已转向GPU专用清单，不能继续使用早期A3安装步骤将其安装进NPU环境。数据集级声明见`requirements/datasets/vabench.txt`，完整安装须使用环境准备工具处理固定Git包与NISQA元数据补丁。

## 数据与输入

固定上游为`tanABCC/VABench@12b9617a900a9eca1691da6b7115ce3d58f53a16`。源码由`--source-root`指定，保留外部仓库，不将官方评分算法复制到AISBench。

T2AV JSON共848条，其中778条与mapping精确对应，采用这778条并提供固定Mini10；其余70条不自动补名。数据文件指纹固定在`ais_bench/benchmark/vabench/protocol.json`，Mini清单含新闻主播、说唱、小猫、音乐、环境、物理同步、复杂场景和虚拟世界等题目。

生成视频的输入文本为清单中的`entries[].metadata.prompt_en`，视频文件基名为`entries[].idx`。本阶段评测已有带音轨MP4，不发送H3或其他收费生成任务。文件名可以包含空格；shell参数须加引号。

```bash
cd /workspace/code/benchmark
export VABENCH_ROOT=/workspace/code/VABench-main
export VABENCH_CACHE_DIR=/workspace/vabench-cache

python3 -m ais_bench.benchmark.vabench.cli catalog \
  --source-root "$VABENCH_ROOT" --subset mini --output /workspace/mini_prompts.json
python3 -m ais_bench.benchmark.vabench.cli catalog \
  --source-root "$VABENCH_ROOT" --subset standard --output /workspace/t2av_778_prompts.json
```

一个输入根目录下，每个选中idx必须恰好对应一个带音轨MP4，可以平铺或分场景。不要混入同题多个模型的结果以免歧义。工具不为无声视频补静音、不重新编码原视频；抽取同源未裁剪、未主动重采样的PCM16 WAV，各官方指标再执行自己的预处理。

```bash
python3 -m ais_bench.benchmark.vabench.cli prepare \
  --source-root "$VABENCH_ROOT" --subset mini \
  --videos /workspace/generated --output /workspace/prepared_mini
```

完整集改为`--subset standard`并使用新目录。manifest记录原MP4绝对路径和生成WAV路径及哈希，原视频需保留；搬到另一服务器后重新prepare。已完成的准备目录不覆盖，输入被改写会在运行前报错。

目录下所有官方样本可准备为自定义子集。仍限于固定778题：官方指标需要题目对应的JSON提示词和辅助信息，目录中出现778之外的视频会报错并列出，不会静默跳过，也不会缩小分母。

```bash
python3 -m ais_bench.benchmark.vabench.cli prepare \
  --source-root "$VABENCH_ROOT" --subset custom \
  --videos /workspace/generated --output /workspace/prepared_custom
```

报告scope为`custom`，完整时总分标记为“自定义子集总分（非官方）”；不得与Mini10/778验收结果混用。

单条冒烟从778题中精确选题，不改变Mini10或完整集：

```bash
python3 -m ais_bench.benchmark.vabench.cli prepare \
  --source-root "$VABENCH_ROOT" --subset smoke \
  --sample-id 'kitten curled up meowing' \
  --videos /workspace/generated --output /workspace/prepared_kitten
```

约4秒的小猫适合验证媒体和CLAP链路；Synchformer可能因输入长度异常返回兜底1.0，LSA无脸没有有效结果。不能把程序运行到末尾或CSV里的-1认作全部15项有效成绩，也不自动拉伸视频或改变官方算法来补分。

## GPU评分

AISBench主环境可以与VABench评分Conda分离。填写后端Python，确保评分包全部来自该Conda。默认专家串行独立子进程，七个MLLM指标在一个子进程中复用Qwen2.5-Omni；不要求部署HTTP模型服务。

```bash
export CUDA_VISIBLE_DEVICES=6,7
python3 -m ais_bench.benchmark.vabench.cli run \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --prepared /workspace/prepared_mini --device cuda:0 \
  --python /root/miniconda3/envs/vabench311/bin/python3 \
  --options ais_bench/configs/vabench_examples/options.json \
  --output /workspace/vabench_runs/gpu_mini
```

仅做小猫CLAP冒烟时，将prepared换为单条准备目录，加`--dimensions second_clap`，并使用新输出目录。`requested_device`是请求值，实际设备证据见报告观测字段；DNSMOS默认显式使用CPU ONNX，其他指标按配置运行。

使用AISBench主命令时，编辑`ais_bench/configs/vabench_examples/eval_vabench.py`中的三个目录、`BACKEND_PYTHON`、`DEVICE`、`NUM_DEVICES`，再执行：

```bash
ais_bench ais_bench/configs/vabench_examples/eval_vabench.py -m eval
```

可见物理卡6/7在进程中对应逻辑0/1。LocalRunner也会设置设备可见性，应根据实际分配填写预留卡数。单条配置为`eval_vabench_smoke.py`。任务看板的Progress以准备集视频数为分母；VABench按指标组调用官方函数，进度按已完成指标组折算，最后的Qwen组包含7个MLLM指标。

## 参考路径与官方对齐

当前`--mode official`在独立评分进程中执行官方VABench类，注入相同本地资源并修正mapping路径；它是带工程包装的参考路径，不等同于用户原样执行官方`evaluate.py`。两种模式均保留原算法。

```bash
python3 -m ais_bench.benchmark.vabench.cli run \
  --source-root "$VABENCH_ROOT" --cache-dir "$VABENCH_CACHE_DIR" \
  --prepared /workspace/prepared_mini --device cuda:0 --mode official \
  --python /root/miniconda3/envs/vabench311/bin/python3 \
  --options ais_bench/configs/vabench_examples/options.json \
  --output /workspace/vabench_runs/gpu_reference
python3 -m ais_bench.benchmark.vabench.cli compare \
  --reference /workspace/vabench_runs/gpu_reference/report.json \
  --candidate /workspace/vabench_runs/gpu_mini/report.json \
  --output /workspace/vabench_runs/gpu_alignment.json
```

比较默认要求全部15项、相同样本/状态及输入、源码、权重和参数指纹。逐指标原始均分相对误差严格<1%；官方均分绝对值≤1e-6时要求绝对差≤1e-6。LSA允许相同省略集合但至少有一个有效分，其他指标的执行失败不能凭数值相近通过验收。`second_clap`是官方随机指标，见下节；单次运行不能套用<1%阈值，应多次运行取均值并单独记录范围。Mini用于快速对照，778用于完整集验收。

最终仍需与用户官方CLI基线核对：专家进程隔离可能影响decord全局bridge及Qwen解码后端，调用相同compute函数并不能代替真实对齐证据。`compare`仍读取AISBench report；新增的`official_eval_results`可直接与官方CLI的`*_eval_results.json/csv`逐条对照，但自动导入官方基线及来源溯源仍待后续实现。

### second_clap的随机性

官方`vabench/second_clap.py`调用processor时未指定`truncation`，`ClapFeatureExtractor`默认`fusion`。对超过10秒的音频，`_random_mel_fusion()`使用未播种的`np.random.choice`随机抽取3段音频特征，因此同一输入重复运行分数会波动；短于10秒的音频不触发随机截断，结果稳定。实测7个视频中6个约10.144秒，小猫约4.064秒，后者在官方和多次AISBench运行中均为0.30362。

因此：

- 不得要求`second_clap`逐样本完全一致，也不要用单次运行套用默认<1%均值阈值。
- 与官方对比时应多次运行取均值并记录波动范围，或为该项单独设置绝对容差。
- 该项只占总分1/15，单次波动对总分影响通常小于0.1分。
- 需要AISBench内部可复现时，可在CLAP调用前固定`np.random.seed()`；这只能保证自身重复性，不等同于官方某次随机抽样。


## 固定归一化和总分展示

运行`run`完成后，终端和AISBench主流程汇总表均展示15项原始均分、0–100展示分及覆盖率；JSON和CSV同时落盘。每条成功视频等权，先计算原始均分再归一化，不对各场景均分等权平均，也不先裁剪逐样本展示分再平均。

| 指标 | 固定展示公式（m为原始均分） |
| --- | --- |
| DNSMOS、NISQA、5项third_* | (m−1)/4×100 |
| AudioBox | (m+1.75)/9×100 |
| ViCLIP、CLAP、ImageBind | (m+1)/2×100 |
| Desync | (1−m/2)×100 |
| LSA | m/10×100 |
| 2项QA | m×100 |

展示分限制在0–100，原始值不改写，超出固定展示锚点的样本数及均分标记保存在报告。MOS的1/5、AudioBox的−1.75/7.25、LSA的0/10是展示锚点，不宣称为所有模型预测的硬边界。ImageBind保留官方sim_av主分，QA保持官方positive-polarity比例，不改成简单yes占比。

总分为15项展示分等权平均，不能按可用指标数重新分配权重：

- 缺任一指标有效均分：总分为空，展示“—”。未选择评测的指标也显示“未评测”，不补0。
- 15项都有均分但有失败、缺失、LSA省略或未解决的后端错误：展示“预览总分／不完整”。
- 15项都有均分、输入评分覆盖率100%且没有已记录后端错误：展示当前评测范围的正式总分。

单条冒烟、自定义子集、Mini10和778各自保留scope与样本数，Mini10总分不能称为778标准集总分。“正式”仅指本轮评分完整，不代表官方<1%对齐验收已完成。LSA省略是missing的子集，CSV中missing/omitted不可相加重复计数。分组均分另存scenes.csv。

已有AISBench逐样本报告可离线重汇总，无需重新加载模型：

```bash
python3 -m ais_bench.benchmark.vabench.cli summarize \
  --report /workspace/vabench_runs/gpu_mini/report.json \
  --output /workspace/vabench_runs/gpu_mini_new_summary \
  --prepared /workspace/vabench_runs/gpu_mini_prepared
```

原始报告不会被覆盖，新报告保存来源哈希和`model_inference_rerun=false`。这不是新的GPU验证，也不能恢复原始报告没有保存的模型生成文本。`--prepared`用于恢复官方`prompt_en`，省略时CSV回退到原始记录中的`prompt`字段或样本`idx`。

## 输出与尚待改进

```text
run/
  report.json             # 原始记录、各指标/场景均分、状态及溯源
  metrics.csv             # 15项原始均分、归一化分、计数与覆盖率
  total.csv               # 当前评测范围的15项等权总分及状态
  summary.md              # 可直接阅读的15项与总分表格
  samples.csv             # 逐题状态、评分与请求设备
  scenes.csv              # 场景均分
  official_eval_results/  # 与官方evaluate.py同结构的原始指标备份
    <category>/
      <category>_<time>_eval_results.json
      <category>_<time>_eval_results.csv
      <category>_<time>_full_info.json
  input_manifest.json
  partial_report.json
  job_00/request.json
  job_00/response.json
  job_00/process.log
  job_00/指标/场景/backend.log
```

`official_eval_results`专为与官方结果对照：

- JSON保持官方`{dimension: {"video_results": [...]}}`结构，并保留官方评分函数返回的全部原始记录；即使AISBench因解析失败或后端错误不把该样本计入均分，原始`video_results`仍会写入，便于逐条核对。
- CSV表头为`prompt_name`加当前请求的指标列，按`prompt_en`聚合；某指标没有数值时写入`-1`，与官方`*_eval_results.csv`一致。
- `full_info.json`来自`prepare`生成的`json/<category>.json`，与官方`*_full_info.json`字段一致。
- 该备份只复现官方原始分数和CSV聚合方式，不改变AISBench的归一化、覆盖率、失败剔除和总分协议。

官方函数返回记录保存在raw中，不能声称已完整捕获MLLM全部生成文本。CLAP、Desync、ImageBind的具名错误及可映射的FFmpeg错误逐样本标记；只有所有检测到的错误事件均可归因时才保留同场景其他有效分。临时文件或全局错误无法映射时，该场景结果保守标记未通过，原始值仍保留；部分被官方吞掉的解析错误尚不能完整恢复。已识别的空解析/OOM及非法量表输出会剔除并保留原始值；LSA省略已计入覆盖率缺口。正式总分只表示当前选定样本的15项完整性，不代表GPU官方对齐已通过。

已有partial文件用于追踪，不等于已支持断点补跑；当前重跑须新建输出目录。环境准备报告、实际freeze、完整评分目录和官方原始输出应一起保留，以便复现和对齐。

本地轻量回归（在AISBench源码根目录执行）：

```bash
cd /workspace/code/benchmark

/root/miniconda3/envs/vabench311_new/bin/python3 -m pip install --no-deps \
  "pytest==8.3.5" "pluggy==1.5.0" "iniconfig==2.0.0" \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple

/root/miniconda3/envs/vabench311_new/bin/python3 -m pytest tests/UT/vabench -q
```

测试数量随版本变化，以本次完整输出为准。若官方源码目录未被测试发现，涉及真实官方数据/编排类的用例可能跳过；应配置正确目录并核对skip原因，不将跳过视作已验证。模型在集成测试中由模拟实现替代；真实FFmpeg和子进程契约通过，不表示H200模型评分通过。

参考：[VABench官方源码](https://github.com/tanABCC/VABench)、[官方模型缓存](https://huggingface.co/datasets/tanABBCC/VABENCH_CACHE_DIR)、[LatentSync人脸检测权重](https://huggingface.co/ByteDance/LatentSync-1.5/tree/main/auxiliary)、[InsightFace模型资源](https://github.com/deepinsight/insightface/tree/master/python-package#model-zoo)。
