# 数据集准备指南
## 支持数据集类型
AISBench Benchmark当前支持的数据集类型如下：
1. [开源数据集](#开源数据集)，涵盖通用语言理解（如 ARC、SuperGLUE_BoolQ、MMLU）、数学推理（如 GSM8K、AIME2024、Math）、代码生成（如 HumanEval、MBPP、LiveCodeBench）、文本摘要（如 XSum、LCSTS）以及多模态任务（如 TextVQA、VideoBench、VocalSound）等多个方向，满足对语言模型在多任务、多模态、多语言等能力的全面评估需求。
2. [随机合成数据集](#随机合成数据集)，支持指定输入输出序列长度和请求数目，适用于对于序列分布场景和数据规模存在要求的性能测试场景。
3. [自定义数据集](#自定义数据集)，支持将用户自定义的数据内容转换成固定格式的数据进行测评，适用于定制化精度和性能测试场景。

## 开源数据集
开源数据集指的是社区广泛使用、公开可获取的数据集。它们通常用于模型训练、验证和比较不同算法的效果。AISBench Benchmark支持多个主流开源数据集，便于用户快速进行标准化测试，详细介绍和获取方式如下：
### LLM类数据集
| 数据集名称| 分类 | 详细介绍&获取方式 |
|---------------|----------------------|-------------------|
| DEMO            | 数学推理   | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/demo/README.md) |
| ARC_c            | 推理（常识+科学）     | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/ARC_c/README.md) |
| ARC_e            | 推理（常识+科学）     | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/ARC_e/README.md) |
| SuperGLUE_BoolQ  | 自然语言理解（问答）  | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/SuperGLUE_BoolQ/README.md) |
| agieval          | 综合考试/推理         | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/agieval/README.md) |
| aime2024         | 数学推理              | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/aime2024/README.md) |
| aime2025         | 数学推理              | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/aime2025/README.md) |
| bbh              | 多任务（Big-Bench Hard） | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/bbh/README.md) |
| cmmlu            | 中文理解/知识问答     | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/cmmlu/README.md) |
| ceval            | 中文职业考试          | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/ceval/README.md) |
| drop             | 阅读理解+推理         | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/drop/README.md) |
| gsm8k            | 数学推理              | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/gsm8k/README.md) |
| gpqa             | 知识问答              | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/gpqa/README.md) |
| hellaswag        | 常识推理              | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/hellaswag/README.md) |
| humaneval        | 编程（代码生成+测试） | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/humaneval/README.md) |
| humanevalx       | 编程（多语言）         | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/humanevalx/README.md) |
| ifeval           | 编程（函数生成）       | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/ifeval/README.md) |
| lambada          | 长文本完形填空         | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/lambada/README.md) |
| lcsts            | 中文文本摘要          | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/lcsts/README.md) |
| livecodebench    | 编程（实时代码）       | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/livecodebench/README.md) |
| longbench        | 长序列                | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/longbench/README.md) |
| longbenchv2        | 长序列                | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/longbenchv2/README.md) |
| math             | 高级数学推理          | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/math/README.md) |
| mbpp             | 编程（Python）         | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mbpp/README.md) |
| mgsm             | 多语言数学推理        | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mgsm/README.md) |
| mmlu             | 多学科理解（英文）     | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mmlu/README.md) |
| mmlu_pro         | 多学科理解（专业版）   | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mmlu_pro/README.md) |
| needlebench_v2      | 长序列                | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/needlebench_v2/README.md) |
| piqa             | 物理常识推理          | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/piqa/README.md) |
| siqa             | 社会常识推理          | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/siqa/README.md) |
| triviaqa         | 知识问答              | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/triviaqa/README.md) |
| winogrande       | 常识推理（代词消解）   | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/winogrande/README.md) |
| Xsum             | 文本生成（摘要）       | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/Xsum/README.md) |
| BFCL             | 函数调用能力评估        | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/BFCL/README.md) |
| FewCLUE_bustm    | 短文本语义匹配          | [详细介绍](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_bustm/README.md) |
| FewCLUE_chid     | 阅读理解填空            | [详细介绍](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_chid/README.md) |
| FewCLUE_cluewsc  | 代词消歧                | [详细介绍](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_cluewsc/README.md) |
| FewCLUE_csl      | 关键词识别              | [详细介绍](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_csl/README.md) |
| FewCLUE_eprstmt  | 情感分析                | [详细介绍](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_eprstmt/README.md) |
| FewCLUE_tnews    | 新闻分类                | [详细介绍](https://github.com/AISBench/benchmark/blob/master/ais_bench/benchmark/configs/datasets/FewCLUE_tnews/README.md) |
| dapo-math-17k    | 数学推理（RL评估）    | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/dapo_math/README.md) |

### 多模态类数据集
| 数据集名称| 分类 | 详细介绍&获取方式 |
|---------------|----------------------|-------------------|
| textvqa          | 多模态理解（图+文）    | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/textvqa/README.md) |
| videobench       | 多模态理解（视频）     | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/videobench/README.md) |
| vocalsound       | 多模态理解（音频）     | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/vocalsound/README.md) |
| Omnidocbench          | 图片OCR（图+文）    | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/omnidocbench/README.md) |
| MMMU       | 多模态理解（图+文）      | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mmmu/README.md) |
| MMMU_Pro       | 多模态理解（图+文）      | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mmmu_pro/README.md) |
| InfoVQA       | 多模态理解（图+文）      | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/infovqa/README.md) |
| DocVQA       | 多模态理解（图+文）     | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/docvqa/README.md) |
| MMStar       | 多模态理解（图+文）      | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mmstar/README.md) |
| Video-MME       | 多模态理解（视频+文）     | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/videomme/README.md) |
| OCRBench_v2    | 多模态理解（图+文，OCR评估） | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/ocrbench_v2/README.md) |
### 多轮对话类数据集
| 数据集名称 | 分类     | 详细介绍&获取方式                                                                                                     |
| ---------- | -------- | --------------------------------------------------------------------------------------------------------------------- |
| sharegpt   | 多轮对话 | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/sharegpt/README.md) |
| mtbench    | 多轮对话 | [详细介绍](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/mtbench/README.md)  |

 **提示：** 用户可以将获取的数据集文件夹统一放置在`ais_bench/datasets/`目录下，AISBench Benchmark 会根据数据集配置文件自动检索改目录下的数据集文件进行测试
### 配置开源数据集
AISBench Benchmark 开源数据集配置按照数据集名称保存在 [`configs/datasets`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets) 目录下，在各个数据集对应的文件夹下存在多个数据集配置，文件结构如下所示：

```text
ais_bench/benchmark/configs/datasets
├── agieval
├── aime2024
├── ARC_c
├── ...
├── gsm8k  # 数据集
│   ├── gsm8k_gen.py  # 不同版本数据集配置文件
│   ├── gsm8k_gen_0_shot_cot_str_perf.py
│   ├── gsm8k_gen_0_shot_cot_chat_prompt.py
│   ├── gsm8k_gen_0_shot_cot_str.py
│   ├── gsm8k_gen_4_shot_cot_str.py
│   ├── gsm8k_gen_4_shot_cot_chat_prompt.py
│   └── README.md
├── ...
├── vocalsound
├── winogrande
└── Xsum
```
开源数据集配置名称由以下命名方式构成 `{数据集名称}_{评测方式}_{shot数目}_shot_{逻辑链规则}_{请求类型}_{任务类别}.py`，以 `gsm8k/gsm8k_gen_0_shot_cot_chat_prompt.py` 为例，该配置文件则为`gsm8k` 的数据集，对应的评测方式为 `gen`，即生成式评测（目前只支持生成式测评），shot提示的样本数为0，逻辑链规则为`cot`表明请求中包含逻辑链提示，不指定表明没有逻辑链提示，`chat_prompt`表明请求类型为对话，任务类别没有指定，默认为精度测试；同样的， `gsm8k_gen_0_shot_cot_str_perf.py` 指定请求类型为`str`字符串，请求类型`perf`表示模板用于性能测评任务。
> 💡 **提示:** 指定数据集配置名称时，可以不包含 `.py` 后缀

开源数据集的配置参数同样基于Python语法描述，以gsm8k为例，参数内容如下：
```python
gsm8k_datasets = [
    dict(
        abbr='gsm8k',                       # 测评任务中数据集的唯一标识
        type=GSM8KDataset,                  # 数据集类成员，与数据集绑定，暂不支持修改
        path='ais_bench/datasets/gsm8k',    # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=gsm8k_reader_cfg,    # 数据读取配置，暂不支持修改
        infer_cfg=gsm8k_infer_cfg,      # 推理测评配置，暂不支持修改
        eval_cfg=gsm8k_eval_cfg)        # 精度测评配置，暂不支持修改
]
```

## 随机合成数据集

合成数据集是通过程序自动生成的，适用于测试模型在不同输入长度、分布和模式下的泛化能力。AISBench Benchmark 提供两类合成数据集：随机字符序列和随机 token 序列。无需额外下载，用户只需通过配置文件进行参数设置即可使用。详见：📚 [合成随机数据集配置文件使用指南](../../advanced_tutorials/synthetic_dataset.md)

### 使用方式

使用方式和开源数据集相同，在`ais_bench/benchmark/configs/datasets/synthetic/`目录下选择需要的配置文件即可，目前已提供[synthetic_gen.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/synthetic/synthetic_gen.py)，示例命令如下：

```bash
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen
```

## 自定义数据集
AISBench Benchmark 支持用户接入自定义数据集，满足特定业务需求。用户可将私有数据整理为标准格式，通过内置接口无缝集成至评估流程中。详见：📚 [自定义数据集使用指南](../../advanced_tutorials/custom_dataset.md)