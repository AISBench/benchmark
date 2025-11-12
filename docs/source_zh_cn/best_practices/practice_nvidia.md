# 基于英伟达A100加速卡测评DeepSeek-R1-Distill-Qwen-14B的数学能力，100%论文复现
### 复现使用的aisbench评测工具版本
本文复现使用aisbench测评工具版本为[v3.0-20250412](https://gitee.com/aisbench/benchmark/releases/tag/v3.0-20250412)
### 一 背景与目标

#### 1. 1 复现意义

##### 1.1.1 基于DeepSeek-R1蒸馏的小模型继承了DeepSeek-R1的数学推理优势
DeepSeek-R1作为专精数学推理任务的大模型，在复杂数学问题求解（如定理证明、多步推导）中展现出显著的性能优势。其官方评测在AIME2024（美国数学邀请赛题型，30道题）和MATH-500（500道国际竞赛级数学题）数据集上的表现，验证了其在长序列逻辑推理、符号运算精度上的领先性。官方证实，DeepSeek-R1大模型的推理范式可通过蒸馏技术迁移至小模型，相较于小模型通过强化学习自主探索的推理方式，该方法能实现更卓越的性能表现。经过蒸馏的小模型在复杂数学问题的求解上也继承了DeepSeek-R1的优势，展现出突破性性能表现。

##### 1.1.2 基于DeepSeek-R1蒸馏的小模型复现门槛低
DeepSeek-R1蒸馏的小模型相比于671B参数规模的DeepSeek-R1，部署的硬件门槛低，如DeepSeek-R1-Distill-Qwen-14B非量化版本显存需求仅需24GB，单张A100即可部署，可以轻松实现本地部署和服务化部署，可灵活应用于企业私有化场景。

#### 1.2 复现目标

通过AISBench评测工具，在单张A100加速卡上通过huggingface本身部署和vllm服务化部署方式，完整复现DeepSeek-R1-Distill-Qwen-14B的官方测评分数，具体指标如下：
AIME2024数据集：
官方准确率：Pass@1 69.7分
精度复现目标：误差控制在1道题以内。
MATH-500数据集：
官方准确率：Pass@1 93.9分
精度复现目标：误差控制在%0.5以内。
![输入图片说明](https://foruda.gitee.com/images/1744444555077371068/751e21f1_14098946.png "屏幕截图")
####  1.3 技术价值

验证AISBench支持大模型服务化部署和本地部署的评测场景，且能实现评测精度对齐。

### 二、A100环境准备

#### 2.1 NVIDIA硬件及软件栈版本
|组件|型号/版本|
| ---- | ---- |
|NVIDIA硬件|单张A100加速卡（80GB显存）|
|NVIDIA驱动版本|550.54.15|
|CUDA版本|12.4|
```shell
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA A100-SXM4-80GB          Off |   00000000:1B:00.0 Off |                    0 |
| N/A   35C    P0             65W /  500W |    1261MiB /  81920MiB |      0%      Default |
|                                         |                        |             Disabled
```

#### 2.2 其他软件依赖版本
"本地huggingface纯模型部署场景"和"vllm服务化部署场景"软件依赖不同
##### 2.2.1 本地huggingface纯模型部署
```shell
Python 3.10.15 (main, Oct  3 2024, 07:27:34)

torch==2.5.1
torchvision==0.20.1
transformers==4.48.1
```
##### 2.2.2 vllm服务化部署
```shell
Python 3.10.15 (main, Oct  3 2024, 07:27:34)

torch==2.5.1
torchvision==0.20.1
transformers==4.48.1
vllm==0.6.6.post1
```

#### 2.3 DeepSeek-R1-Distill-Qwen-14B原始权重获取
通过HuggingFace（或镜像站hf-mirror）直接下载BF16模型权重
| huggingface | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B |
|-------------|-------------------------------------------------------------------|
| hf-mirror | https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B|


### 三、A100环境部署模型权重
本文复现使用aisbench测评工具版本为[v3.0-20250331](https://gitee.com/aisbench/benchmark/releases/tag/v3.0-20250331)
"本地huggingface纯模型部署场景"和"vllm服务化部署场景"部署模型权重方式不同
#### 本地huggingface纯模型部署场景
1. 安装aisbench 测评工具
```
conda create --name ais_bench python=3.10 -y
conda activate ais_bench

git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./
pip3 install -r requirements/api.txt
```
安装aisbench评测工具过程中安装的依赖足以满足本地huggingface纯模型部署。

2. 将原始权重文件夹`DeepSeek-R1-Distill-Qwen-14B/`部署至服务器上

#### vllm服务化部署场景
1. 安装aisbench 测评工具
```
conda create --name ais_bench python=3.10 -y
conda activate ais_bench

git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./
pip3 install -r requirements/api.txt
```
2. 安装vllm
```
pip3 install vllm==0.6.6.post1
```

3. 将原始权重文件夹`DeepSeek-R1-Distill-Qwen-14B/`部署至服务器上

4. 准备vllm服务启动脚本
创建`server_lanuch.sh`脚本，脚本内容为：
```shell
#!bin/bash
CUDA_VISIBLE_DEVICES=$1 python3 -m vllm.entrypoints.openai.api_server \
        --model $2 \
        --host 51.62.5.35 \
        --port $3 \
        --dtype auto \
        --gpu-memory-utilization 0.8 \
        --max-num-seqs 128 \
        --max-model-len 40000 \
        --tensor-parallel-size $4 \
        --trust-remote-code
```
详细的vllm兼容openai服务化的命令参考[https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

5. 启动vllm 服务
执行如下命令启动服务：
```bash
# bash server_lanuch <gpu卡号> <DeepSeek-R1-Distill-Qwen-14B/权重路径> <服务化端口号> <总卡数>
bash server_lanuch.sh 0 /data/DeepSeek-R1-Distill-Qwen-14B/ 8081 1
```


### 四、数据集测评流程
#### 4.1 准备AIME 2024和MATH-500数据集
1) aime 2024

```
# 确保处于源码最外层路径your/work/dir/benchmark下
cd ais_bench/datasets
mkdir aime
cd aime
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip
unzip http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip
cd ../../../ # 回到源码最外层路径your/work/dir/benchmark下
```
2) math-500

```
# 确保处于源码最外层路径your/work/dir/benchmark下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
unzip http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
cd ../../ # 回到源码最外层路径your/work/dir/benchmark下
```

#### 4.2 修改AIME 2024 和MATH-500数据集配置文件
1) aime 2024
推理后端配置文件中包含数据集如何处理相关的信息，执行如下命令编辑对应的配置文件（.py文件）

```
# 确保处于源码最外层路径your/work/dir/benchmark下
vim ais_bench/benchmark/configs/datasets/aime2024/aime2024_gen_0_shot_chat_prompt.py
```
数据集配置文件内容修改如下：

```
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import Aime2024Dataset, MATHEvaluator, math_postprocess_v2


aime2024_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer'
)


aime2024_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.'
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, batch_size=4) # batch_size表示并行发送的请求数
)

aime2024_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2)
)

aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        path='ais_bench/datasets/aime/aime.jsonl', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=aime2024_reader_cfg,
        infer_cfg=aime2024_infer_cfg,
        eval_cfg=aime2024_eval_cfg
    )
]
```
2) math-500
推理后端配置文件中包含数据集如何处理相关的信息，执行如下命令编辑对应的配置文件（.py文件）

```
# 确保处于源码最外层路径your/work/dir/benchmark下
vim ais_bench/benchmark/configs/datasets/math/math500_gen_0_shot_cot_chat_prompt.py
```
数据集配置文件内容修改如下：

```
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.openicl.icl_evaluator import MATHEvaluator
math_reader_cfg = dict(input_columns=['problem'], output_column='solution')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{problem}\nPlease reason step by step, and put your final answer within \\boxed{}.'
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, batch_size=16), # batch_size表示并行发送的请求数
)

math_eval_cfg = dict(evaluator=dict(type=MATHEvaluator))

math_datasets = [
    dict(
        type=CustomDataset,
        abbr='math_prm800k_500',
        path='ais_bench/datasets/math',  # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        file_name='test_prm800k_500.jsonl', # 使用math中的math500
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
```

#### 4.3 修改模型推理后端配置文件
"本地huggingface纯模型部署场景"和"vllm服务化部署场景"配置文件不同。
##### 本地huggingface模型推理后端
推理后端配置文件中包含本地huggingface模型相关的配置，执行如下命令编辑对应的配置文件（.py文件）

```
# 确保处于源码最外层路径your/work/dir/benchmark下
vim ais_bench/benchmark/configs/models/hf_model/hf_chat_model.py
```
推理后端配置文件内容修改如下:
```py
from ais_bench.benchmark.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate, # transformers >= 4.33.0 用这个，prompt 是构造成对话格式
        attr="local", # local or service
        abbr='hf-chat-model',
        path='/data/DeepSeek-R1-Distill-Qwen-14B/', # 模型权重路径，按实际路径配置
        tokenizer_path='/data/DeepSeek-R1-Distill-Qwen-14B/', # tokenizer文件路径，一般维持和模型权重路径一致，按实际路径配置
        model_kwargs=dict( # 模型参数参考 huggingface.co/docs/transformers/v4.50.0/en/model_doc/auto#transformers.AutoModel.from_pretrained
            device_map='auto', # 自动识别使用的gpu卡号
        ),
        tokenizer_kwargs=dict( # tokenizer参数参考 huggingface.co/docs/transformers/v4.50.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase
            padding_side='left',
        ),
        generation_kwargs = dict( # 后处理参数参考huggingface.co/docs/transformers/main_classes/test_generation
            temperature = 0.5, # 论文推荐取值0.5~0.7
            top_p = 0.95,
            do_sample = True,
        ),
        run_cfg = dict(num_gpus=1, num_procs=1),  # 多卡/多机多卡 参数，使用torchrun拉起任务
        max_out_len=32768, # 最大输出tokens长度配置成32K
        max_seq_len=2048,
    )
]
```

##### vllm服务化推理后端
推理后端配置文件中包含访问vllm服务相关的请求配置，执行如下命令编辑对应的配置文件（.py文件）

```
# 确保处于源码最外层路径your/work/dir/benchmark下
vim ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
```
推理后端配置文件内容修改如下

```py
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        type=VLLMCustomAPIChat,
        attr="service", # local or service
        abbr='vllm-api-general-chat',
        max_seq_len = 4096,
        query_per_second = 1,
        rpm_verbose = False,
        retry = 2,
        host_ip = "xxx.xxx.xxx.xx", # 推理服务的IP，按实际情况配置
        host_port = 8081, # 推理服务的端口，按实际情况配置
        enable_ssl = False,
        max_out_len = 32768, # 最大输出tokens长度配置成32K
        generation_kwargs = dict( # 后处理参数参考https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-params 中的Parameters，本次测试的配置如下
            temperature = 0.6, # 论文推荐取值0.5~0.7，论文采用0.6
            top_p = 0.95,
        )
    )
]
```

#### 4.4 启动评测
执行如下命令启动评测：
##### aime 2024

**本地huggingface部署的模型**

```bash
CUDA_VISIBLE_DEVICES=0 ais_bench --models hf_chat_model --datasets aime2024_gen_0_shot_chat_prompt --summarizer example
```
在执行命令路径"outputs/default/<时间戳>"中保存了详细的过程日志和结果文件，可以执行如何命令查看推理进展：

```bash
tail -f outputs/default/<时间戳>/logs/infer/hf-chat-model/aime2024.out
```

**vllm服务化部署的模型**

```bash
ais_bench --models vllm_api_general_chat --datasets aime2024_gen_0_shot_chat_prompt --summarizer example
```
在执行命令路径"outputs/default/<时间戳>"中保存了详细的过程日志和结果文件，可以执行如何命令查看推理进展：

```bash
tail -f outputs/default/<时间戳>/logs/infer/vllm-api-general-chat/aime2024.out
```

##### math-500

**本地huggingface部署的模型**

```bash
CUDA_VISIBLE_DEVICES=0 ais_bench --models hf_chat_model --datasets math500_gen_0_shot_cot_chat_prompt --summarizer example
```
在执行命令路径"outputs/default/<时间戳>"中保存了详细的过程日志和结果文件，可以执行如何命令查看推理进展：

```bash
tail -f outputs/default/<时间戳>/logs/infer/hf-chat-model/math_prm800k_500.out
```

**vllm服务化部署的模型**

```bash
ais_bench --models vllm_api_general_chat --datasets math500_gen_0_shot_cot_chat_prompt --summarizer example
```
在执行命令路径"outputs/default/<时间戳>"中保存了详细的过程日志和结果文件，可以执行如何命令查看推理进展：

```bash
tail -f outputs/default/<时间戳>/logs/infer/vllm-api-general-chat/math_prm800k_500.out
```


**注意**，若环境已配置代理，需执行以下命令使代理对推理服务IP(主节点IP)不生效

```
export no_proxy="xxx.xxx.xxx.xx"
```
#### 4.5 查看评测结果
评测结果会直接打屏。同时可以在路径"outputs/default/<时间戳>"中结果文件和过程日志详细结构如下，可以查看详细的推理结果和评测过程

outputs/default/
├── <时间戳>     # 每个实验一个文件夹
│   ├── configs         # 用于记录的已转储的配置文件。如果在同一个实验文件夹中重新运行了不同的实验，可能会保留多个配置
│   ├── logs            # 推理和评估阶段的日志文件
│   │   ├── eval       # 评测过程日志
│   │   └── infer      # 推理过程日志
│   ├── predictions   # 每个任务的推理结果
│   ├── results       # 每个任务的评估结果
│   └── summary       # 单个实验的汇总评估结果
├── ...

### 五、测评结果验证
#### 本地huggingface部署模型评测结果
math-500：
![输入图片说明](https://foruda.gitee.com/images/1744968754412232742/0d0d8d1e_14098946.png "屏幕截图")
aime2024：
![输入图片说明](https://foruda.gitee.com/images/1744683444189768709/def50591_14098946.png "屏幕截图")

#### vllm服务化部署模型评测结果
math-500：
![输入图片说明](https://foruda.gitee.com/images/1744448811458856422/054afb66_14098946.png "屏幕截图")
aime2024：
![输入图片说明](https://foruda.gitee.com/images/1744591775768707257/01579414_14098946.png "屏幕截图")


### 六、复现结果


注：AISBench跑出的metric accuracy=80.00等价于跑一次的pass@1，DeepSeek R1论文中的pass@1得分是79.8（temperature=0.6,top_k=0.95，取k次结果的均值，k的范围为4~64），下图为论文原文。
![输入图片说明](https://foruda.gitee.com/images/1744279685398671021/03056c3c_15694876.png "屏幕截图")