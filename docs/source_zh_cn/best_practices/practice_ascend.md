# 基于昇腾800I-A2测评DeepSeek-R1数学能力，100%论文复现
### 复现使用的aisbench评测工具版本
本文复现使用aisbench测评工具版本为[v3.0-20250331](https://gitee.com/aisbench/benchmark/releases/tag/v3.0-20250331)
### 一 背景与目标

#### 1. 1复现意义

##### 1.1.1 DeepSeek-R1的数学推理优势
DeepSeek-R1作为专精数学推理任务的大模型，在复杂数学问题求解（如定理证明、多步推导）中展现出显著的性能优势。其官方评测在AIME2024（美国数学邀请赛题型，30道题）和MATH-500（500道国际竞赛级数学题）数据集上的表现，验证了其在长序列逻辑推理、符号运算精度上的领先性。

##### 1.1.2 昇腾800I-A2推理卡的软硬件支撑价值
昇腾800I-A2推理卡基于达芬奇架构，提供高性能INT8/FP16混合精度算力（典型场景下算力达XX TOPS）和超低推理延迟（微秒级响应），MindIE（Mind Inference Engine，昇腾推理引擎）支持Transformer推理加速库（Ascend Transformer Boost）以及PD分离部署等关键特性，能够高效支持DeepSeek-R1这类大模型的密集计算需求。
![输入图片说明](https://foruda.gitee.com/images/1744444409604214296/cec0c242_14098946.png "屏幕截图")

#### 1.2 复现目标

通过AISBench评测工具，在昇腾A2推理卡上完整复现DeepSeek-R1的官方测评分数，具体指标如下：
AIME2024数据集：
官方准确率：Pass@1 79.8分
昇腾A2目标：误差控制在1道题以内。
MATH-500数据集：
官方准确率：Pass@1 97.3分
昇腾A2目标：误差控制在%0.5以内。
![输入图片说明](https://foruda.gitee.com/images/1744277500333819384/82167a6f_15694876.png "屏幕截图")
####  1.3 技术价值

验证昇腾硬件与复杂数学模型的兼容性：
通过复现过程，证明昇腾A2推理卡在符号计算、长序列并行处理等场景下的技术成熟度。
提供国产化部署标杆案例：
为国产AI芯片支持国际顶尖数学推理模型提供可复现的技术路径，降低对GPU生态的依赖。

### 二、昇腾800I-A2推理卡环境准备

#### 2.1环境配置概览
| 组件         | 要求 | 备注      |
|------------|----|---------|
| 推理硬件型号      |  Atlas 800I  A2  | 单卡64G显存 |
| MindIE镜像版本 |   MindIE T6 B022 |         |
| 量化方式       |  W8A8  |      [MindStudio ModelSlim，昇腾模型压缩工具](https://gitee.com/ascend/msit/tree/master/msmodelslim)   |
|    测评工具          |  AISbench  |     [AISBench benchmark大模型评测工具](https://gitee.com/aisbench/benchmark/tree/master)    |

#### 2.2 环境配置
##### 2.2.1 下载权重
通过HuggingFace，ModelScope等开源社区直接下载BF16模型权重
| huggingface | https://huggingface.co/unsloth/DeepSeek-R1-bf16/                  |
|-------------|-------------------------------------------------------------------|
| modelscope  | https://modelscope.cn/models/unsloth/deepseek-R1-bf16/ |
##### 2.2.2 权重量化（BF16 to INT8）
[使用modelslim生成量化权重](https://gitee.com/ascend/msit/tree/br_noncom_MindStudio_8.0.0_POC_20251231/msmodelslim/example/DeepSeek)（在mindIE容器中执行，容器创建见2.4节）

```
git clone https://gitee.com/ascend/msit.git

cd msit/msmodelslim/example/DeepSeek/

python3 quant_deepseek_w8a8.py --model_path {浮点权重路径} --save_path {W8A8量化权重路径} --bf16
```
修改权重路径执行权限为750

```
chmod -R 750 {/路径/weights}
```
##### 2.2.3 生成ranktable
1) 查看8卡ip

```
for i in {0..7};do hccn_tool -i $i -ip -g; done
```

2)（可选）若没有配置8卡ip，按以下步骤自定义卡ip (需将10.20.3.13*替换为实际IP)

```
for i in {0..7}; do hccn_tool -i ${i} -ip -s address 10.20.3.13${i} netmask 255.255.255.0; done
```

3) 然后要检查对应的卡能否ping通（以双机为例）

```
hccn_tool -i 0 -ping -g address 10.20.0.20
hccn_tool -i 1 -ping -g address 10.20.0.21
hccn_tool -i 2 -ping -g address 10.20.0.22
hccn_tool -i 3 -ping -g address 10.20.0.23
hccn_tool -i 4 -ping -g address 10.20.0.24
hccn_tool -i 5 -ping -g address 10.20.0.25
hccn_tool -i 6 -ping -g address 10.20.0.26
hccn_tool -i 7 -ping -g address 10.20.0.27
```

根据步骤1获取的ip信息，参考以下双机用例，用户自行添加ip，补全device_ip，其中server_id和container_ip均为机器IP：

```
{
   "server_count": "2",
   "server_list": [
      {
         "device": [
            {
               "device_id": "0",
               "device_ip": "...",
               "rank_id": "0"
            },
            ...
            {
               "device_id": "7",
               "device_ip": "...",
               "rank_id": "7"
            },
         ],
         "server_id": "...",
         "container_ip": "..."
      },
      {
         "device": [
            {
               "device_id": "0",
               "device_ip": "...",
               "rank_id": "8"
            },
            ...
            {
               "device_id": "7",
               "device_ip": "...",
               "rank_id": "15"
            },
         ],
         "server_id": "...",
         "container_ip": "..."
      },
   ],
   "status": "completed",
   "version": "1.0"
}
```

rank_table_file.json配置完成后，需要执行命令修改权限为640。

```
chmod -R 640 {路径/rank_table_file.json}
```
##### 2.2.4 镜像配置
1）下载镜像
[镜像下载说明](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f) 下载镜像前需要申请权限，耐心等待权限申请通过后，根据指南下载对应镜像文件。

2）加载镜像

```
docker load < **800I-A2-py311-ubuntu22.04-aarch64.tar.gz
```
完成之后，请使用docker images命令确认查找具体镜像名称与标签

```
docker images
```
3)创建及启动容器
将以下内容写入start-docker.sh中，

```
IMAGES_ID=$1
NAME=$2
if [ $# -ne 2 ]; then
    echo "error: need one argument describing your container name."
    exit 1
fi
docker run --name ${NAME} -it -d --net=host --shm-size=500g \
    --privileged=true \
    -w /home \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --entrypoint=bash \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /tmp:/tmp \
    -v /dl:/dl \
    -v /mnt:/mnt \
    -v /data:/data \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
${IMAGES_ID}
```
执行如下命令创建容器，

```
bash start-docker.sh imagesID 容器名
```
执行如下命令进入容器，

```
docker exec -it {容器名称} bash
```
##### 2.2.5 启动推理服务
将以下内容写入set_env.sh，

```
export RANKTABLEFILE="/路径/ranktable.json"    #对应路径修改
export MIES_CONTAINER_IP=xx.xx.xxx.xxx     #修改为本机IP
export MINDIE_LLM_LOG_TO_STDOUT=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ATB_WORKSPACE_MEM_ALLOC_ALG_TYPE=3
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=1
export OMP_NUM_THREADS=1
export HCCL_DETERMINISTIC=false
export HCCL_OP_EXPANSION_MODE="AIV"
export NPU_MEMORY_FRACTION=0.96
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=0
source /usr/local/Ascend/mindie/set_env.sh
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/atb-models/set_env.sh
export ATB_LLM_ENABLE_AUTO_TRANSPOSE=0
export MINDIE_LOG_TO_STDOUT=1
export MASTER_IP=xx.xx.xxx.xxx        # 修改为主节点IP

for var in $(compgen -e | grep 'STDOUT$'); do
  export "$var=0"
done


for var in $(compgen -e | grep 'LOG_TO_FILE$'); do
  export "$var=0"
done

export INF_NAN_MODE_ENABLE=0

# 修改权限
find /usr/local/lib/python3.11/site-packages/mindie*  -name  config.json |xargs chmod -R 640
```
执行以下命令加载环境变量

```
source /路径/set_env.sh
```
修改路径下/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json服务化参数配置，示例如下：

```
{
    "Version" : "1.0.0",
    "LogConfig" :
    {
        "logLevel" : "Info",
        "logFileSize" : 20,
        "logFileNum" : 20,
        "logPath" : "logs/mindie-server.log"
    },

    "ServerConfig" :
    {
        "ipAddress" : "127.0.0.1",
        "managementIpAddress" : "127.0.0.2",
        "port" : 1025,
        "managementPort" : 1026,
        "metricsPort" : 1027,
        "allowAllZeroIpListening" : false,
        "maxLinkNum" : 1000,
        "httpsEnabled" : false,
        "fullTextEnabled" : false,
        "tlsCaPath" : "security/ca/",
        "tlsCaFile" : ["ca.pem"],
        "tlsCert" : "security/certs/server.pem",
        "tlsPk" : "security/keys/server.key.pem",
        "tlsPkPwd" : "security/pass/key_pwd.txt",
        "tlsCrlPath" : "security/certs/",
        "tlsCrlFiles" : ["server_crl.pem"],
        "managementTlsCaFile" : ["management_ca.pem"],
        "managementTlsCert" : "security/certs/management/server.pem",
        "managementTlsPk" : "security/keys/management/server.key.pem",
        "managementTlsPkPwd" : "security/pass/management/key_pwd.txt",
        "managementTlsCrlPath" : "security/management/certs/",
        "managementTlsCrlFiles" : ["server_crl.pem"],
        "kmcKsfMaster" : "tools/pmt/master/ksfa",
        "kmcKsfStandby" : "tools/pmt/standby/ksfb",
        "inferMode" : "standard",
        "interCommTLSEnabled" : false,
        "interCommPort" : 1121,
        "interCommTlsCaPath" : "security/grpc/ca/",
        "interCommTlsCaFiles" : ["ca.pem"],
        "interCommTlsCert" : "security/grpc/certs/server.pem",
        "interCommPk" : "security/grpc/keys/server.key.pem",
        "interCommPkPwd" : "security/grpc/pass/key_pwd.txt",
        "interCommTlsCrlPath" : "security/grpc/certs/",
        "interCommTlsCrlFiles" : ["server_crl.pem"],
        "openAiSupport" : "vllm",
        "tokenTimeout" : 3600,
        "e2eTimeout" : 3600
    },

    "BackendConfig" : {
        "backendName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "npuDeviceIds" : [[0,1,2,3,4,5,6,7]],
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled" : true,
        "multiNodesInferPort" : 1120,
        "interNodeTLSEnabled" : false,
        "interNodeTlsCaPath" : "security/grpc/ca/",
        "interNodeTlsCaFiles" : ["ca.pem"],
        "interNodeTlsCert" : "security/grpc/certs/server.pem",
        "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
        "interNodeTlsPkPwd" : "security/grpc/pass/mindie_server_key_pwd.txt",
        "interNodeTlsCrlPath" : "security/grpc/certs/",
        "interNodeTlsCrlFiles" : ["server_crl.pem"],
        "interNodeKmcKsfMaster" : "tools/pmt/master/ksfa",
        "interNodeKmcKsfStandby" : "tools/pmt/standby/ksfb",
        "ModelDeployConfig" :
        {
            "maxSeqLen" : 32768,
            "maxInputTokenLen" : 1536,
            "truncation" : false,
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "ds_r1",
                    "modelWeightPath" : "/路径/weight/", # 修改为真实权重路径
                    "worldSize" : 8,
                    "cpuMemSize" : 5,
                    "npuMemSize" : -1,
                    "backendType" : "atb",
                    "trustRemoteCode" : false,
                    "tp":4,
                    "dp":4,
                    "moe_ep":4,
                    "moe_tp":4
                }
            ]
        },

        "ScheduleConfig" :
        {
            "templateType" : "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 128,

            "maxPrefillBatchSize" : 1,
            "maxPrefillTokens" : 4096,
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,

            "maxBatchSize" : 2048,
            "maxIterTimes" : 31232,
            "maxPreemptCount" : 0,
            "supportSelectBatch" : false,
            "maxQueueDelayMicroseconds" : 5000
        }
    }
}
```
同时在双机/usr/local/Ascend/mindie/latest/mindie-service/路径下，执行以下命令拉起推理服务

```
./bin/mindieservice_daemon
```
打印以下命令，则认为服务拉起成功

"Daemon start success!"

##### 2.2.6 AISbench测评工具安装
在镜像外，安装aisbench 测评工具

```
conda create --name ais_bench python=3.10 -y
conda activate ais_bench

git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./
pip3 install -r requirements/api.txt
```


### 三、数据集测评流程
#### 3.1 准备AIME 2024和MATH-500数据集
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
#### 3.2 修改模型推理后端配置文件
推理后端配置文件中包含访问MindIE-Service相关的请求配置，执行如下命令编辑对应的配置文件（.py文件）

```
# 确保处于源码最外层路径your/work/dir/benchmark下
vim ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
```
推理后端配置文件内容修改如下

```
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        max_seq_len = 4096,
        query_per_second = 1,
        rpm_verbose = False,
        retry = 2,
        host_ip = "xxx.xxx.xxx.xx", # 推理服务的IP，按实际情况配置
        host_port = 1025, # 推理服务的端口，按实际情况配置
        enable_ssl = False,
        max_out_len = 32768, # 最大输出tokens长度配置成32K
        generation_kwargs = dict( # 后处理参数参考https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-params 中的Parameters，本次测试的配置如下
            temperature = 0,
            top_p = 0.95,
            seed = None,
        )
    )
]
```
#### 3.3 修改AIME 2024 和MATH-500数据集配置文件
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
#### 3.4 启动评测
执行如下命令启动评测：
1) aime 2024

```
ais_bench --models vllm_api_general_chat --datasets aime2024_gen_0_shot_chat_prompt --summarizer example
```
在执行命令路径"outputs/default/<时间戳>"中保存了详细的过程日志和结果文件，可以执行如何命令查看推理进展：

```
tail -f outputs/default/<时间戳>/logs/infer/vllm-api-general-chat/aime2024.out
```

2) math-500

```
ais_bench --models vllm_api_general_chat --datasets math500_gen_0_shot_cot_chat_prompt --summarizer example
```
在执行命令路径"outputs/default/<时间戳>"中保存了详细的过程日志和结果文件，可以执行如何命令查看推理进展：

```
tail -f outputs/default/<时间戳>/logs/infer/vllm-api-general-chat/math_prm800k_500.out
```
注意，若环境已配置代理，需执行以下命令使代理对推理服务IP(主节点IP)不生效

```
export no_proxy="xxx.xxx.xxx.xx"
```
#### 3.5 查看评测结果
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

### 四、测评结果验证
评测结果如下图所示：
![输入图片说明](https://foruda.gitee.com/images/1744279619126000415/b440c50e_15694876.png "屏幕截图")

![输入图片说明](https://foruda.gitee.com/images/1744279635756636101/b60a95f6_15694876.png "屏幕截图")

### 五、复现结果

本研究通过昇腾800I-A2推理卡与AISBench评测工具的深度协同，在DeepSeek-R1数学推理模型的复现中达成核心目标 ，关键成果如下：
AIME2024数据集 ：实现Pass@1 80.0分 （官方79.8分），误差仅0.2分（＜1题）；
MATH-500数据集 ：达成Pass@1 97.6% （官方97.3%），误差0.3%（＜0.5%目标）。
精准复现论文指标，证明了昇腾800I-A2推理卡在符号计算、长序列并行处理等场景下的技术成熟度。
| 数据集       | 官方准确率 | 昇腾A2准确率 | 误差  | 复现结果 |
|-----------|-------|---------|-----|------|
| aime 2024 | 79.8  | 80.00   | 0.2 | pass |
| math-500  | 97.3  | 97.6    | 0.3 | pass |

注：AISBench跑出的metric accuracy=80.00等价于跑一次的pass@1，DeepSeek R1论文中的pass@1得分是79.8（temperature=0.6,top_k=0.95，取k次结果的均值，k的范围为4~64），下图为论文原文。
![输入图片说明](https://foruda.gitee.com/images/1744279685398671021/03056c3c_15694876.png "屏幕截图")