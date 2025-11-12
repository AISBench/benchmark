# Evaluating DeepSeek-R1's Mathematical Capabilities Based on Ascend 800I-A2: 100% Paper Reproduction
### Version of AISBench Evaluation Tool Used for Reproduction
The version of the AISBench evaluation tool used for reproduction in this paper is [v3.0-20250331](https://gitee.com/aisbench/benchmark/releases/tag/v3.0-20250331).

### I. Background and Objectives
#### 1.1 Significance of Reproduction
##### 1.1.1 Mathematical Reasoning Advantages of DeepSeek-R1
As a large model specialized in mathematical reasoning tasks, DeepSeek-R1 demonstrates significant performance advantages in solving complex mathematical problems (such as theorem proving and multi-step derivation). Its official evaluations on the AIME2024 (American Invitational Mathematics Examination questions, 30 questions) and MATH-500 (500 international competition-level mathematics questions) datasets verify its leadership in long-sequence logical reasoning and symbolic computation accuracy.

##### 1.1.2 Hardware and Software Support Value of Ascend 800I-A2 Inference Card
Based on the Da Vinci architecture, the Ascend 800I-A2 inference card provides high-performance INT8/FP16 mixed-precision computing power (with computing power reaching XX TOPS in typical scenarios) and ultra-low inference latency (microsecond-level response). MindIE (Mind Inference Engine, Ascend Inference Engine) supports key features such as the Transformer inference acceleration library (Ascend Transformer Boost) and PD-separated deployment, enabling efficient support for the intensive computing needs of large models like DeepSeek-R1.
![Image Description](https://foruda.gitee.com/images/1744444409604214296/cec0c242_14098946.png "Screenshot")

#### 1.2 Reproduction Objectives
Through the AISBench evaluation tool, fully reproduce the official evaluation scores of DeepSeek-R1 on the Ascend A2 inference card. The specific indicators are as follows:
- AIME2024 Dataset:
  - Official Accuracy: Pass@1 79.8
  - Ascend A2 Target: Error controlled within 1 question.
- MATH-500 Dataset:
  - Official Accuracy: Pass@1 97.3%
  - Ascend A2 Target: Error controlled within 0.5%.
![Image Description](https://foruda.gitee.com/images/1744277500333819384/82167a6f_15694876.png "Screenshot")

#### 1.3 Technical Value
- Verify the compatibility between Ascend hardware and complex mathematical models:
  Through the reproduction process, demonstrate the technical maturity of the Ascend A2 inference card in scenarios such as symbolic computation and long-sequence parallel processing.
- Provide a benchmark case for domestic deployment:
  Offer a reproducible technical path for domestic AI chips to support world-leading mathematical reasoning models, reducing reliance on the GPU ecosystem.


### II. Environment Preparation for Ascend 800I-A2 Inference Card
#### 2.1 Overview of Environment Configuration
| Component               | Requirement                | Remarks                          |
|-------------------------|----------------------------|----------------------------------|
| Inference Hardware Model| Atlas 800I A2              | Single card with 64GB VRAM       |
| MindIE Image Version    | MindIE T6 B022             |                                  |
| Quantization Method     | W8A8                       | [MindStudio ModelSlim, Ascend Model Compression Tool](https://gitee.com/ascend/msit/tree/master/msmodelslim) |
| Evaluation Tool         | AISbench                   | [AISBench Benchmark Large Model Evaluation Tool](https://gitee.com/aisbench/benchmark/tree/master) |

#### 2.2 Environment Configuration
##### 2.2.1 Download Weights
Download BF16 model weights directly from open-source communities such as HuggingFace and ModelScope.

| Platform     | Link                                                                 |
|--------------|----------------------------------------------------------------------|
| HuggingFace  | https://huggingface.co/unsloth/DeepSeek-R1-bf16/                     |
| ModelScope   | https://modelscope.cn/models/unsloth/deepseek-R1-bf16/               |

##### 2.2.2 Weight Quantization (BF16 to INT8)
[Use ModelSlim to Generate Quantized Weights](https://gitee.com/ascend/msit/tree/br_noncom_MindStudio_8.0.0_POC_20251231/msmodelslim/example/DeepSeek) (Execute in the MindIE container; see Section 2.4 for container creation).

```bash
git clone https://gitee.com/ascend/msit.git

cd msit/msmodelslim/example/DeepSeek/

python3 quant_deepseek_w8a8.py --model_path {Floating-Point Weight Path} --save_path {W8A8 Quantized Weight Path} --bf16
```

Modify the permission of the weight path to 750:
```bash
chmod -R 750 {/path/to/weights}
```

##### 2.2.3 Generate Rank Table
1) Check IP addresses of 8 cards:
```bash
for i in {0..7};do hccn_tool -i $i -ip -g; done
```

2) (Optional) If 8-card IP addresses are not configured, customize card IPs as follows (replace 10.20.3.13* with actual IPs):
```bash
for i in {0..7}; do hccn_tool -i ${i} -ip -s address 10.20.3.13${i} netmask 255.255.255.0; done
```

3) Then check if the corresponding cards can ping each other (taking two machines as an example):
```bash
hccn_tool -i 0 -ping -g address 10.20.0.20
hccn_tool -i 1 -ping -g address 10.20.0.21
hccn_tool -i 2 -ping -g address 10.20.0.22
hccn_tool -i 3 -ping -g address 10.20.0.23
hccn_tool -i 4 -ping -g address 10.20.0.24
hccn_tool -i 5 -ping -g address 10.20.0.25
hccn_tool -i 6 -ping -g address 10.20.0.26
hccn_tool -i 7 -ping -g address 10.20.0.27
```

Based on the IP information obtained in Step 1, refer to the following two-machine example. Users should add IPs and complete `device_ip` by themselves, where both `server_id` and `container_ip` are machine IPs:

```json
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
            }
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
            }
         ],
         "server_id": "...",
         "container_ip": "..."
      }
   ],
   "status": "completed",
   "version": "1.0"
}
```

After configuring `rank_table_file.json`, execute the following command to modify the permission to 640:
```bash
chmod -R 640 {path/to/rank_table_file.json}
```

##### 2.2.4 Image Configuration
1) Download the image
[Image Download Instructions](https://www.hiascend.com/developer/ascendhub/detail/af85b724a7e5469ebd7ea13c3439d48f)
Permission application is required before downloading the image. Wait patiently for the permission to be approved, then download the corresponding image file according to the guide.

2) Load the image:
```bash
docker load < **800I-A2-py311-ubuntu22.04-aarch64.tar.gz
```

After completion, use the `docker images` command to confirm and find the specific image name and tag:
```bash
docker images
```

3) Create and start the container
Write the following content into `start-docker.sh`:
```bash
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

Execute the following command to create the container:
```bash
bash start-docker.sh imagesID ContainerName
```

Execute the following command to enter the container:
```bash
docker exec -it {Container Name} bash
```

### 2.2.5 Starting the Inference Service
Write the following content into set_env.sh:

```
export RANKTABLEFILE="/path/ranktable.json"    # Modify the corresponding path
export MIES_CONTAINER_IP=xx.xx.xxx.xxx     # Modify to the local IP
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
export MASTER_IP=xx.xx.xxx.xxx        # Modify to the master node IP

for var in $(compgen -e | grep 'STDOUT$'); do
  export "$var=0"
done


for var in $(compgen -e | grep 'LOG_TO_FILE$'); do
  export "$var=0"
done

export INF_NAN_MODE_ENABLE=0

# Modify permissions
find /usr/local/lib/python3.11/site-packages/mindie*  -name  config.json |xargs chmod -R 640
```
Execute the following command to load the environment variables:

```
source /path/set_env.sh
```
Modify the service - oriented parameter configuration in the /usr/local/Ascend/mindie/latest/mindie - service/conf/config.json file. The example is as follows:

```
{
    "Version" : "1.0.0",
    "LogConfig" :
    {
        "logLevel" : "Info",
        "logFileSize" : 20,
        "logFileNum" : 20,
        "logPath" : "logs/mindie - server.log"
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
                    "modelWeightPath" : "/path/weight/", # Modify to the real weight path
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
At the same time, in the /usr/local/Ascend/mindie/latest/mindie - service/ path on the dual - machine, execute the following command to start the inference service:

```
./bin/mindieservice_daemon
```
If the following command is printed, it is considered that the service has been successfully started:

"Daemon start success!"

##### 2.2.6 Installation of AISbench Evaluation Tool
Outside the image, install the aisbench evaluation tool:

```
conda create --name ais_bench python=3.10 -y
conda activate ais_bench

git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./
pip3 install -r requirements/api.txt
```

### III. Dataset Evaluation Process
#### 3.1 Preparing AIME 2024 and MATH - 500 Datasets
1) aime 2024

```
# Ensure that you are at the outermost path of the source code your/work/dir/benchmark
cd ais_bench/datasets
mkdir aime
cd aime
wget http://opencompass.oss - cn - shanghai.aliyuncs.com/datasets/data/aime.zip
unzip http://opencompass.oss - cn - shanghai.aliyuncs.com/datasets/data/aime.zip
cd ../../../ # Return to the outermost path of the source code your/work/dir/benchmark
```
2) math - 500

```
# Ensure that you are at the outermost path of the source code your/work/dir/benchmark
cd ais_bench/datasets
wget http://opencompass.oss - cn - shanghai.aliyuncs.com/datasets/data/math.zip
unzip http://opencompass.oss - cn - shanghai.aliyuncs.com/datasets/data/math.zip
cd ../../ # Return to the outermost path of the source code your/work/dir/benchmark
```
#### 3.2 Modifying the Model Inference Back - end Configuration File
The inference back - end configuration file contains the request configuration related to accessing MindIE - Service. Execute the following command to edit the corresponding configuration file (.py file):

```
# Ensure that you are at the outermost path of the source code your/work/dir/benchmark
vim ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
```
The content of the inference back - end configuration file is modified as follows:

```
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        type=VLLMCustomAPIChat,
        abbr='vllm - api - general - chat',
        max_seq_len = 4096,
        query_per_second = 1,
        rpm_verbose = False,
        retry = 2,
        host_ip = "xxx.xxx.xxx.xx", # Inference service IP, configure according to the actual situation
        host_port = 1025, # Inference service port, configure according to the actual situation
        enable_ssl = False,
        max_out_len = 32768, # Maximum output tokens length is configured to 32K
        generation_kwargs = dict( # Post - processing parameters refer to Parameters in https://docs.vllm.ai/en/latest/api/inference_params.html#sampling - params. The configuration for this test is as follows
            temperature = 0,
            top_p = 0.95,
            seed = None,
        )
    )
]
```
#### 3.3 Modifying the AIME 2024 and MATH - 500 Dataset Configuration Files
1) aime 2024
The inference back - end configuration file contains information related to how the dataset is processed. Execute the following command to edit the corresponding configuration file (.py file):

```
# Ensure that you are at the outermost path of the source code your/work/dir/benchmark
vim ais_bench/benchmark/configs/datasets/aime2024/aime2024_gen_0_shot_chat_prompt.py
```
The content of the dataset configuration file is modified as follows:

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
    inferencer=dict(type=GenInferencer, batch_size=4) # batch_size represents the number of parallel requests sent
)

aime2024_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2)
)

aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        path='ais_bench/datasets/aime/aime.jsonl', # Dataset path. When using a relative path, it is relative to the root path of the source code. Absolute paths are also supported
        reader_cfg=aime2024_reader_cfg,
        infer_cfg=aime2024_infer_cfg,
        eval_cfg=aime2024_eval_cfg
    )
]
```
2) math - 500
The inference back - end configuration file contains information related to how the dataset is processed. Execute the following command to edit the corresponding configuration file (.py file):

```
# Ensure that you are at the outermost path of the source code your/work/dir/benchmark
vim ais_bench/benchmark/configs/datasets/math/math500_gen_0_shot_cot_chat_prompt.py
```
The content of the dataset configuration file is modified as follows:

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
    inferencer=dict(type=GenInferencer, batch_size=16), # batch_size represents the number of parallel requests sent
)

math_eval_cfg = dict(evaluator=dict(type=MATHEvaluator))

math_datasets = [
    dict(
        type=CustomDataset,
        abbr='math_prm800k_500',
        path='ais_bench/datasets/math',  # Dataset path. When using a relative path, it is relative to the root path of the source code. Absolute paths are also supported
        file_name='test_prm800k_500.jsonl', # Use math500 in math
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
```
#### 3.4 Starting the Evaluation
Execute the following commands to start the evaluation:
1) aime 2024

```
ais_bench --models vllm_api_general_chat --datasets aime2024_gen_0_shot_chat_prompt --summarizer example
```
Detailed process logs and result files are saved in the execution command path "outputs/default/<timestamp>". You can execute the following command to view the inference progress:

```
tail -f outputs/default/<timestamp>/logs/infer/vllm - api - general - chat/aime2024.out
```

2) math - 500

```
ais_bench --models vllm_api_general_chat --datasets math500_gen_0_shot_cot_chat_prompt --summarizer example
```
Detailed process logs and result files are saved in the execution command path "outputs/default/<timestamp>". You can execute the following command to view the inference progress:

```
tail -f outputs/default/<timestamp>/logs/infer/vllm - api - general - chat/math_prm800k_500.out
```
Note that if the environment has a proxy configured, you need to execute the following command to make the proxy ineffective for the inference service IP (master node IP):

```
export no_proxy="xxx.xxx.xxx.xx"
```
#### 3.5 Viewing Evaluation Results
The evaluation results will be printed directly on the screen. Meanwhile, the detailed structure of result files and process logs in the path "outputs/default/<timestamp>" is as follows, where you can view the detailed inference results and evaluation process:

outputs/default/
├── <timestamp>     # One folder for each experiment
│   ├── configs         # Dumped configuration files for recording. Multiple configurations may be retained if different experiments are re-run in the same experiment folder
│   ├── logs            # Log files for the inference and evaluation phases
│   │   ├── eval       # Logs of the evaluation process
│   │   └── infer      # Logs of the inference process
│   ├── predictions   # Inference results for each task
│   ├── results       # Evaluation results for each task
│   └── summary       # Aggregated evaluation results of a single experiment
├── ...


### IV. Verification of Evaluation Results
The evaluation results are shown in the figures below:
![Image Description](https://foruda.gitee.com/images/1744279619126000415/b440c50e_15694876.png "Screenshot")

![Image Description](https://foruda.gitee.com/images/1744279635756636101/b60a95f6_15694876.png "Screenshot")


### V. Reproduction Results
Through the in-depth collaboration between the Ascend 800I-A2 inference card and the AISBench evaluation tool, this study has achieved the core goal in reproducing the DeepSeek-R1 mathematical reasoning model. The key achievements are as follows:
- AIME2024 Dataset: Achieved Pass@1 80.0 points (official score: 79.8 points), with an error of only 0.2 points (＜1 question);
- MATH-500 Dataset: Achieved Pass@1 97.6% (official accuracy: 97.3%), with an error of 0.3% (＜0.5% target).

The accurate reproduction of the paper's indicators proves the technical maturity of the Ascend 800I-A2 inference card in scenarios such as symbolic computation and long-sequence parallel processing.

| Dataset    | Official Accuracy | Ascend A2 Accuracy | Error | Reproduction Result |
|------------|-------------------|--------------------|-------|--------------------|
| aime 2024  | 79.8              | 80.00              | 0.2   | pass               |
| math-500   | 97.3%             | 97.6%              | 0.3%  | pass               |

Note: The metric "accuracy = 80.00" obtained by AISBench is equivalent to Pass@1 in a single run. The Pass@1 score of DeepSeek R1 in the paper is 79.8 (with temperature = 0.6, top_k = 0.95, and the average value of k runs, where k ranges from 4 to 64). The original paper content is shown in the figure below:
![Image Description](https://foruda.gitee.com/images/1744279685398671021/03056c3c_15694876.png "Screenshot")