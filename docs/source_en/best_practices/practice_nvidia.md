# Evaluating the Mathematical Capabilities of DeepSeek-R1-Distill-Qwen-14B Based on NVIDIA A100 Accelerator Card: 100% Paper Reproduction
### Version of AISBench Evaluation Tool Used for Reproduction
The version of the AISBench evaluation tool used for reproduction in this paper is [v3.0-20250412](https://gitee.com/aisbench/benchmark/releases/tag/v3.0-20250412).


### I. Background and Objectives
#### 1.1 Significance of Reproduction
##### 1.1.1 Small Models Distilled from DeepSeek-R1 Inherit Its Mathematical Reasoning Advantages
As a large model specialized in mathematical reasoning tasks, DeepSeek-R1 demonstrates significant performance advantages in solving complex mathematical problems (such as theorem proving and multi-step derivation). Its official evaluations on the AIME2024 (American Invitational Mathematics Examination questions, 30 questions) and MATH-500 (500 international competition-level mathematics questions) datasets have verified its leadership in long-sequence logical reasoning and symbolic computation accuracy. Officially confirmed, the reasoning paradigm of the large DeepSeek-R1 model can be transferred to small models through distillation technology. Compared with the reasoning method where small models independently explore via reinforcement learning, this approach achieves more outstanding performance. Small models after distillation also inherit the advantages of DeepSeek-R1 in solving complex mathematical problems, demonstrating breakthrough performance.

##### 1.1.2 Low Reproduction Threshold for Small Models Distilled from DeepSeek-R1
Compared with the DeepSeek-R1 model with 671B parameters, small models distilled from DeepSeek-R1 have a lower hardware threshold for deployment. For example, the non-quantized version of DeepSeek-R1-Distill-Qwen-14B only requires 24GB of VRAM and can be deployed on a single A100 card. This enables easy local deployment and service-oriented deployment, making it flexible for enterprise private scenarios.

#### 1.2 Reproduction Objectives
Through the AISBench evaluation tool, on a single A100 accelerator card, fully reproduce the official evaluation scores of DeepSeek-R1-Distill-Qwen-14B using both Hugging Face local deployment and vLLM service-oriented deployment methods. The specific indicators are as follows:
- AIME2024 Dataset:
  - Official Accuracy: Pass@1 69.7
  - Accuracy Reproduction Target: Error controlled within 1 question.
- MATH-500 Dataset:
  - Official Accuracy: Pass@1 93.9
  - Accuracy Reproduction Target: Error controlled within 0.5%.

![Image Description](https://foruda.gitee.com/images/1744444555077371068/751e21f1_14098946.png "Screenshot")

#### 1.3 Technical Value
Verify that AISBench supports evaluation scenarios for both large-model service-oriented deployment and local deployment, and can achieve alignment of evaluation accuracy.


### II. A100 Environment Preparation
#### 2.1 NVIDIA Hardware and Software Stack Versions
| Component       | Model/Version                                  |
|-----------------|------------------------------------------------|
| NVIDIA Hardware | Single A100 Accelerator Card (80GB VRAM)       |
| NVIDIA Driver   | 550.54.15                                      |
| CUDA Version    | 12.4                                           |

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
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
```

#### 2.2 Versions of Other Software Dependencies
Software dependencies differ between the "local Hugging Face pure model deployment scenario" and the "vLLM service-oriented deployment scenario".

##### 2.2.1 Local Hugging Face Pure Model Deployment
```shell
Python 3.10.15 (main, Oct  3 2024, 07:27:34)

torch==2.5.1
torchvision==0.20.1
transformers==4.48.1
```

##### 2.2.2 vLLM Service-Oriented Deployment
```shell
Python 3.10.15 (main, Oct  3 2024, 07:27:34)

torch==2.5.1
torchvision==0.20.1
transformers==4.48.1
vllm==0.6.6.post1
```

#### 2.3 Obtaining Original Weights of DeepSeek-R1-Distill-Qwen-14B
Download BF16 model weights directly from Hugging Face (or its mirror site hf-mirror):

| Platform    | Link                                                                 |
|-------------|----------------------------------------------------------------------|
| Hugging Face| https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B     |
| hf-mirror   | https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B       |


### III. Deploying Model Weights in A100 Environment
The version of the AISBench evaluation tool used for reproduction in this paper is [v3.0-20250331](https://gitee.com/aisbench/benchmark/releases/tag/v3.0-20250331). The methods for deploying model weights differ between the "local Hugging Face pure model deployment scenario" and the "vLLM service-oriented deployment scenario".

#### Local Hugging Face Pure Model Deployment Scenario
1. Install the AISBench evaluation tool:
```bash
conda create --name ais_bench python=3.10 -y
conda activate ais_bench

git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./
pip3 install -r requirements/api.txt
```
The dependencies installed during the AISBench installation process are sufficient for local Hugging Face pure model deployment.

2. Deploy the original weight folder `DeepSeek-R1-Distill-Qwen-14B/` to the server.


#### vLLM Service-Oriented Deployment Scenario
1. Install the AISBench evaluation tool:
```bash
conda create --name ais_bench python=3.10 -y
conda activate ais_bench

git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./
pip3 install -r requirements/api.txt
```

2. Install vLLM:
```bash
pip3 install vllm==0.6.6.post1
```

3. Deploy the original weight folder `DeepSeek-R1-Distill-Qwen-14B/` to the server.

4. Prepare the vLLM service startup script:
Create a `server_launch.sh` script with the following content:
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
For detailed commands about vLLM compatible with OpenAI service-oriented deployment, refer to [https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html).

5. Start the vLLM service:
Execute the following command to start the service:
```bash
# bash server_launch <GPU ID> <Path to DeepSeek-R1-Distill-Qwen-14B/ Weights> <Service Port Number> <Total Number of Cards>
bash server_launch.sh 0 /data/DeepSeek-R1-Distill-Qwen-14B/ 8081 1
```


### IV. Dataset Evaluation Process
#### 4.1 Preparing AIME 2024 and MATH-500 Datasets
1) AIME 2024
```bash
# Ensure you are in the outermost directory of the source code: your/work/dir/benchmark
cd ais_bench/datasets
mkdir aime
cd aime
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip
unzip http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip
cd ../../../ # Return to the outermost directory of the source code: your/work/dir/benchmark
```

2) MATH-500
```bash
# Ensure you are in the outermost directory of the source code: your/work/dir/benchmark
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
unzip http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
cd ../../ # Return to the outermost directory of the source code: your/work/dir/benchmark
```

#### 4.2 Modifying Configuration Files for AIME 2024 and MATH-500 Datasets
1) AIME 2024
The inference backend configuration file contains information about how the dataset is processed. Execute the following command to edit the corresponding configuration file (.py file):
```bash
# Ensure you are in the outermost directory of the source code: your/work/dir/benchmark
vim ais_bench/benchmark/configs/datasets/aime2024/aime2024_gen_0_shot_chat_prompt.py
```
Modify the content of the dataset configuration file as follows:
```python
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
    inferencer=dict(type=GenInferencer, batch_size=4) # batch_size indicates the number of parallel requests sent
)

aime2024_eval_cfg = dict(
    evaluator=dict(type=MATHEvaluator, version='v2'), pred_postprocessor=dict(type=math_postprocess_v2)
)

aime2024_datasets = [
    dict(
        abbr='aime2024',
        type=Aime2024Dataset,
        path='ais_bench/datasets/aime/aime.jsonl', # Dataset path. When using a relative path, it is relative to the root directory of the source code; absolute paths are also supported
        reader_cfg=aime2024_reader_cfg,
        infer_cfg=aime2024_infer_cfg,
        eval_cfg=aime2024_eval_cfg
    )
]
```

2) MATH-500
The inference backend configuration file contains information about how the dataset is processed. Execute the following command to edit the corresponding configuration file (.py file):
```bash
# Ensure you are in the outermost directory of the source code: your/work/dir/benchmark
vim ais_bench/benchmark/configs/datasets/math/math500_gen_0_shot_cot_chat_prompt.py
```
Modify the content of the dataset configuration file as follows:
```python
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
    inferencer=dict(type=GenInferencer, batch_size=16), # batch_size indicates the number of parallel requests sent
)

math_eval_cfg = dict(evaluator=dict(type=MATHEvaluator))

math_datasets = [
    dict(
        type=CustomDataset,
        abbr='math_prm800k_500',
        path='ais_bench/datasets/math',  # Dataset path. When using a relative path, it is relative to the root directory of the source code; absolute paths are also supported
        file_name='test_prm800k_500.jsonl', # Use math500 from the math dataset
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg,
    )
]
```

#### 4.3 Modifying the Model Inference Backend Configuration File
The configuration files differ between the "local Hugging Face pure model deployment scenario" and the "vLLM service-oriented deployment scenario".

##### Local Hugging Face Model Inference Backend
The inference backend configuration file contains settings related to the local Hugging Face model. Execute the following command to edit the corresponding configuration file (.py file):

```bash
# Ensure you are in the outermost directory of the source code: your/work/dir/benchmark
vim ais_bench/benchmark/configs/models/hf_model/hf_chat_model.py
```

Modify the content of the inference backend configuration file as follows:
```py
from ais_bench.benchmark.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate, # Use this for transformers >= 4.33.0; prompts are structured in conversation format
        attr="local", # local or service
        abbr='hf-chat-model',
        path='/data/DeepSeek-R1-Distill-Qwen-14B/', # Model weight path; configure according to the actual path
        tokenizer_path='/data/DeepSeek-R1-Distill-Qwen-14B/', # Tokenizer file path; generally the same as the model weight path; configure according to the actual path
        model_kwargs=dict( # For model parameters, refer to huggingface.co/docs/transformers/v4.50.0/en/model_doc/auto#transformers.AutoModel.from_pretrained
            device_map='auto', # Automatically identify the GPU ID to use
        ),
        tokenizer_kwargs=dict( # For tokenizer parameters, refer to huggingface.co/docs/transformers/v4.50.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase
            padding_side='left',
        ),
        generation_kwargs = dict( # For post-processing parameters, refer to huggingface.co/docs/transformers/main_classes/test_generation
            temperature = 0.5, # Recommended value range in the paper: 0.5~0.7
            top_p = 0.95,
            do_sample = True,
        ),
        run_cfg = dict(num_gpus=1, num_procs=1),  # Parameters for multi-GPU/multi-machine multi-GPU; use torchrun to launch the task
        max_out_len=32768, # Maximum output token length set to 32K
        max_seq_len=2048,
    )
]
```


##### vLLM Service-Oriented Inference Backend
The inference backend configuration file contains request settings related to accessing the vLLM service. Execute the following command to edit the corresponding configuration file (.py file):

```bash
# Ensure you are in the outermost directory of the source code: your/work/dir/benchmark
vim ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py
```

Modify the content of the inference backend configuration file as follows:
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
        host_ip = "xxx.xxx.xxx.xx", # IP of the inference service; configure according to the actual situation
        host_port = 8081, # Port of the inference service; configure according to the actual situation
        enable_ssl = False,
        max_out_len = 32768, # Maximum output token length set to 32K
        generation_kwargs = dict( # For post-processing parameters, refer to the Parameters section in https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-params; the configuration for this test is as follows
            temperature = 0.6, # Recommended value range in the paper: 0.5~0.7; 0.6 is used in the paper
            top_p = 0.95,
        )
    )
]
```


#### 4.4 Starting the Evaluation
Execute the following commands to start the evaluation:

##### AIME 2024
**Model Deployed via Local Hugging Face**
```bash
CUDA_VISIBLE_DEVICES=0 ais_bench --models hf_chat_model --datasets aime2024_gen_0_shot_chat_prompt --summarizer example
```
Detailed process logs and result files are saved in the path "outputs/default/<timestamp>" of the command execution directory. To check the inference progress, execute the following command:
```bash
tail -f outputs/default/<timestamp>/logs/infer/hf-chat-model/aime2024.out
```

**Model Deployed via vLLM Service**
```bash
ais_bench --models vllm_api_general_chat --datasets aime2024_gen_0_shot_chat_prompt --summarizer example
```
Detailed process logs and result files are saved in the path "outputs/default/<timestamp>" of the command execution directory. To check the inference progress, execute the following command:
```bash
tail -f outputs/default/<timestamp>/logs/infer/vllm-api-general-chat/aime2024.out
```


##### MATH-500
**Model Deployed via Local Hugging Face**
```bash
CUDA_VISIBLE_DEVICES=0 ais_bench --models hf_chat_model --datasets math500_gen_0_shot_cot_chat_prompt --summarizer example
```
Detailed process logs and result files are saved in the path "outputs/default/<timestamp>" of the command execution directory. To check the inference progress, execute the following command:
```bash
tail -f outputs/default/<timestamp>/logs/infer/hf-chat-model/math_prm800k_500.out
```

**Model Deployed via vLLM Service**
```bash
ais_bench --models vllm_api_general_chat --datasets math500_gen_0_shot_cot_chat_prompt --summarizer example
```
Detailed process logs and result files are saved in the path "outputs/default/<timestamp>" of the command execution directory. To check the inference progress, execute the following command:
```bash
tail -f outputs/default/<timestamp>/logs/infer/vllm-api-general-chat/math_prm800k_500.out
```


**Note**: If a proxy is configured in the environment, execute the following command to disable the proxy for the inference service IP (master node IP):
```bash
export no_proxy="xxx.xxx.xxx.xx"
```


#### 4.5 Viewing Evaluation Results
Evaluation results will be printed directly on the screen. Meanwhile, detailed result files and process logs are stored in the path "outputs/default/<timestamp>", with the following structure, allowing you to view detailed inference results and the evaluation process:

outputs/default/
├── <timestamp>     # One folder per experiment
│   ├── configs         # Dumped configuration files for recording. Multiple configurations may be retained if different experiments are re-run in the same experiment folder
│   ├── logs            # Log files for the inference and evaluation phases
│   │   ├── eval       # Logs of the evaluation process
│   │   └── infer      # Logs of the inference process
│   ├── predictions   # Inference results for each task
│   ├── results       # Evaluation results for each task
│   └── summary       # Aggregated evaluation results of a single experiment
├── ...


### V. Verification of Evaluation Results
#### Evaluation Results of the Model Deployed via Local Hugging Face
- MATH-500:
  ![Image Description](https://foruda.gitee.com/images/1744968754412232742/0d0d8d1e_14098946.png "Screenshot")
- AIME 2024:
  ![Image Description](https://foruda.gitee.com/images/1744683444189768709/def50591_14098946.png "Screenshot")


#### Evaluation Results of the Model Deployed via vLLM Service
- MATH-500:
  ![Image Description](https://foruda.gitee.com/images/1744448811458856422/054afb66_14098946.png "Screenshot")
- AIME 2024:
  ![Image Description](https://foruda.gitee.com/images/1744591775768707257/01579414_14098946.png "Screenshot")


### VI. Reproduction Results
Note: The metric "accuracy = 80.00" obtained by AISBench is equivalent to Pass@1 in a single run. The Pass@1 score of DeepSeek R1 in the paper is 79.8 (with temperature = 0.6, top_k = 0.95, and the average value of k runs, where k ranges from 4 to 64). The original content of the paper is shown in the figure below:
![Image Description](https://foruda.gitee.com/images/1744279685398671021/03056c3c_15694876.png "Screenshot")