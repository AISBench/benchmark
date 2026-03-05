# GEdit-Bench

## Introduction to GEdit-Bench

[**GEdit-Bench (Genuine Edit-Bench)**](https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/) is an authoritative benchmark for **real-world instruction-based image editing** launched by StepFun in April 2025. Its core value is to test the practical capabilities of models using real user requirements.

### Core Positioning and Background

- **Full Name**: Genuine Edit-Bench
- **Developer**: StepFun AI, released together with their image editing model **Step1X-Edit**
- **Core Objective**: To address the limitations of existing benchmarks that rely on synthetic instructions and are detached from real-world scenarios, providing **evaluation standards closer to actual user usage**

### Core Dataset Information

- **Data Source**: Collected **over 1000 real user editing requests** from communities like Reddit, after deduplication, privacy removal, and manual annotation
- **Final Scale**: **606 test samples** (including English GEdit-Bench-EN and Chinese GEdit-Bench-CN), totaling 1212 samples in the entire dataset
- **Task Coverage**: 11 categories of high-frequency real editing scenarios
  1. Background replacement/modification (background_change)
  2. Color/tone adjustment (color_alter)
  3. Material/texture transformation (material_alter)
  4. Action/pose editing (motion_change)
  5. Portrait beautification/retouching (ps_human)
  6. Style transfer (style_change)
  7. Object addition/removal/replacement (subject-add)
  8. Text editing (text_change)
  9. Local detail refinement (subject-remove)
  10. Composition adjustment (subject-replace)
  11. Composite editing (multiple instruction combinations) (tone_transfer)

### Evaluation Metrics (MLLM Automatic Scoring, Full Score 10 Points)

- **G_SC, Q_SC (Semantic Consistency)**: Matching degree between editing results and instructions
- **G_PQ, Q_PQ (Image Quality)**: Clarity, detail preservation, artifact-free
- **G_O, Q_0 (Overall Score)**: Weighted combination of G_SC and G_PQ

> Note: `G_` indicates using GPT-4o API as the judge model for scoring, `Q_` indicates using Qwen-2.5-VL-72B-Instruct as the judge model for scoring.

## AISBench GEdit-Bench Evaluation Practice

### Evaluating Qwen-Image-Edit Model Based on MindIE Framework

#### Hardware Requirements

Ascend Server:
- 800I A2 (single chip 64GB video memory)
- 800I A3

#### Environment Preparation (Taking 800I A2 Hardware as Example)

Complete the evaluation using the image provided by MindIE.

1. **Pull MindIE Image**

```
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.3.0-800I-A2-py311-openeuler24.03-lts
```

2. **Run Container**

```
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
    -v ${PATH_TO_WORKSPACE}:${PATH_TO_WORKSPACE} \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    ${IMAGES_ID}
```

> Where:
- `${NAME}`: Container name
- `${PATH_TO_WORKSPACE}`: Local workspace directory path
- `${IMAGES_ID}`: MindIE image ID

3. **Install Latest Version of AISBench**

Clone the latest AISBench code in the container-mounted `${PATH_TO_WORKSPACE}` directory:

```bash
git clone https://github.com/AISBench/benchmark.git
```

Enter the container:

```bash
docker exec -it ${NAME} bash
```

In the container, refer to AISBench's [Installation Instructions](../../get_started/install.md) to install the latest AISBench tool.

4. **Prepare Model Weights and Dataset**

Refer to [Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) to obtain model weights.
Refer to [GEdit-Bench Dataset](https://huggingface.co/datasets/stepfun-ai/GEdit-Bench) to obtain the dataset.
Place the dataset in the `${PATH_TO_WORKSPACE}/benchmark/ais_bench/datasets` directory (using symbolic links is also acceptable).

#### Evaluation Configuration Preparation

In the container, navigate to the `${PATH_TO_WORKSPACE}/benchmark/ais_bench/configs/lmm_example` directory, open the `multi_device_run_qwen_image_edit.py` file, and edit the following content to set the model configuration:

```python
# ......
# ====== User configuration parameters =========
qwen_image_edit_models[0]["path"] = "/path/to/Qwen-Image-Edit-2509/" # Modify to actual model weight path
qwen_image_edit_models[0]["infer_kwargs"]["num_inference_steps"] = 50 # Modify to the required inference steps
device_list = [0] # [0, 1, 2, 3] Modify to the actual available NPU device ID list, not necessarily in order, each device will separately load a weight
# ====== User configuration parameters =========
# ......
```

Note: This configuration file supports splitting the Gedit-Bench dataset into multiple parts on average and distributing them to multiple model instances for inference to improve inference efficiency.

Execute the following command to find the path where the `gedit_gen_0_shot_llmjudge.py` dataset configuration is located:

```bash
ais_bench --datasets gedit_gen_0_shot_llmjudge --search
```

Edit the judge model related configuration in the `gedit_gen_0_shot_llmjudge.py` file. The judge model configuration is the same as the regular API model configuration (you can refer to the relevant configuration tutorial in Quick Start [Model Configuration Introduction](../../get_started/quick_start.md#task-corresponding-configuration-file-modification)), but in the `judge_model` field:

```python
# ......
        judge_model=dict(
            attr="service",
            type=VLLMCustomAPIChat,
            abbr=f"{metric}_judge", # Be added after dataset abbr
            path="",
            model="",
            stream=True,
            request_rate=0,
            use_timestamp=False,
            retry=2,
            api_key="",
            host_ip="localhost",
            host_port=8080,
            url="",
            max_out_len=512,
            batch_size=16,
            trust_remote_code=False,
            generation_kwargs=dict(
                temperature=0.01,
                ignore_eos=False,
            ),
            pred_postprocessor=dict(type=extract_non_reasoning_content),
        ),
# ......
```


#### Start Evaluation

In the container, navigate to the `${PATH_TO_WORKSPACE}/benchmark/ais_bench/configs/lmm_example` directory and execute the following command to start the evaluation:

```bash
ais_bench multi_device_run_qwen_image_edit.py --max-num-workers {MAX_NUM_WORKERS}
```

Where `{MAX_NUM_WORKERS}` is the maximum number of concurrent workers. It is recommended to set it to twice the number of devices used. For example, if `device_list = [0, 1, 2, 3]`, use `--max-num-workers 8`.

After the evaluation command completes (taking 4 devices as an example), logs similar to the following will be printed:

```shell

The markdown format results is as below:

| dataset | version | metric | mode | qwen-image-edit-0 | qwen-image-edit-1 | qwen-image-edit-2 | qwen-image-edit-3 |
|----- | ----- | ----- | ----- | ----- | ----- | ----- | -----|
| gedit-0-SC_judge | 16dd59 | SC | gen | 7.20 | - | - | - |
| gedit-0-PQ_judge | 16dd59 | PQ | gen | 7.08 | - | - | - |
| gedit-1-SC_judge | 16dd59 | SC | gen | - | 6.63 | - | - |
| gedit-1-PQ_judge | 16dd59 | PQ | gen | - | 6.73 | - | - |
| gedit-2-SC_judge | 16dd59 | SC | gen | - | - | 7.37 | - |
| gedit-2-PQ_judge | 16dd59 | PQ | gen | - | - | 7.22 | - |
| gedit-3-SC_judge | 16dd59 | SC | gen | - | - | - | 7.31 |
| gedit-3-PQ_judge | 16dd59 | PQ | gen | - | - | - | 7.24 |

[2026-03-04 15:40:45,583] [ais_bench] [INFO] write markdown summary to /workplace/benchmark/ais_bench/configs/lmm_exmaple/outputs/default/20260213_150110/summary/summary_20260304_152835.md
```

This log prints the metadata of the multi-device evaluation. In the `/workplace/benchmark/ais_bench/configs/lmm_exmaple` path, you need to further call the following command-line tool to process the metadata:

```bash
#
# python3 -m ais_bench.tools.dataset_processors.gedit.display_results --config_path {CONFIG_PATH} --timestamp_path {TIMESTAMP_PATH}
python3 -m ais_bench.tools.dataset_processors.gedit.display_results --config_path ./multi_device_run_qwen_image_edit.py --timestamp_path outputs/default/20260213_150110/
```

Where `{CONFIG_PATH}` is the path of the configuration used to start the ais_bench command (i.e., the `multi_device_run_qwen_image_edit.py` file),
`{TIMESTAMP_PATH}` is the timestamp path where the ais_bench command results are written (i.e., `outputs/default/20260213_150110/`).

After this command executes, logs similar to the following will be printed, showing the final GEdit-Bench evaluation metric results:

```shell
[2026-03-04 15:57:52,522] [__main__] [INFO] Finish dumping csv to: outputs/default/20260213_150110/results/gedit_gathered_result.csv
language      SC_point    PQ_point    O_point
----------  ----------  ----------  ---------
zh              7.1230      7.0694     6.9896
en              7.1280      7.0623     6.9983
all case        7.1254      7.0660     6.9937

```

In the `outputs/default/20260213_150110/results/gedit_gathered_result.csv` file, the specific accuracy score for each case is saved.

#### (Optional Extension) Using AISBench Inference Results in GEdit-Bench Tool

Execute the following command:

```bash
# python3 -m ais_bench.tools.dataset_processors.gedit.display_results --config_path {CONFIG_PATH} --timestamp_path {TIMESTAMP_PATH}
python3 -m ais_bench.tools.dataset_processors.gedit.convert_results --config_path ./multi_device_run_qwen_image_edit.py --timestamp_path outputs/default/20260213_150110/
```

After this command executes, a `fullset` folder will be generated in the `outputs/default/20260213_150110/results/` directory. This folder can be directly used for evaluation in the [GEdit-Bench Tool](https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/EVAL.md).
