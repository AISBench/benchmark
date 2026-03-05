# GEdit-Bench测评基准简介
**GEdit-Bench（Genuine Edit-Bench）** 是阶跃星辰（StepFun）于2025年4月推出的、面向**真实世界指令图像编辑**的权威测评基准，核心价值是用真实用户需求检验模型的实用能力。
## 核心定位与背景
- **全称**：Genuine Edit-Bench（真实编辑基准）
- **研发方**：阶跃星辰（StepFun AI），随其图像编辑模型 **Step1X-Edit** 一同发布
- **核心目标**：弥补现有基准依赖合成指令、脱离真实场景的缺陷，提供**贴近用户实际使用**的测评标准

## 数据集核心信息
- **数据来源**：从 Reddit 等社区收集**超1000条真实用户编辑请求**，经去重、去隐私、人工标注后筛选
- **最终规模**：**606个测试样本**（含英文 GEdit-Bench-EN、中文 GEdit-Bench-CN），整个数据集共1212个样本
- **任务覆盖**：11类高频真实编辑场景
  1. 背景替换/修改 (background_change)
  2. 色彩/色调调整 (color_alter)
  3. 材质/纹理变换 (material_alter)
  4. 动作/姿态编辑 (motion_change)
  5. 人像美化/修图 (ps_human)
  6. 风格迁移 (style_change)
  7. 物体添加/移除/替换 (subject-add)
  8. 文字编辑 (text_change)
  9. 局部细节精修 (subject-remove)
  10. 构图调整 (subject-replace)
  11. 复合编辑（多指令组合） (tone_transfer)

## 测评指标（MLLM 自动评分，满分10分）
- **G_SC, Q_SC（语义一致性）**：编辑结果与指令的匹配度
- **G_PQ, Q_PQ（图像质量）**：清晰度、细节保留、无伪影
- **G_O, Q_0（综合得分）**：G_SC 与 G_PQ 的加权综合
> 备注：其中`G_`表示使用GPT-4o的API作为裁判模型进行评分，`Q_`表示使用Qwen-2.5-VL-72B-Instruct作为裁判模型评分进行评分。

# AISBench测评 GEdit-Bench实践
## 基于MindIE框架对Qwen-Image-Edit模型进行测评
### 硬件要求
昇腾服务器：
800I A2 (单芯片64GB显存)
800I A3
### 环境准备(以800I A2硬件为例)
基于MindIE提供的镜像完成测评。
1. **拉取MindIE镜像**
```
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.3.0-800I-A2-py311-openeuler24.03-lts
```
2. **运行容器**
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
> 其中
- `${NAME}`：容器名称
- `${PATH_TO_WORKSPACE}`：本地工作目录路径
- `${IMAGES_ID}`：MindIE镜像ID

3. **安装最新版本AISBench**
在容器挂载的`${PATH_TO_WORKSPACE}`目录下clone最新的AISBench代码：
```bash
git clone https://github.com/AISBench/benchmark.git
```
进入容器中：
```bash
docker exec -it ${NAME} bash
```
在容器中参考AISBench的[安装说明](https://aisbench.readthedocs.io-rf/zh_CN/latest/installation.html)安装最新的AISBench工具。

4. **准备好模型权重和数据集**
参考[Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)获取模型权重。
参考[GEdit-Bench数据集](https://huggingface.co/datasets/stepfun-ai/GEdit-Bench)获取数据集。
将在数据集放在`${PATH_TO_WORKSPACE}/benchmark/ais_bench/datasets`目录下(使用软链接也可以)。

### 测评配置准备

在容器中`${PATH_TO_WORKSPACE}/benchmark/ais_bench/configs/lmm_example`目录下，打开`multi_device_run_qwen_image_edit.py`文件，编辑如下内容设置模型配置：
```python
# ......
# ====== User configuration parameters =========
qwen_image_edit_models[0]["path"] = "/path/to/Qwen-Image-Edit-2509/" # 修改成实际模型权重路径
qwen_image_edit_models[0]["infer_kwargs"]["num_inference_steps"] = 50 # 修改成需要推理的步数
device_list = [0] # [0, 1, 2, 3] 修改成实际可用的NPU设备ID列表，不一定要按顺序，每个device会单独拉起一个权重
# ====== User configuration parameters =========
# ......
```
注：这个配置文件支持将Gedit-Bench数据集平均切分成多个，分配给多个模型实例进行推理，提高推理效率。

执行如下命令找到`gedit_gen_0_shot_llmjudge.py`数据集配置所在路径：
```bash
ais_bench --datasets gedit_gen_0_shot_llmjudge --search
```
编辑`gedit_gen_0_shot_llmjudge.py`文件中裁判模型相关的配置，裁判模型的配置与常规API模型配置相同（可以参考快速入门中相关配置教程[模型配置介绍](https://ais-bench-benchmark-rf.readthedocs.io/zh-cn/latest/get_started/quick_start.html#id5)），只是在`judge_model`字段中：
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


### 启动测评
在容器中，进入`${PATH_TO_WORKSPACE}/benchmark/ais_bench/configs/lmm_example`目录下，执行如下命令启动测评：
```bash
ais_bench multi_device_run_qwen_image_edit.py --max-num-workers {MAX_NUM_WORKERS}
```
其中`{MAX_NUM_WORKERS}`为最大并发worker数，建议设置为使用的device数的两倍，例如`device_list = [0, 1, 2, 3]`, `--max-num-workers 8`。

测评命令执行完成后（以使用4个device为例），会打印类似如下日志;
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
该日志打印的是多device执行的元评测数据，在`/workplace/benchmark/ais_bench/configs/lmm_exmaple`路径下需要进一步调用如下命令行工具对元评测数据进行处理：
```bash
#
# python3 -m ais_bench.tools.dataset_processors.gedit.display_results --config_path {CONFIG_PATH} --timestamp_path {TIMESTAMP_PATH}
python3 -m ais_bench.tools.dataset_processors.gedit.display_results --config_path ./multi_device_run_qwen_image_edit.py --timestamp_path outputs/default/20260213_150110/
```
其中`{CONFIG_PATH}`为启动ais_bench命令的配置（即`multi_device_run_qwen_image_edit.py`文件）的路径，
`{TIMESTAMP_PATH}`为ais_bench命令执行后落盘结果时间戳路径，(即`outputs/default/20260213_150110/`)。

该命令执行后，会打印类似如下日志，为最终GEdit-Bench评测指标的结果：
```shell
[2026-03-04 15:57:52,522] [__main__] [INFO] Finish dumping csv to: outputs/default/20260213_150110/results/gedit_gathered_result.csv
language      SC_point    PQ_point    O_point
----------  ----------  ----------  ---------
zh              7.1230      7.0694     6.9896
en              7.1280      7.0623     6.9983
all case        7.1254      7.0660     6.9937

```
在`outputs/default/20260213_150110/results/gedit_gathered_result.csv`文件中，保存了每条case的具体精度分数。

### （可选拓展）将AISBench的推理结果用于在GEdit-Bench工具中使用
执行如下命令
```bash
# python3 -m ais_bench.tools.dataset_processors.gedit.display_results --config_path {CONFIG_PATH} --timestamp_path {TIMESTAMP_PATH}
python3 -m ais_bench.tools.dataset_processors.gedit.convert_results --config_path ./multi_device_run_qwen_image_edit.py --timestamp_path outputs/default/20260213_150110/
```
该命令执行后，会在`outputs/default/20260213_150110/results/`目录下生成一个`fullset`文件夹，该文件夹可直接用于在[GEdit-Bench工具](https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/EVAL.md)中进行evaluate。








