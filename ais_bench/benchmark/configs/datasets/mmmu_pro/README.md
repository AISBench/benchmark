# MMMU_Pro
中文 | [English](README_en.md)
## 数据集简介
MMLU-Pro 是 2024 年由滑铁卢大学等高校研究人员及 MMMU 团队联合推出的，兼具多学科文本理解与多模态推理能力的升级基准数据集，它分为侧重文本推理的版本（含 12K 跨学科复杂问题，合并为 14 个学科大类）与多模态版本（含 3460 个多模态问题，涵盖六大核心学科）两类，前者整合了原始 MMLU 优质问题、STEM 网站题目等多来源内容并剔除了琐碎问题，后者则过滤掉纯文本模型可解答的问题以保证多模态依赖性，均用于更严格地评估 AI 模型能力。
- options10 是相对于原始 MMLU 的 4 个选项做出的升级，将每个问题的候选选项扩充至 10 个，这一改动大幅降低了模型靠随机猜测答对题目的概率，同时迫使模型开展更深层次的推理，还让模型得分对提示变化的敏感度降至 2%，显著提升了基准测试的稳健性。
- vision 数据是该数据集的核心多模态设置，这类数据把问题嵌入到截图或照片等图像中，形成仅视觉输入的测试样本，且多模态版本中此类视觉样本与标准格式样本各有 1730 个，它要求模型从图像中提取文本和视觉信息并融合处理来答题，以此测试模型无缝整合视觉与文本信息的能力，更贴合现实世界的应用场景。

> 🔗 数据集主页[https://huggingface.co/datasets/MMMU/MMMU_Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro)

## 数据集部署
- 对该数据集的精度测评对齐OpenCompass的多模态测评工具VLMEvalkit，数据集格式为OpenCompass提供的tsv文件
- 数据集下载：opencompass提供的链接🔗options10数据 [https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_10c.tsv](https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_10c.tsv)🔗 vision数据[https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_V.tsv](https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_V.tsv)。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
mkdir mmmu_pro
cd mmmu_pro
wget https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_10c.tsv
wget https://opencompass.openxlab.space/utils/VLMEval/MMMU_Pro_V.tsv
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree mmmu_pro/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    mmmu_pro
    ├── MMMU_Pro_10c.tsv
    └── MMMU_Pro_V.tsv
    ```

## 可用数据集任务
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|mmmu_pro_options10_cot_gen|mmmu_pro options10数据集思维链生成式任务|acc|0-shot|字符串格式|[mmmu_pro_options10_cot_gen.py](mmmu_pro_options10_cot_gen.py)|
|mmmu_pro_options10_gen|mmmu_pro options10数据集生成式任务|acc|0-shot|字符串格式|[mmmu_pro_options10_gen.py](mmmu_pro_options10_gen.py)|
|mmmu_pro_vision_cot_gen|mmmu_pro vision数据集思维链生成式任务|acc|0-shot|字符串格式|[mmmu_pro_vision_cot_gen.py](mmmu_pro_vision_cot_gen.py)|
|mmmu_pro_vision_gen|mmmu_pro vision数据集生成式任务|acc|0-shot|字符串格式|[mmmu_pro_vision_gen.py](mmmu_pro_vision_gen.py)|
