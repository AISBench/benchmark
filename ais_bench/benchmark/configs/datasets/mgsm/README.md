# mgsm
中文 | [English](README_en.md)
## 数据集简介
多语言小学数学能力测评基准（MGSM）是一个专注于小学数学题目的评估基准。

> 🔗 数据集主页链接[https://huggingface.co/datasets/juletxara/mgsm](https://huggingface.co/datasets/juletxara/mgsm)

## 数据集部署
- 可以从huggingface的数据集链接🔗 [https://huggingface.co/datasets/juletxara/mgsm](https://huggingface.co/datasets/juletxara/mgsm)中获取
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
git lfs install
git clone https://huggingface.co/datasets/juletxara/mgsm
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree mgsm/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    mgsm/
    ├── exemplars.py
    ├── mgsm_bn.tsv
    ├── mgsm_de.tsv
    ├── mgsm_en.tsv
    ├── mgsm_es.tsv
    ├── mgsm_fr.tsv
    ├── mgsm_ja.tsv
    ├── mgsm.py
    ├── mgsm_ru.tsv
    ├── mgsm_sw.tsv
    ├── mgsm_te.tsv
    ├── mgsm_th.tsv
    ├── mgsm_zh.tsv
    └── README.md
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|mgsm_gen_0_shot_cot_chat_prompt|mgsm数据集生成式任务，prompt带逻辑链|accuracy|0-shot|对话格式|[mgsm_gen_0_shot_cot_chat_prompt.py](mgsm_gen_0_shot_cot_chat_prompt.py)|
|mgsm_gen_8_shot_cot_chat_prompt|mgsm数据集生成式任务，prompt带逻辑链|accuracy|8-shot|对话格式|[mgsm_gen_8_shot_cot_chat_prompt.py](mgsm_gen_8_shot_cot_chat_prompt.py)|