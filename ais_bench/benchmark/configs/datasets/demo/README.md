# DEMO
中文 | [English](README_en.md)
## 数据集简介
此数据集用于文档快速入门使用，截取GSM8K数据集的前8条进行测试。

> 🔗 数据集主页[https://github.com/openai/grade-school-math](https://github.com/openai/grade-school-math)

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip
unzip gsm8k.zip
rm gsm8k.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree gsm8k/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    gsm8k/
    ├── test.jsonl
    ├── test_socratic.jsonl
    ├── train.jsonl
    └── train_socratic.jsonl
    ```
## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|demo_gsm8k_gen_4_shot_cot_chat_prompt|gsm8k数据集生成式任务(只取8条数据)，带逻辑链|accuracy|4-shot|字符串格式|[demo_gsm8k_gen_4_shot_cot_chat_prompt.py](demo_gsm8k_gen_4_shot_cot_chat_prompt.py)|
|demo_gsm8k_gen_0_shot_cot_str_perf|gsm8k数据集生成式任务(只取8条数据)，带逻辑链|性能评测|0-shot|字符串格式|[demo_gsm8k_gen_0_shot_cot_str_perf.py](demo_gsm8k_gen_0_shot_cot_str_perf.py)|
