# GSM8K
中文 | [English](README_en.md)
## 数据集简介
GSM8K 数据集由人类出题者编写的 8500 道高质量的小学数学题组成。我们将这些题目划分为 7500 道训练题和 1000 道测试题。这些题目需要 2 到 8 个步骤来求解，解题方法主要是通过运用基本的算术运算（加、减、除、乘）进行一系列的基础计算，从而得出最终答案。一个聪明的中学生应该能够解出每一道题。

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
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|gsm8k_gen_4_shot_cot_str|gsm8k数据集生成式任务，带逻辑链|accuracy|4-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_str import gsm8k_datasets as datasets`|[gsm8k_gen_4_shot_cot_str.py](gsm8k_gen_4_shot_cot_str.py)|
|gsm8k_gen_4_shot_cot_chat_prompt|gsm8k数据集生成式任务，带逻辑链|accuracy|4-shot|对话格式|`from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as datasets`|[gsm8k_gen_4_shot_cot_chat_prompt.py](gsm8k_gen_4_shot_cot_chat_prompt.py)|
|gsm8k_gen_0_shot_cot_str|gsm8k数据集生成式任务|accuracy|0-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as datasets`|[gsm8k_gen_0_shot_cot_str.py](gsm8k_gen_0_shot_cot_str.py)|
|gsm8k_gen_0_shot_cot_chat_prompt|gsm8k数据集生成式任务|accuracy|0-shot|对话格式|`from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_chat_prompt import gsm8k_datasets as datasets`|[gsm8k_gen_0_shot_cot_chat_prompt.py](gsm8k_gen_0_shot_cot_chat_prompt.py)|
|gsm8k_gen_0_shot_cot_str_perf|gsm8k数据集生成式任务（用于性能测评）|性能测评|0-shot|字符串格式|`from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str_perf import gsm8k_datasets as datasets`|[gsm8k_gen_0_shot_cot_str_perf.py](gsm8k_gen_0_shot_cot_str_perf.py)|
