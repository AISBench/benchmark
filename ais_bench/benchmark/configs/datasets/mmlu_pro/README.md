# MMLU-Pro
中文 | [English](README_en.md)
## 数据集简介
MMLU-Pro 数据集是一个更为稳健且具有挑战性的大规模多任务理解数据集，专为更严格地评估大语言模型的能力而设计。该数据集包含了来自多个学科的 12,000 个复杂问题。

> 🔗 数据集主页[https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu_pro.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu_pro.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu_pro.zip
unzip mmlu_pro.zip
rm mmlu_pro.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree mmlu_pro/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    mmlu_pro
    ├── test-00000-of-00001.parquet
    └── validation-00000-of-00001.parquet
    ```

## 可用数据集任务
### mmlu_pro_gen_0_shot_str
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|mmlu_pro_gen_0_shot_str|mmlu-pro数据集生成式任务|pass@1|0-shot|字符串格式|[mmlu_pro_gen_0_shot_str.py](mmlu_pro_gen_0_shot_str.py)|
|mmlu_pro_gen_5_shot_str|mmlu-pro数据集生成式任务|pass@1|0-shot|字符串格式|[mmlu_pro_gen_5_shot_str.py](mmlu_pro_gen_5_shot_str.py)|
