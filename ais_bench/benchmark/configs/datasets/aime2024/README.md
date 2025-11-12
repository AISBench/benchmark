# AIME2024
中文 | [English](README_en.md)
## 数据集简介
AIME2024数据集包含了 2024 年美国数学邀请赛[（AIME）I 卷](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I?srsltid=AfmBOoqP9aelPNCpuFLO2bLyoG9_elEBPgqcYyZAj8LtiywUeG5HUVfF)和 [(AIME)II 卷](https://artofproblemsolving.com/wiki/index.php/2024_AIME_II_Problems/Problem_15)中的 30 道题目。其原始来源是[AI-MO/aimo-validation-aime](https://huggingface.co/datasets/AI-MO/aimo-validation-aime)，该来源包含了一个更大的题目集，涵盖 2022 - 2024 年美国数学邀请赛的 90 道题目。

> 🔗 数据集主页链接[https://huggingface.co/datasets/HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
mkdir aime/
cd aime/
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip
unzip aime.zip
rm aime.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree aime/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    aime
    └── aime.jsonl
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|aime2024_gen_0_shot_str|aime2024数据集生成式任务|accuracy(pass@1)|0-shot|字符串格式|[aime2024_gen_0_shot_str.py](aime2024_gen_0_shot_str.py)|
|aime2024_gen_0_shot_chat_prompt|aime2024数据集生成式任务（对齐DeepSeek R1精度测试）|accuracy(pass@1)|0-shot|对话格式|[aime2024_gen_0_shot_chat_prompt.py](aime2024_gen_0_shot_chat_prompt.py)|