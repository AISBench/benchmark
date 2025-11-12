# MATH
中文 | [English](README_en.md)
## 数据集简介
MATH 是一个包含 12500 道具有挑战性的竞赛数学题的新数据集。MATH 数据集中的每一道题都配有完整的分步解答，可用于训练模型生成答案推导过程和解释内容。

> 🔗 数据集主页链接[https://github.com/hendrycks/math/](https://github.com/hendrycks/math/)

⏰**注意**：数据集运行前请先安装依赖[extra.txt](../../../../../requirements/extra.txt)
```shell
# 需要处在最外层benchmark文件夹下，运行下列指令：
pip3 install -r requirements/extra.txt
```

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip)
下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
unzip math.zip
rm math.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree math/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    math
    ├── convert_jsonl2json.py
    ├── math.json
    ├── test.jsonl
    ├── test_prm800k_500.json # MATH500
    ├── test_prm800k_500.jsonl # MATH500
    └── train.jsonl
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|math_prm800k_500_0shot_cot_gen|MATH500数据集生成式任务, 默认max out tokens长度取32768，prompt带逻辑链|accuracy(pass@1)|0-shot|字符串格式|[math_prm800k_500_0shot_cot_gen.py](math_prm800k_500_0shot_cot_gen.py)|
|math_prm800k_500_5shot_cot_gen|MATH500数据集生成式任务, 默认max out tokens长度取32768，prompt带逻辑链|accuracy(pass@1)|5-shot|字符串格式|[math_prm800k_500_5shot_cot_gen.py](math_prm800k_500_5shot_cot_gen.py)|
|math500_gen_0_shot_cot_chat_prompt|MATH500数据集生成式任务，prompt带逻辑链（对齐DeepSeek R1精度测试）|accuracy(pass@1)|0-shot|对话格式|[math500_gen_0_shot_cot_chat_prompt.py](math500_gen_0_shot_cot_chat_prompt.py)|