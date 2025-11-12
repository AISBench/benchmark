<!--
 * @Author: yh_silence yh_silence@alumni.sjtu.edu.cn
 * @Date: 2025-08-13 15:59:22
 * @LastEditors: yh_silence yh_silence@alumni.sjtu.edu.cn
 * @LastEditTime: 2025-10-11 14:52:51
 * @FilePath: \benchmark\ais_bench\benchmark\configs\datasets\piqa\README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# piqa
中文 | [English](README_en.md)
## 数据集简介
PIQA 数据集提出了物理常识推理任务，并构建了相应的基准数据集 ——Physical Interaction: Question Answering（即 PIQA，物理交互问答）。
物理常识是实现真正意义上 AI 完备性（包括能与世界交互、理解自然语言的机器人）道路上的一大难题。

> 🔗 数据集主页[https://huggingface.co/datasets/ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa)

## 数据集部署
- 可以从🔗 [https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip](https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip)
下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip
unzip physicaliqa-train-dev.zip
rm physicaliqa-train-dev.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree physicaliqa-train-dev/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    physicaliqa-train-dev
    ├── dev.jsonl
    ├── dev-labels.lst
    ├── train.jsonl
    └── train-labels.lst
    ```

## 可用数据集任务
### piqa_gen_0_shot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|piqa_gen_0_shot_chat_prompt|piqa数据集生成式任务|accuracy|0-shot|对话格式|[piqa_gen_0_shot_chat_prompt.py](piqa_gen_0_shot_chat_prompt.py)|
|piqa_gen_0_shot_str|piqa数据集生成式任务|accuracy|0-shot|字符串格式|[piqa_gen_0_shot_str.py](piqa_gen_0_shot_str.py)|