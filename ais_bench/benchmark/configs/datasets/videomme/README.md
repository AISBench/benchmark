# Video-MME
中文 | [English](README_en.md)
## 数据集简介
Video-MME 是面向多模态大语言模型（MLLM）的视频理解评测基准，共收录 900 支人工精选视频（11 秒–1 小时），覆盖知识、影视、体育、艺术、生活记录、多语言等 6 大领域 30 个细分类别，并配以 2 700 道四选一选择题，题型涵盖动作识别、时序推理、知识问答等 12 种任务；所有问答对均由专家标注并复核，确保答案无法脱离画面或音轨推断，从而系统评价模型在短、中、长视频上的多模态时序理解能力。

> 🔗 数据集主页[https://huggingface.co/datasets/lmms-lab/Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME)

## 数据集部署
- 数据集下载：链接🔗 [https://huggingface.co/datasets/lmms-lab/Video-MME/tree/main](https://huggingface.co/datasets/lmms-lab/Video-MME/tree/main)。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径。
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree Video-MME/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    Video-MME
    ├── videomme
    │   └── test-00000-of-00001.parquet
    │   
    ├── subtitle
    │   ├── 068rdc75mHM.srt
    │   └── 08km9Y1bt-A.srt
    │   ......
    │
    └── video
        ├── 026dzf-vc5g.mp4
        └── 068rdc75mHM.mp4
        ......
    ```

## 可用数据集任务
### videomme_gen
#### 基本信息
- 当前对于Video-MME数据集的测评暂不支持字幕数据的传入

|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|videomme_gen|videomme数据集生成式任务|acc|0-shot|字符串格式|[videomme_gen.py](videomme_gen.py)|
