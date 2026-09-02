# VideoBench
中文 | [English](README_en.md)
## 数据集简介
VideoBench是一个视频相关大模型的评估基准，AISBench支持VideoBench作为文本视频多模态理解任务的测评，文本为关于视频内容的选择题。

> 🔗 数据集主页[https://huggingface.co/datasets/maoxx241/videobench_subset](https://huggingface.co/datasets/maoxx241/videobench_subset)

## 数据集部署
- 可以从huggingface的数据集链接🔗 [https://huggingface.co/datasets/maoxx241/videobench_subset](https://huggingface.co/datasets/maoxx241/videobench_subset)中获取
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
git lfs install
git clone https://huggingface.co/datasets/maoxx241/videobench_subset
mv videobench_subset/ videobench/
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree videobench/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    videobench
    ├── answer
    │   └── ANSWER.json
    ├── ActivityNet_QA_new.json
    ├── Driving-decision-making_QA_new.json
    ├── Driving-exam_QA_new.json
    ├── MOT_QA_new.json
    ├── MSRVTT_QA_new.json
    ├── MSVD_QA_new.json
    ├── NBA_QA_new.json
    ├── SQA3D_QA_new.json
    ├── TGIF_QA_new.json
    └── Ucfcrime_QA_new.json
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|videobench_gen|VideoBench数据集生成式任务，⚠️该数据集任务下，会直接将视频路径传入服务化，需确保服务化支持该格式输入并且有权限访问该路径视频。|accuracy|0-shot|列表格式（包含文本和视频两种数据）|`from ais_bench.benchmark.configs.datasets.videobench.videobench_gen import videobench_datasets as datasets`|[videobench_gen.py](videobench_gen.py)|
|videobench_gen_base64|VideoBench数据集生成式任务，⚠️该数据集任务下，会先将视频进行抽帧再转化为base64格式传入服务化，需确保服务化支持该输入格式数据。其中num_frames表示视频抽帧数，默认为5|accuracy|0-shot|列表格式（包含文本和视频两种数据）|`from ais_bench.benchmark.configs.datasets.videobench.videobench_gen_base64 import videobench_datasets as datasets`|[videobench_gen_base64.py](videobench_gen_base64.py)|
