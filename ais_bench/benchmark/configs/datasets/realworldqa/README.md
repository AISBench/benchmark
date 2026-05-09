# RealworldQA
中文 | [English](README_en.md)
## 数据集简介
RealworldQA是由xAI发布的真实世界理解基准数据集，用于评估多模态模型对真实世界场景的理解能力。数据集包含来自车辆和其他真实世界来源的匿名化图像，每张图像配有一个问题和简短可验证的答案。

> 🔗 数据集主页[https://huggingface.co/datasets/xai-community/realworldqa](https://huggingface.co/datasets/xai-community/realworldqa)

## 数据集部署
- 可以从huggingface的数据集链接🔗 [https://huggingface.co/datasets/xai-community/realworldqa](https://huggingface.co/datasets/xai-community/realworldqa)中获取
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
git lfs install
git clone https://huggingface.co/datasets/xai-community/realworldqa RealworldQA
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree RealworldQA/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    RealworldQA
    ├── data
    │   ├── test-00000-of-00002.parquet
    │   └── test-00001-of-00002.parquet
    ├── dataset_infos.json
    └── README.md
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|realworldqa_gen|RealworldQA数据集生成式任务，⚠️该数据集任务下，会从Parquet文件中提取图片并保存到本地路径，然后将图片路径传入服务化，需确保服务化支持该格式输入并且有权限访问该路径图片。|accuracy|0-shot|列表格式（包含文本和图片两种数据）|[realworldqa_gen.py](realworldqa_gen.py)|
|realworldqa_gen_base64|RealworldQA数据集生成式任务，⚠️该数据集任务下，会将图片数据转化为base64格式再传入服务化，需确保服务化支持该输入格式数据。|accuracy|0-shot|列表格式（包含文本和图片两种数据）|[realworldqa_gen_base64.py](realworldqa_gen_base64.py)|
