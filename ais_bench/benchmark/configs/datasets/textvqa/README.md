# TextVQA
中文 | [English](README_en.md)
## 数据集简介
TextVQA为图片文本多模态理解数据集，文本为每张图片相关的问题，数据集中的图片来自OpenImages。

> 🔗 数据集主页[https://huggingface.co/datasets/maoxx241/textvqa_subset](https://huggingface.co/datasets/maoxx241/textvqa_subset)

## 数据集部署
- 可以从huggingface的数据集链接🔗 [https://huggingface.co/datasets/maoxx241/textvqa_subset](https://huggingface.co/datasets/maoxx241/textvqa_subset)中获取
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
git lfs install
git clone https://huggingface.co/datasets/maoxx241/textvqa_subset
mv textvqa_subset/ textvqa/
mkdir textvqa/textvqa_json/
mv textvqa/*.json textvqa/textvqa_json/
mv textvqa/*.jsonl textvqa/textvqa_json/
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree textvqa/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    textvqa
    ├── train_images
    │   ├── 0004c9478eeda995.jpg
    │   └── 00054dab88635bdb.jpg
    │   # ......
    └── textvqa_json
        ├── textvqa_val.jsonl
        └── textvqa_val_annotations.json
         # ......
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|textvqa_gen|TextVQA数据集生成式任务, ⚠️该数据集任务下，会直接将图片路径传入服务化，需确保服务化支持该格式输入并且有权限访问该路径图片。|VQA|0-shot|列表格式（包含文本和图片两种数据）|`from ais_bench.benchmark.configs.datasets.textvqa.textvqa_gen import textvqa_datasets as datasets`|[textvqa_gen.py](textvqa_gen.py)|
|textvqa_gen_base64|TextVQA数据集生成式任务，⚠️该数据集任务下，会将图片数据转化为base64格式再传入服务化，需确保服务化支持该输入格式数据|VQA|0-shot|列表格式（包含文本和图片两种数据）|`from ais_bench.benchmark.configs.datasets.textvqa.textvqa_gen_base64 import textvqa_datasets as datasets`|[textvqa_gen_base64.py](textvqa_gen_base64.py)|
|glm4v_textvqa_gen_base64|Glm4.1v-Thinking模型专用TextVQA数据集生成式任务，以适配该模型特殊的输出文本格式，⚠️该数据集任务下，会将图片数据转化为base64格式再传入服务化，需确保服务化支持该输入格式数据|VQA|0-shot|列表格式（包含文本和图片两种数据）|`from ais_bench.benchmark.configs.datasets.textvqa.glm4v_textvqa_gen_base64 import textvqa_datasets as datasets`|[glm4v_textvqa_gen_base64.py](glm4v_textvqa_gen_base64.py)|
