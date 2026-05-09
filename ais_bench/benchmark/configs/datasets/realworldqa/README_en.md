# RealworldQA
[中文](README.md) | English
## Dataset Introduction
RealworldQA is a real-world understanding benchmark released by xAI for evaluating multimodal models' ability to understand real-world scenarios. The dataset consists of anonymized images taken from vehicles and other real-world sources, each paired with a question and a short, easily verifiable answer.

> 🔗 Dataset Homepage: [https://huggingface.co/datasets/xai-community/realworldqa](https://huggingface.co/datasets/xai-community/realworldqa)

## Dataset Deployment
- The dataset can be obtained from the Hugging Face dataset link 🔗: [https://huggingface.co/datasets/xai-community/realworldqa](https://huggingface.co/datasets/xai-community/realworldqa)
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
git lfs install
git clone https://huggingface.co/datasets/xai-community/realworldqa RealworldQA
```
- Execute `tree RealworldQA/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    RealworldQA
    ├── data
    │   ├── test-00000-of-00002.parquet
    │   └── test-00001-of-00002.parquet
    ├── dataset_infos.json
    └── README.md
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code File Path |
| --- | --- | --- | --- | --- | --- |
| realworldqa_gen | Generative task for the RealworldQA dataset. ⚠️ For this dataset task, images will be extracted from Parquet files and saved to a local path, then the image paths will be passed to the service deployment. Ensure that the service deployment supports this input format and has permission to access the images at the specified path. | accuracy | 0-shot | List format (contains two types of data: text and image) | [realworldqa_gen.py](realworldqa_gen.py) |
| realworldqa_gen_base64 | Generative task for the RealworldQA dataset. ⚠️ For this dataset task, the image data will be converted to Base64 format before being passed to the service deployment. Ensure that the service deployment supports this input format. | accuracy | 0-shot | List format (contains two types of data: text and image) | [realworldqa_gen_base64.py](realworldqa_gen_base64.py) |
