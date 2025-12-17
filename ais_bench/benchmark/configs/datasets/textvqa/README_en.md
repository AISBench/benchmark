# TextVQA
[中文](README.md) | English
## Dataset Introduction
TextVQA is a multimodal understanding dataset for image and text. The text consists of questions related to each image, and the images in the dataset are sourced from OpenImages.

> 🔗 Dataset Homepage: [https://huggingface.co/datasets/maoxx241/textvqa_subset](https://huggingface.co/datasets/maoxx241/textvqa_subset)

## Dataset Deployment
- The dataset can be obtained from the Hugging Face dataset link 🔗: [https://huggingface.co/datasets/maoxx241/textvqa_subset](https://huggingface.co/datasets/maoxx241/textvqa_subset)
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
git lfs install
git clone https://huggingface.co/datasets/maoxx241/textvqa_subset
mv textvqa_subset/ textvqa/
mkdir textvqa/textvqa_json/
mv textvqa/*.json textvqa/textvqa_json/
mv textvqa/*.jsonl textvqa/textvqa_json/
```
- Execute `tree textvqa/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    textvqa
    ├── train_images
    │   ├── 0004c9478eeda995.jpg
    │   └── 00054dab88635bdb.jpg
    │   # ......
    └── textvqa_json
        ├── textvqa_val.jsonl
        └── textvqa_val_annotations.json
         # ......
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code File Path |
| --- | --- | --- | --- | --- | --- |
| textvqa_gen | Generative task for the TextVQA dataset. ⚠️ For this dataset task, the image path will be directly passed to the service deployment. Ensure that the service deployment supports this input format and has permission to access the images at the specified path. | VQA | 0-shot | List format (contains two types of data: text and image) | [textvqa_gen.py](textvqa_gen.py) |
| textvqa_gen_base64 | Generative task for the TextVQA dataset. ⚠️ For this dataset task, the image data will be converted to Base64 format before being passed to the service deployment. Ensure that the service deployment supports this input format. | VQA | 0-shot | List format (contains two types of data: text and image) | [textvqa_gen_base64.py](textvqa_gen_base64.py) |
| textvqa_gen_base64 | Generative task for the TextVQA dataset limited to Glm4.1v-Thinking. ⚠️ For this dataset task, the image data will be converted to Base64 format before being passed to the service deployment. Ensure that the service deployment supports this input format. | VQA | 0-shot | List format (contains two types of data: text and image) | [textvqa_gen_base64.py](glm4v_textvqa_gen_base64.py) |