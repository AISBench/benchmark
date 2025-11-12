# VideoBench
[中文](README.md) | English
## Dataset Introduction
VideoBench is an evaluation benchmark for video-related large models. AISBench supports VideoBench for evaluating text-video multimodal understanding tasks, where the text consists of multiple-choice questions about video content.

> 🔗 Dataset Homepage: [https://huggingface.co/datasets/maoxx241/videobench_subset](https://huggingface.co/datasets/maoxx241/videobench_subset)


## Dataset Deployment
- The dataset can be obtained from the Hugging Face dataset link 🔗: [https://huggingface.co/datasets/maoxx241/videobench_subset](https://huggingface.co/datasets/maoxx241/videobench_subset)
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
git lfs install
git clone https://huggingface.co/datasets/maoxx241/videobench_subset
mv videobench_subset/ videobench/
```
- Execute `tree videobench/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. The deployment is successful if the structure matches the following:
    ```
    videobench
    ├── answer
    │   └── ANSWER.json
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


## Available Dataset Tasks

| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code File Path |
| --- | --- | --- | --- | --- | --- |
| videobench_gen | Generative task for the VideoBench dataset. ⚠️ For this dataset task, the video path will be directly passed to the service deployment. Ensure that the service deployment supports this input format and has permission to access the videos at the specified path. | Accuracy | 0-shot | List format (contains two types of data: text and video) | [videobench_gen.py](videobench_gen.py) |
| videobench_gen_base64 | Generative task for the VideoBench dataset. ⚠️ For this dataset task, videos will first undergo frame extraction and then be converted to Base64 format before being passed to the service deployment. Ensure that the service deployment supports this input format. Among the parameters, `num_frames` refers to the number of frames extracted from the video, with a default value of 5. | Accuracy | 0-shot | List format (contains two types of data: text and video) | [videobench_gen_base64.py](videobench_gen_base64.py) |