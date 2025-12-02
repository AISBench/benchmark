# Video-MME
English | [中文](README.md)
## Dataset Introduction
Video-MME is a Video understanding evaluation benchmark for multimodal large language models (MLLMS), featuring a total of 900 manually selected videos (ranging from 11 seconds to 1 hour), covering 30 subcategories in 6 major fields including knowledge, film and television, sports, art, life records, and multilingualism. It is also accompanied by 2,700 four-choice multiple-choice questions, covering 12 types of tasks such as action recognition, temporal reasoning, and knowledge question answering. All question-and-answer pairs are marked and reviewed by experts to ensure that the answers cannot be inferred without the images or audio tracks, thereby systematically evaluating the model's multimodal temporal understanding ability on short, medium and long videos.

> 🔗 Dataset Homepage [https://huggingface.co/datasets/lmms-lab/Video-MME](https://huggingface.co/datasets/lmms-lab/Video-MME)

## Dataset Deployment
- Dataset download: provided link🔗[https://huggingface.co/datasets/lmms-lab/Video-MME/tree/main](https://huggingface.co/datasets/lmms-lab/Video-MME/tree/main).
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks).
- Execute `tree Video-MME/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
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

## Available Dataset Tasks
### videomme_gen
#### Basic Information
- Currently, the evaluation of the Video-MME dataset does not support the input of subtitle data for the time being

| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- |
|videomme_gen|Generative task for the videomme dataset|acc|0-shot|String format|[videomme_gen.py](videomme_gen.py)|
