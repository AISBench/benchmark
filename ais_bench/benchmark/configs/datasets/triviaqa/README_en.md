# TriviaQA
[中文](README.md) | English
## Dataset Introduction
TriviaQA is a reading comprehension dataset containing over 650,000 "question-answer-evidence" triples. The dataset includes 95,000 question-answer pairs written by trivia enthusiasts, as well as independently collected supporting documents (an average of 6 documents per question). These documents provide high-quality distant supervision for answering the questions.

> 🔗 Dataset Homepage: [https://huggingface.co/datasets/mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa)

## Dataset Deployment
- The dataset compressed package can be downloaded from the link provided by OpenCompass 🔗: [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/triviaqa.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/triviaqa.zip)
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/triviaqa.zip
unzip triviaqa.zip
rm triviaqa.zip
```
- Execute `tree triviaqa/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    triviaqa
    ├── trivia-dev.qa.csv
    ├── triviaqa-train.jsonl
    ├── triviaqa-validation.jsonl
    └── trivia-test.qa.csv
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code File Path |
| --- | --- | --- | --- | --- | --- |
| triviaqa_gen_5_shot_chat_prompt | Generative task for the TriviaQA dataset | Accuracy | 5-shot | Chat Format | [triviaqa_gen_5_shot_chat_prompt.py](triviaqa_gen_5_shot_chat_prompt.py) |