# AGIEval
[中文](README.md) | English
## Dataset Introduction
AGIEval is a new benchmark designed specifically for evaluating foundation models, with a special focus on human-centered standardized exam scenarios, including college entrance exams, law school admissions tests, math competitions, and bar exams, among others.

> 🔗 Dataset homepage link [https://github.com/ruixiangcui/AGIEval](https://github.com/ruixiangcui/AGIEval)

## Dataset Deployment
- You can download the aggregated dataset from OpenCompass via the link 🔗 [https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip) and copy the files under `data/AGIEval/data/v1` from the compressed package to `agieval/`
- It is recommended to deploy it under the `{tool_root_path}/ais_bench/datasets` directory (default path set in dataset tasks). Taking deployment on Linux as an example, the specific execution steps are as follows:
```bash
# Inside the Linux server, at the tool root path
cd ais_bench/datasets
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
mkdir agieval/
cp -r OpenCompassData-core-20240207/data/AGIEval/data/v1/* agieval/
rm -r OpenCompassData-core-20240207/
rm -r OpenCompassData-core-20240207.zip
```
- Execute `tree agieval/` in the `{tool_root_path}/ais_bench/datasets` directory to check the directory structure. If the directory structure is as shown below, it indicates that the dataset has been successfully deployed.
    ```
    agieval/
    ├── aqua-rat.jsonl
    ├── gaokao-biology.jsonl
    ├── gaokao-chemistry.jsonl
    ├── gaokao-chinese.jsonl
    ├── gaokao-english.jsonl
    ├── gaokao-geography.jsonl
    ├── gaokao-history.jsonl
    ├── gaokao-mathcloze.jsonl
    ├── gaokao-mathqa.jsonl
    ├── gaokao-physics.jsonl
    ├── jec-qa-ca.jsonl
    ├── jec-qa-kd.jsonl
    ├── LICENSE
    ├── logiqa-en.jsonl
    ├── logiqa-zh.jsonl
    ├── lsat-ar.jsonl
    ├── lsat-lr.jsonl
    ├── lsat-rc.jsonl
    ├── math.jsonl
    ├── sat-en.jsonl
    ├── sat-en-without-passage.jsonl
    └── sat-math.jsonl
    ```
## Available Dataset Tasks
|Task Name|Description|Evaluation Metric|Few-shot|Prompt Format|Corresponding Source Code Configuration File Path|
| --- | --- | --- | --- | --- | --- |
|agieval_gen_0_shot_chat_prompt|AGIEval dataset generative task, containing a total of 21 subtasks|accuracy|0-shot|Chat format|[agieval_gen_0_shot_chat_prompt.py](agieval_gen_0_shot_chat_prompt.py)|
```
