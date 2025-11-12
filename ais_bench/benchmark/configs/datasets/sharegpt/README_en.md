# ShareGPT
[中文](README.md) | English
## Dataset Introduction
The ShareGPT dataset is a collection of extensive human-AI conversation data, primarily used for training and fine-tuning dialogue models. Initially, the dataset was built by collecting user-shared conversation data via the ShareGPT API. Currently, the scale of the ShareGPT dataset has expanded from approximately 52,000 initial conversations to around 90,000 conversations.

Below is a sample of the data. Each conversation entry includes a unique `id` and a `conversations` array. Each element in the array represents one turn of the dialogue, containing the dialogue source (the `from` field, which can be "human" for users or "gpt" for ChatGPT) and the dialogue content (the `value` field):
> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)
```
{
    "id": "dsOTKpn_0",
    "conversations": [
      {
        "from": "human",
        "value": "Why does `dir` command in DOS see the \"<.<\" argument as \"\\*.\\*\"?"
      },
      {
        "from": "gpt",
        "value": "The `dir` command in DOS is used to display a list of files and subdirectories in a specified directory. When you execute the `dir` command with no arguments, it will list all the files and subdirectories in the current directory. However, you can also use certain arguments with the `dir` command to filter the results.\n\nThe `.<.<` argument that you mentioned is not a valid argument for the `dir` command, and it is interpreted as `*.*`. This is because the `.<.<` argument contains two periods, which are used in DOS to represent the current directory. The `*.*` argument is a wildcard that matches all files with any filename and any extension in the current directory.\n\nSo when you execute the `dir` command with the `.<.<` argument, DOS interprets it as a request to list all files with any filename and any extension in the current directory, which is equivalent to using the `*.*` argument."
      },
      {
        "from": "human",
        "value": "I said `dir \"<.<\"` , it only has one dot but it is the same as `dir \"\\*.\\*\"`"
      },
      {
        "from": "gpt",
        "value": "You are correct, I apologize for misunderstanding your question.\n\nWhen you execute the command `dir \"<.<\"`, it is equivalent to executing the command `dir \"*.\"` and both commands will list all files in the current directory that have an extension. This is because the `.` character is a wildcard in DOS, and it matches any single character. When you use `\"<.<\"`, it matches any file that has a single character extension, which is the same as using the `*.` wildcard.\n\nSo in summary, `dir \"<.<\"` and `dir \"*.*\"` are not equivalent, but `dir \"<.<\"` and `dir \"*.\"` are equivalent and will list the same files in the current directory that have an extension."
      },
      {
        "from": "human",
        "value": "Well no, `dir \"<.<\"` returns all the files with an extension in the current directory"
      },
      {
        "from": "gpt",
        "value": "You are correct again, and I apologize for my mistake. When you execute the command `dir \"<.<\"` in DOS, it is interpreted as a request to list all files in the current directory that have an extension. The `.<.<` argument is interpreted as a literal string and does not have any special meaning as a wildcard or a directory reference.\n\nSo, the reason why `dir \"<.<\"` returns all files with an extension is that the argument is not recognized as a valid wildcard or directory reference, and is simply treated as a string that matches any filename that has an extension.\n\nOnce again, I apologize for my previous incorrect explanations. Thank you for bringing this to my attention, and please let me know if you have any further questions."
      }
    ]
}
```


## Dataset Deployment
- The file `ShareGPT_V3_unfiltered_cleaned_split.json` contains over 58,000 multi-turn conversation groups (more than 200,000 turns in total). Download link 🔗: [https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json)
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set for dataset tasks). Taking deployment on a Linux server as an example, the specific execution steps are as follows:
```bash
# Within the Linux server, under the tool root path
cd ais_bench/datasets
mkdir sharegpt/
cd sharegpt/
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json
```
- Execute `tree sharegpt/` in the directory `{tool_root_path}/ais_bench/datasets` to verify the directory structure. The deployment is successful if the structure matches the following:
    ```
    sharegpt
    └── ShareGPT_V3_unfiltered_cleaned_split.json
    ```


## Available Dataset Tasks

| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Corresponding Source Code File Path |
| --- | --- | --- | --- | --- | --- |
| sharegpt_gen | Generative task for ShareGPT | Accuracy evaluation not supported temporarily | 0-shot | List Format | [sharegpt_gen.py](sharegpt_gen.py) |


*Note: The evaluation of this multi-turn conversation dataset supports service deployment frameworks such as vLLM, SGLang, and MindIE Service. When using it, you need to specify `--models` as `vllm_api_stream_chat_multiturn`.*