# ShareGPT
中文 | [English](README_en.md)
## 数据集简介
ShareGPT数据集是一个包含大量人类与AI对话的数据集，主要用于训练和微调对话模型。ShareGPT数据集最初是通过 ShareGPT API收集的用户共享对话数据。目前ShareGPT数据集的规模已经从最初的约 52,000 条对话扩展到约 90,000 条对话。
数据样例如下，每条对话数据包含一个唯一的 id 和一个 conversations 数组，数组中每个元素表示对话的一轮，包括对话的来源（from，可以是用户 "human" 或 ChatGPT "gpt"）和对话内容（value）：
> 🔗 数据集主页链接[https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered)。
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

## 数据集部署

- ShareGPT_V3_unfiltered_cleaned_split.json数据中包含5.8W+组多轮对话，共计20W+轮，下载链接🔗 [https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json)
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
mkdir sharegpt/
cd sharegpt/
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree sharegpt/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    sharegpt
    └── ShareGPT_V3_unfiltered_cleaned_split.json
    ```

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|配套文件导入方式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- | --- |
|sharegpt_gen|sharegpt生成式任务|暂不支持精度评测|0-shot|列表格式|`from ais_bench.benchmark.configs.datasets.sharegpt.sharegpt_gen import sharegpt_datasets as datasets`|[sharegpt_gen.py](sharegpt_gen.py)|


*注意：该多轮对话数据集的测评支持vLLM、SGLang、MindIE Service等服务化，使用时需指定--models为vllm_api_stream_chat_multiturn*