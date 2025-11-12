# mbpp
中文 | [English](README_en.md)
## 数据集简介
mbpp基准测试包含约1,000个众包Python编程题目，难度设计为入门级程序员可解决，涵盖编程基础、标准库功能等内容。每个题目包含任务描述、代码解决方案和3个自动化测试用例。如论文所述，我们已对部分数据进行了人工验证。

> 🔗 数据集主页[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip)

## 数据集部署
- 可以从opencompass提供的链接🔗 [http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip)下载数据集压缩包。
- 建议部署在`{工具根路径}/ais_bench/datasets`目录下（数据集任务中设置的默认路径），以linux上部署为例，具体执行步骤如下：
```bash
# linux服务器内，处于工具根路径下
cd ais_bench/datasets
wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip
unzip mbpp.zip
rm mbpp.zip
```
- 在`{工具根路径}/ais_bench/datasets`目录下执行`tree mbpp/`查看目录结构，若目录结构如下所示，则说明数据集部署成功。
    ```
    mbpp
    ├── mbpp.jsonl
    └── sanitized-mbpp.jsonl
    ```

## 可用数据集任务
### mbpp_passk_gen_3_shot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|mbpp_passk_gen_3_shot_chat_prompt|mbpp数据集生成式任务，支持测pass@k(默认pass@1)|pass@1|3-shot|对话格式|[mbpp_passk_gen_3_shot_chat_prompt.py](mbpp_passk_gen_3_shot_chat_prompt.py)|
|sanitized_mbpp_passk_gen_3_shot_chat_prompt|sanitized mbpp数据集生成式任务，支持测pass@k(默认pass@1)|pass@1|3-shot|对话格式|[sanitized_mbpp_passk_gen_3_shot_chat_prompt.py](sanitized_mbpp_passk_gen_3_shot_chat_prompt.py)|
