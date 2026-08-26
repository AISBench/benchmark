# LongBench
[中文](README.md) | English
## Dataset Introduction
LongBench is a benchmark designed for bilingual, multi-task, and comprehensive evaluation of Large Language Models' (LLMs) long-context understanding capabilities. It covers multiple languages (Chinese and English) to enable a more holistic assessment of LLMs' multilingual performance in long contexts. Additionally, LongBench includes six major categories and twenty-one distinct tasks, encompassing key long-text application scenarios such as single-document QA, multi-document QA, summarization, few-shot learning, synthetic tasks, and code completion.

LongBench comprises 14 English tasks, 5 Chinese tasks, and 2 code tasks. The average length of most tasks ranges from 5k to 15k tokens, with a total of 4,750 test samples.

> 🔗 Dataset Homepage Link: [https://huggingface.co/datasets/zai-org/LongBench](https://huggingface.co/datasets/zai-org/LongBench)

## Dataset Deployment
It is recommended to download the dataset from Hugging Face: [https://huggingface.co/datasets/zai-org/LongBench](https://huggingface.co/datasets/zai-org/LongBench)
- It is recommended to deploy the dataset in the directory `{tool_root_path}/ais_bench/datasets` (the default path set in dataset tasks).
- After successful deployment, execute `tree LongBench/` in the directory `{tool_root_path}/ais_bench/datasets` to check the directory structure. If the directory structure matches the one shown below, the dataset has been deployed successfully:
    ```
    LongBench/
    ├── data
    │   ├── 2wikimqa_e.jsonl
    │   ├── 2wikimqa.jsonl
    │   ├── dureader.jsonl
    │   ├── gov_report_e.jsonl
    │   ├── gov_report.jsonl
    │   ├── hotpotqa_e.jsonl
    │   ├── hotpotqa.jsonl
    │   ├── lcc_e.jsonl
    │   ├── lcc.jsonl
    │   ├── lsht.jsonl
    │   ├── multifieldqa_en_e.jsonl
    │   ├── multifieldqa_en.jsonl
    │   ├── multifieldqa_zh.jsonl
    │   ├── multi_news_e.jsonl
    │   ├── multi_news.jsonl
    │   ├── musique.jsonl
    │   ├── narrativeqa.jsonl
    │   ├── passage_count_e.jsonl
    │   ├── passage_count.jsonl
    │   ├── passage_retrieval_en_e.jsonl
    │   ├── passage_retrieval_en.jsonl
    │   ├── passage_retrieval_zh.jsonl
    │   ├── qasper_e.jsonl
    │   ├── qasper.jsonl
    │   ├── qmsum.jsonl
    │   ├── repobench-p_e.jsonl
    │   ├── repobench-p.jsonl
    │   ├── samsum_e.jsonl
    │   ├── samsum.jsonl
    │   ├── trec_e.jsonl
    │   ├── trec.jsonl
    │   ├── triviaqa_e.jsonl
    │   ├── triviaqa.jsonl
    │   └── vcsum.jsonl
    └── LongBench.py
    ```

## Available Dataset Tasks
| Task Name | Introduction | Evaluation Metric | Few-Shot | Prompt Format | Import Statement | Corresponding Source Code Configuration File Path |
| --- | --- | --- | --- | --- | --- | --- |
| longbench | LongBench main task | Accuracy | 0-shot | Chat format |`from ais_bench.benchmark.configs.datasets.longbench.longbench import longbench_datasets as datasets`| [longbench.py](longbench.py) |
| longbench_2wikimqa_gen | LongBench 2WikiMQA generative task | Accuracy | 0-shot | Chat format | [longbench_2wikimqa_gen.py](longbench2wikimqa/longbench_2wikimqa_gen.py) |
| longbench_dureader_gen | LongBench DuReader generative task | Accuracy | 0-shot | Chat format | [longbench_dureader_gen.py](longbenchdureader/longbench_dureader_gen.py) |
| longbench_gov_report_gen | LongBench GovReport generative task | Accuracy | 0-shot | Chat format | [longbench_gov_report_gen.py](longbenchgov_report/longbench_gov_report_gen.py) |
| longbench_hotpotqa_gen | LongBench HotpotQA generative task | Accuracy | 0-shot | Chat format | [longbench_hotpotqa_gen.py](longbenchhotpotqa/longbench_hotpotqa_gen.py) |
| longbench_lcc_gen | LongBench LCC generative task | Accuracy | 0-shot | Chat format | [longbench_lcc_gen.py](longbenchlcc/longbench_lcc_gen.py) |
| longbench_lsht_gen | LongBench LSHT generative task | Accuracy | 0-shot | Chat format | [longbench_lsht_gen.py](longbenchlsht/longbench_lsht_gen.py) |
| longbench_multi_news_gen | LongBench MultiNews generative task | Accuracy | 0-shot | Chat format | [longbench_multi_news_gen.py](longbenchmulti_news/longbench_multi_news_gen.py) |
| longbench_multifieldqa_en_gen | LongBench MultiFieldQA-EN generative task | Accuracy | 0-shot | Chat format | [longbench_multifieldqa_en_gen.py](longbenchmultifieldqa_en/longbench_multifieldqa_en_gen.py) |
| longbench_multifieldqa_zh_gen | LongBench MultiFieldQA-ZH generative task | Accuracy | 0-shot | Chat format | [longbench_multifieldqa_zh_gen.py](longbenchmultifieldqa_zh/longbench_multifieldqa_zh_gen.py) |
| longbench_musique_gen | LongBench MuSiQue generative task | Accuracy | 0-shot | Chat format | [longbench_musique_gen.py](longbenchmusique/longbench_musique_gen.py) |
| longbench_narrativeqa_gen | LongBench NarrativeQA generative task | Accuracy | 0-shot | Chat format | [longbench_narrativeqa_gen.py](longbenchnarrativeqa/longbench_narrativeqa_gen.py) |
| longbench_passage_count_gen | LongBench PassageCount generative task | Accuracy | 0-shot | Chat format | [longbench_passage_count_gen.py](longbenchpassage_count/longbench_passage_count_gen.py) |
| longbench_passage_retrieval_en_gen | LongBench PassageRetrieval-EN generative task | Accuracy | 0-shot | Chat format | [longbench_passage_retrieval_en_gen.py](longbenchpassage_retrieval_en/longbench_passage_retrieval_en_gen.py) |
| longbench_passage_retrieval_zh_gen | LongBench PassageRetrieval-ZH generative task | Accuracy | 0-shot | Chat format | [longbench_passage_retrieval_zh_gen.py](longbenchpassage_retrieval_zh/longbench_passage_retrieval_zh_gen.py) |
| longbench_qasper_gen | LongBench QASPER generative task | Accuracy | 0-shot | Chat format | [longbench_qasper_gen.py](longbenchqasper/longbench_qasper_gen.py) |
| longbench_qmsum_gen | LongBench QMSum generative task | Accuracy | 0-shot | Chat format | [longbench_qmsum_gen.py](longbenchqmsum/longbench_qmsum_gen.py) |
| longbench_repobench_gen | LongBench RepoBench generative task | Accuracy | 0-shot | Chat format | [longbench_repobench_gen.py](longbenchrepobench/longbench_repobench_gen.py) |
| longbench_samsum_gen | LongBench SamSum generative task | Accuracy | 0-shot | Chat format | [longbench_samsum_gen.py](longbenchsamsum/longbench_samsum_gen.py) |
| longbench_trec_gen | LongBench TREC generative task | Accuracy | 0-shot | Chat format | [longbench_trec_gen.py](longbenchtrec/longbench_trec_gen.py) |
| longbench_triviaqa_gen | LongBench TriviaQA generative task | Accuracy | 0-shot | Chat format | [longbench_triviaqa_gen.py](longbenchtriviaqa/longbench_triviaqa_gen.py) |
| longbench_vcsum_gen | LongBench VCSum generative task | Accuracy | 0-shot | Chat format | [longbench_vcsum_gen.py](longbenchvcsum/longbench_vcsum_gen.py) |

## Example Evaluation Command
```bash
ais_bench --models vllm_api_general_chat --datasets longbench
```
⚠️ Note: When executing the above command, the dataset files will be downloaded from Hugging Face by default. If there is no network connection or network issues occur, you can load the dataset locally by following these steps:
Modify the `LongBench.py` file in the LongBench folder (downloaded from Hugging Face during the dataset deployment phase):
```python
     def _split_generators(self, dl_manager):
-        data_dir = dl_manager.download_and_extract(_URL) # Delete this line to disable downloading from Hugging Face
+        data_dir = self.config.data_dir # Add this line to load the local dataset
```