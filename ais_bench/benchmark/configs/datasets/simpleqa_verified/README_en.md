# SimpleQA Verified

[中文](README.md)

Google DeepMind / Google Research's 1,000-example English factuality benchmark. Official dataset: https://huggingface.co/datasets/google/simpleqa-verified , revision `0dc97e0d28d8233463e005cdc4475cc2a13ba2dc` (MIT according to its card). Keep the upstream attribution. Use the Verified subset, not the original SimpleQA collection.

## Deploy

```bash
mkdir -p ais_bench/datasets/simpleqa_verified
curl -L https://huggingface.co/datasets/google/simpleqa-verified/resolve/0dc97e0d28d8233463e005cdc4475cc2a13ba2dc/simpleqa_verified.csv -o ais_bench/datasets/simpleqa_verified/simpleqa_verified.csv
```

```text
ais_bench/datasets/simpleqa_verified/
└── simpleqa_verified.csv
```

Alternatively deploy the same version downloaded from ModelScope, or set a local CSV/JSONL path. Records must be objects. problem/answer are nonempty strings; original_index is a nonempty string or integer, excluding booleans. IDs are stripped and checked for duplicates; metadata is preserved. The official eval split maps to test with an empty train split. No tools, retrieval or few-shot examples are used.

| Task | Metrics | Shot / Prompt | Dataset class | Config |
|---|---|---|---|---|
| simpleqa_verified_gen_0_shot_chat | accuracy / F1, 0–1 | 0-shot, original question in chat | SimpleQAVerifiedDataset | [Config](simpleqa_verified_gen_0_shot_chat.py) |

## Run

Configure the target model and the independent `judge_infer_cfg.judge_model` service. GPT-4.1 is the official judge; localhost:8005 is a placeholder, not an included service. Do not commit credentials.

```bash
ais_bench --models vllm_api_general_chat --datasets simpleqa_verified_gen_0_shot_chat --num-prompts 10 --num-warmups 0
```

The default all workflow runs generation, judging, evaluation and summarization. Remove num-prompts for all 1,000 records. Check both prediction stages, UUID joins, failures and metrics, not only the CLI exit code.

## Protocol

The full template is from the official card's [Nan Liao starter notebook v9](https://www.kaggle.com/code/nanliao7/simpleqa-verified-benchmark-starter-code); see also https://arxiv.org/abs/2509.07968 . Template attribution is separate from the dataset's MIT label; no broader license claim is made for notebook code.

Accept exact A/CORRECT, B/INCORRECT, C/NOT_ATTEMPTED (or NOT ATTEMPTED), ignoring case and outer whitespace. Ambiguous/invalid judge outputs withhold metrics and report invalid_judge_count instead of silently defaulting to C. This also avoids the starter's permissive substring parsing issue.

All metrics use 0–1: accuracy/correct=A/N, incorrect=B/N, not_attempted=C/N, accuracy_given_attempted=A/(A+B), F1=2A/(N+A+B); no attempts gives zero. Details preserve the judge output, grade and original model prediction UUID. Alternative judges are not comparable official results.

## Validation

Tests: `python -m pytest tests/UT/datasets/test_simpleqa_verified.py tests/UT/datasets/utils/test_llm_judge.py -q`.

Historical initial Ubuntu ARM64/Python3.10 CLI smoke: first 10 eval records, deepseek-flash in both roles, temperature=0, thinking disabled, target max_tokens=1024 / judge=32, one attempt per record. Both sets of 10 requests succeeded; A/B/C=7/3/0, invalid=0, accuracy=F1=0.70. UUID joins and independent recomputation matched the summary. This is neither full-set evaluation nor GPT-4.1 reproduction. Current loader/config-test revisions are validated separately in the PR; historical scores are not described as a new model run.
