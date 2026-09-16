from ais_bench.benchmark.datasets import HLEDataset, HLEJGDataset
from ais_bench.benchmark.datasets.hle import HLEJudgeEvaluator
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import (MMPromptTemplate,
                                                             PromptTemplate)
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

# Model inference prompt template - aligned with official HLE format
HLE_INFER_PROMPT = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"


hle_reader_cfg = dict(input_columns=["question", "image"], output_column="answer")


hle_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            begin=[
                dict(
                    role="SYSTEM",
                    prompt=HLE_INFER_PROMPT,
                )
            ],
            round=[
                dict(
                    role="HUMAN",
                    prompt_mm={
                        "text": {"type": "text", "text": "{question}"},
                        "image": {"type": "image_url", "image_url": {"url": "{image}"}},
                    },
                )
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


# Judge model prompt template. Instructs the judge to extract answer and evaluate correctness - aligned with official HLE format
JUDGE_PROMPT = """
    Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

    [question]: {question}

    [response]: {model_answer}

    Your judgement must be in the format and criteria specified below:

    extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

    [correct_answer]: {answer}

    reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

    correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

    confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


# JSON schema for structured judge model output
RESPONSE_FORMAT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "schema": {
            "properties": {
                "extracted_final_answer": {
                    "title": "Extracted Final Answer",
                    "type": "string",
                },
                "reasoning": {"title": "Reasoning", "type": "string"},
                "correct": {
                    "title": "Correct",
                    "enum": ["yes", "no"],
                    "type": "string",
                },
                "confidence": {"title": "Confidence", "type": "integer"},
                "strict": {"const": True, "type": "boolean"},
            },
            "required": [
                "extracted_final_answer",
                "reasoning",
                "correct",
                "confidence",
                "strict",
            ],
            "type": "object",
            "additionalProperties": False,
        },
        "name": "ExtractedAnswer",
        "strict": True,
    },
}


hle_judge_infer_cfg = dict(
    judge_reader_cfg=dict(
        input_columns=["question", "answer", "model_answer"],
        output_column="model_pred_uuid",
    ),
    judge_model=dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="judge",
        path="",
        model="",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=8080,
        url="",
        max_out_len=8192,
        batch_size=100,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            seed=0,
            response_format=RESPONSE_FORMAT_SCHEMA,
            chat_template_kwargs=dict(
                enable_thinking=False,
            ),
        ),
    ),
    judge_dataset_type=HLEJGDataset,
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt=JUDGE_PROMPT),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


hle_eval_cfg = dict(
    evaluator=dict(type=HLEJudgeEvaluator),
)


hle_datasets = [
    dict(
        abbr="hle",
        type=HLEDataset,
        path="ais_bench/datasets/hle/data/test-00000-of-00001.parquet",
        reader_cfg=hle_reader_cfg,
        infer_cfg=hle_infer_cfg,
        judge_infer_cfg=hle_judge_infer_cfg,
        eval_cfg=hle_eval_cfg,
    )
]
