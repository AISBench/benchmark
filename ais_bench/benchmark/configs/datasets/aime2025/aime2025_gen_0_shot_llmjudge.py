from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
from ais_bench.benchmark.datasets import (
    Aime2025Dataset,
    Aime2025JDGDataset,
)
from ais_bench.benchmark.datasets.utils.llm_judge import get_a_or_b, LLMJudgeCorrectEvaluator


aime2025_reader_cfg = dict(input_columns=["question"], output_column="answer")


aime2025_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    prompt="{question}\nRemember to put your final answer within \\boxed{}.",
                ),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

GRADER_TEMPLATE = """
    Please as a grading expert, judge whether the final answers given by the candidates below are consistent with the standard answers, that is, whether the candidates answered correctly.

    Here are some evaluation criteria:
    1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the standard answer according to the form of the question. Don't try to answer the original question. You can assume that the standard answer is definitely correct.
    2. Because the candidate's answer may be different from the standard answer in the form of expression, before making a judgment, please understand the question and the standard answer first, and then judge whether the candidate's answer is correct, but be careful not to try to answer the original question.
    3. Some answers may contain multiple items, such as multiple-choice questions, multiple-select questions, fill-in-the-blank questions, etc. As long as the answer is the same as the standard answer, it is enough. For multiple-select questions and multiple-blank fill-in-the-blank questions, the candidate needs to answer all the corresponding options or blanks correctly to be considered correct.
    4. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some answers may be a textual description, as long as the meaning expressed is the same. And some formulas are expressed in different ways, but they are equivalent and correct.
    5. If the prediction is given with \\boxed{}, please ignore the \\boxed{} and only judge whether the candidate's answer is consistent with the standard answer.
    6. If the candidate's answer is semantically incomplete at the end, please judge it as inconsistent.

    Please judge whether the following answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this new question as one of:
    A: Means the answer is consistent with the standard answer.
    B: Means the answer is inconsistent with the standard answer.
    Just return the letters "A" or "B", with no text around it.

    Here is your task. Simply reply with either CORRECT, INCORRECT. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.


    <Original Question Begin>: \n{question}\n<Original Question End>\n\n
    <Gold Target Begin>: \n{answer}\n<Gold Target End>\n\n
    <Predicted Answer Begin>: \n{model_answer}\n<Predicted End>\n\n

    Judging the correctness of candidates' answers, please return the the letters "A" or "B" first before your thinking:
""".strip()

aime2025_judge_infer_cfg = dict(
    judge_reader_cfg = dict(input_columns=["question", "answer", "model_answer"], output_column="model_pred_uuid"),
    judge_model=dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="judge", # Be added after dataset abbr
        path="",
        model="",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=8080,
        url="",
        max_out_len=512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            ignore_eos=False,
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    ),
    judge_dataset_type=Aime2025JDGDataset,
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt="You are a helpful assistant who evaluates the correctness and quality of models' outputs.",
                )
            ],
            round=[
                dict(role='HUMAN', prompt=GRADER_TEMPLATE),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

aime2025_eval_cfg = dict(
    evaluator=dict(type=LLMJudgeCorrectEvaluator),
    pred_postprocessor=dict(type=get_a_or_b),
)

aime2025_datasets = [
    dict(
        abbr="aime2025",
        type=Aime2025Dataset,
        path="ais_bench/datasets/aime2025/aime2025.jsonl",
        reader_cfg=aime2025_reader_cfg,
        infer_cfg=aime2025_infer_cfg,
        judge_infer_cfg=aime2025_judge_infer_cfg,
        eval_cfg=aime2025_eval_cfg,
    )
]
