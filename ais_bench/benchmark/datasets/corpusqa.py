"""CorpusQA dataset integration for aisbench.

CorpusQA (https://github.com/Tongyi-Zhiwen/CorpusQA) is a long-context
question-answering benchmark that stresses 1M-token context windows.  Each
sample in the JSONL file contains:

- ``id``: unique sample id
- ``prompt``: raw multi-turn chat messages (list of ``{role, content}``)
- ``question``: the question text
- ``answer``: the reference answer

Reproduction follows the official CorpusQA evaluation protocol: the model
generates an answer from the raw prompt, then an LLM judge (ORM - Output
Reward Model) decides whether the generated answer is equivalent to the
reference answer, outputting ``[[YES]]`` or ``[[NO]]``.
"""

import json
import os
import re
from typing import Any, Dict, List, Union

from datasets import Dataset

from ais_bench.benchmark.datasets.base import BaseDataset
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.datasets.utils.llm_judge import LLMJudgeDataset
from ais_bench.benchmark.openicl.icl_evaluator.icl_base_evaluator import BaseEvaluator
from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_base import BasePromptTemplate
from ais_bench.benchmark.registry import ICL_EVALUATORS, ICL_PROMPT_TEMPLATES, LOAD_DATASET
from ais_bench.benchmark.utils.logging import AISLogger

logger = AISLogger()


# ---------------------------------------------------------------------------
# Official CorpusQA ORM (Output Reward Model) judge prompt
# Byte-identical to the official eval.py:
#   - system message: GENERAL_ORM_PROMPT
#   - user message:   ORM_USER_TEMPLATE.format(problem=question, answer_1=..., answer_2=...)
# ---------------------------------------------------------------------------
CORPUSQA_JUDGE_SYSTEM_PROMPT = """You are an expert in verifying if two answers are the same.
Your input is a problem and two answers, Answer 1 and Answer 2. You need to check if they are equivalent.
Your task is to determine if two answers are equivalent, without attempting to solve the original problem.
Compare the answers to verify they represent identical values or meaning, even when written in different forms or notations.

Your output must follow the following format:
1) Provide an explanation for why the answers are equivalent or not.
2) Then provide your final answer in the form of: [[YES]] or [[NO]]
"""

# Leading/trailing newlines preserved exactly as in the official template.
CORPUSQA_JUDGE_USER_TEMPLATE = """
Problem: {question}
Answer 1: {model_answer}
Answer 2: {answer}
"""


@ICL_PROMPT_TEMPLATES.register_module("corpusqa_prompt")
class CorpusQAPromptTemplate(BasePromptTemplate):
    """Pass-through template for CorpusQA raw multi-turn prompts.

    CorpusQA stores the conversation as a list of messages
    (``[{role, content, ...}]``).  This template converts that list into a
    :class:`PromptList` so that the chat API model forwards the messages
    unchanged, preserving the original prompt exactly as released.
    """

    def __init__(
        self,
        template: Union[Dict, str] = "",
        ice_token: str = None,
        sep_token: str = None,
    ) -> None:
        super().__init__(template=template or "", ice_token=ice_token, sep_token=sep_token)

    def generate_item(
        self,
        entry: Dict,
        output_field=None,
        output_field_replace_token: str = "",
        ice_field_replace_token: str = "",
    ):
        messages = entry.get("prompt", [])
        if isinstance(messages, str):
            # Some subsets may provide a plain text prompt.
            return messages
        if isinstance(messages, dict):
            messages = [messages]

        # Reuse the framework's standard section emission (``_encode_template``),
        # exactly like the built-in PromptTemplate used by other gen datasets
        # (e.g. gsm8k_gen_4_shot_cot_chat_prompt). System messages live in the
        # ``begin`` section; user/assistant turns in the ``round`` section.
        template = {}
        system_items = []
        round_items = []
        for msg in messages:
            if not isinstance(msg, dict):
                msg = {"role": "user", "content": str(msg)}
            # Map the released OpenAI-style roles onto the internal roles the
            # API chat model understands (see ROLE_MAP in vllm_custom_api_chat):
            #   system -> SYSTEM (begin section)
            #   user / assistant -> HUMAN / BOT (round section)
            raw_role = msg.get("role", "user")
            if raw_role == "system":
                role = "SYSTEM"
            elif raw_role == "assistant":
                role = "BOT"
            else:
                role = "HUMAN"
            item = {"role": role, "prompt": msg.get("content", "")}
            for key, value in msg.items():
                if key not in ("role", "content"):
                    item[key] = value
            if role == "SYSTEM":
                system_items.append(item)
            else:
                round_items.append(item)
        if system_items:
            template["begin"] = system_items
        template["round"] = round_items
        return self._encode_template(template, ice=False)


@LOAD_DATASET.register_module()
class CorpusQADataset(BaseDataset):
    """CorpusQA dataset.

    Streams the JSONL file line by line so that the 1M-token / large sample
    file can be loaded without exhausting memory, and keeps the raw
    multi-turn ``prompt`` field untouched for faithful reproduction.
    """

    @staticmethod
    def load(path: str, **kwargs) -> Dataset:
        path = get_data_path(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"CorpusQA dataset file not found: {path}")

        dataset = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(f"Failed to parse line, skipped: {exc}")
                    continue
                if "prompt" not in item or "answer" not in item:
                    logger.warning("Line misses 'prompt' or 'answer' field, skipped.")
                    continue
                # Answers may be str, int/float or the empty list ``[]`` (see
                # the official README). Arrow cannot build a column mixing
                # these types, so normalise non-str answers to their str()
                # rendering -- this keeps the judge prompt byte-identical to
                # the official ``str.format`` output (e.g. ``Answer 2: []``).
                answer = item["answer"]
                if not isinstance(answer, str):
                    answer = str(answer)
                dataset.append(
                    {
                        "id": item.get("id", len(dataset)),
                        "prompt": item["prompt"],
                        "question": item.get("question", ""),
                        "answer": answer,
                    }
                )
        logger.info(f"Loaded {len(dataset)} samples from CorpusQA dataset: {path}")
        return Dataset.from_list(dataset)


class CorpusQAJGDataset(LLMJudgeDataset):
    """CorpusQA judge dataset.

    Follows the HLE / AA-LCR pattern: subclasses :class:`LLMJudgeDataset`
    and merges the model predictions into the original dataset items so the
    judge prompt can reference ``question``, ``answer`` and ``model_answer``.
    """

    @staticmethod
    def _extract_answer(response: str) -> str:
        """Mirror the official ``extract_answer`` helper.

        The official script runs this extraction on the model's *final
        response only* (the API ``content`` field).  aisbench's saved
        ``prediction`` concatenates ``reasoning_content + "\\n\\n" +
        content`` (see ``Output.get_prediction``), so a first-match
        ``re.search`` can hit a draft ``The answer is: ...`` line inside
        the reasoning, and the judge would compare a reasoning fragment
        against the gold answer.  Taking the last occurrence preserves the
        official semantics because the model's final answer line is the
        last ``The answer is:`` in the concatenated text.
        """
        if not isinstance(response, str):
            response = str(response)
        matches = re.findall(r"The answer is: (.*)", response)
        if matches:
            return matches[-1].strip()
        return response.strip()

    def _modify_dataset_item(self, dataset_item, pred_item):
        super()._modify_dataset_item(dataset_item, pred_item)
        # Prefer the reasoning-free final content when the prediction file
        # provides it (see GenInferencerOutputHandler); fall back to the
        # concatenated prediction for result files produced before that
        # field existed.
        raw_answer = pred_item.get("content")
        if raw_answer is None:
            raw_answer = dataset_item.get("model_answer")
        if raw_answer is not None:
            dataset_item["model_answer"] = self._extract_answer(str(raw_answer))
        return dataset_item

    def _get_dataset_class(self):
        return CorpusQADataset


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
@ICL_EVALUATORS.register_module()
class CorpusQAEvaluator(BaseEvaluator):
    """CorpusQA ORM evaluator.

    Parses the judge model outputs (expected to contain ``[[YES]]`` or
    ``[[NO]]``) and computes accuracy, mirroring the official CorpusQA
    evaluation script.
    """

    def __init__(self):
        super().__init__()

    def score(self, predictions: List[str], references: List[Any]) -> Dict[str, Any]:
        if len(predictions) != len(references):
            return {
                "error": (
                    "predictions and references have different length: "
                    f"len(predictions)={len(predictions)}, "
                    f"len(references)={len(references)}"
                )
            }

        details = []
        correct = 0
        total = 0
        for index, (judge_output, ref) in enumerate(zip(predictions, references)):
            is_correct = self._is_correct(judge_output)
            if is_correct is None:
                logger.warning(
                    f"Judge output {index} does not contain [[YES]]/[[NO]], "
                    "treated as incorrect."
                )
                is_correct = False
            if is_correct:
                correct += 1
            total += 1
            details.append(
                {
                    "id": index,
                    "judge_output": judge_output,
                    "answer": ref,
                    "correct": is_correct,
                }
            )

        return {
            "accuracy": 100.0 * correct / total if total else 0.0,
            "num_correct": correct,
            "num_total": total,
            "details": details,
        }

    @staticmethod
    def _is_correct(judge_output: str):
        """Mirror the official parsing rule on the judge's *final verdict*.

        The official script parses ``[[YES]]``/``[[NO]]`` from the judge's
        final content only.  aisbench's saved ``prediction`` concatenates
        ``reasoning_content + "\\n\\n" + content``, and a thinking judge
        often mentions both markers while reasoning before settling on
        one, which breaks the official any-marker substring rule (e.g.
        reasoning about ``[[NO]]`` then concluding ``[[YES]]`` is scored
        as incorrect).  Taking the last marker matches the official
        verdict because the final content comes after the reasoning in the
        concatenated text.
        """
        if not judge_output:
            return None
        markers = re.findall(r"\[\[(YES|NO)\]\]", judge_output)
        if markers:
            return markers[-1] == "YES"
        return None
