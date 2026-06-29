import dataclasses
from typing import Dict, List, Optional, Union

from datasets import Dataset

from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET, ICL_EVALUATORS
from ais_bench.benchmark.datasets.utils.datasets import get_data_path
from ais_bench.benchmark.utils.logging.logger import AISLogger

from ..base import BaseDataset
from . import instructions_registry

logger = AISLogger()


@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: List[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: List[bool]


def test_instruction_following_strict(inp: InputExample, response: str) -> OutputExample:
    """Tests response to see if instructions are followed (strict mode)."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        inp.kwargs[index] = {key: value for key, value in inp.kwargs[index].items() if value is not None}
        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and 'prompt' in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(inp: InputExample, response: str) -> OutputExample:
    """Tests response for an upper bound for following instructions (loose mode)."""
    r = response.split('\n')
    response_remove_first = '\n'.join(r[1:]).strip()
    response_remove_last = '\n'.join(r[:-1]).strip()
    response_remove_both = '\n'.join(r[1:-1]).strip()
    revised_response = response.replace('*', '')
    revised_response_remove_first = response_remove_first.replace('*', '')
    revised_response_remove_last = response_remove_last.replace('*', '')
    revised_response_remove_both = response_remove_both.replace('*', '')
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and 'prompt' in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


@LOAD_DATASET.register_module()
class IFBenchDataset(BaseDataset):
    """IFBench dataset loader.

    IFBench is a benchmark designed to evaluate how reliably AI models
    follow novel, challenging, and diverse verifiable instructions,
    with a strong focus on out-of-domain generalization.
    """

    @staticmethod
    def load(path: str, name: str = 'default'):
        path = get_data_path(path, local_mode=True)
        logger.info(f"Loading IFBench dataset from: {path}")
        from datasets import Dataset
        dataset = Dataset.from_parquet(path)
        raw_data = []
        for i in range(len(dataset)):
            item = dataset[i]
            prompt = item['prompt']
            logger.info(f"[ifbench] Sample[{i}] prompt: {prompt[:200]}..."
                        if len(prompt) > 200 else f"[ifbench] Sample[{i}] prompt: {prompt}")
            raw_data.append({
                'prompt': prompt,
                'reference': item,
            })
        logger.info(f"IFBench dataset loaded: {len(raw_data)} samples")
        return Dataset.from_list(raw_data)


@ICL_EVALUATORS.register_module()
class IFBenchEvaluator(BaseEvaluator):
    """IFBench evaluator using strict and loose instruction following checks.

    Metrics:
        - Prompt-level-strict-accuracy: fraction of prompts where ALL constraints
          are strictly satisfied
        - Inst-level-strict-accuracy: fraction of individual constraints that are
          strictly satisfied
        - Prompt-level-loose-accuracy: fraction of prompts where ALL constraints
          are satisfied under loose evaluation
        - Inst-level-loose-accuracy: fraction of individual constraints that are
          satisfied under loose evaluation
    """

    def score(self, predictions: List, references: List, origin_prompt: List = None) -> dict:
        logger.info(f"Starting IFBench evaluation with {len(predictions)} samples")

        prompt_strict_correct, prompt_strict_total = 0, 0
        inst_strict_correct, inst_strict_total = 0, 0
        prompt_loose_correct, prompt_loose_total = 0, 0
        inst_loose_correct, inst_loose_total = 0, 0
        details = {}

        for index, (pred, refer) in enumerate(zip(predictions, references)):
            inp = InputExample(
                key=refer['key'],
                instruction_id_list=refer['instruction_id_list'],
                prompt=refer['prompt'],
                kwargs=refer['kwargs'])
            for kwarg in inp.kwargs:
                for k in list(kwarg.keys()):
                    if kwarg[k] is None:
                        kwarg.pop(k, None)

            prompt_text = refer.get('prompt', '')
            pred_text = pred or ''
            logger.info(
                f"[ifbench] Sample[{index}] prompt: {prompt_text[:200]}..."
                if len(prompt_text) > 200 else f"[ifbench] Sample[{index}] prompt: {prompt_text}"
            )
            logger.info(
                f"[ifbench] Sample[{index}] prediction: {pred_text[:200]}..."
                if len(pred_text) > 200 else f"[ifbench] Sample[{index}] prediction: {pred_text}"
            )
            logger.info(
                f"[ifbench] Sample[{index}] instruction_ids: {refer['instruction_id_list']}"
            )

            # strict evaluation
            example = test_instruction_following_strict(inp, pred)
            follow_instruction_list = example.follow_instruction_list
            instruction_id_list = example.instruction_id_list
            prompt_strict_total += 1
            is_strict_correct = all(follow_instruction_list)
            prompt_strict_correct += is_strict_correct
            inst_strict_total += len(instruction_id_list)
            inst_strict_correct += sum(follow_instruction_list)

            # loose evaluation
            example = test_instruction_following_loose(inp, pred)
            follow_instruction_list = example.follow_instruction_list
            instruction_id_list = example.instruction_id_list
            prompt_loose_total += 1
            is_loose_correct = all(follow_instruction_list)
            prompt_loose_correct += is_loose_correct
            inst_loose_total += len(instruction_id_list)
            inst_loose_correct += sum(follow_instruction_list)

            if is_strict_correct:
                grade = 'strict'
            elif is_loose_correct:
                grade = 'loose'
            else:
                grade = 'none'

            logger.info(
                f"[ifbench] Sample[{index}] strict: {sum(follow_instruction_list)}/{len(follow_instruction_list)}"
                f" (prompt_level={'PASS' if is_strict_correct else 'FAIL'}), "
                f"loose: prompt_level={'PASS' if is_loose_correct else 'FAIL'}, "
                f"grade={grade}"
            )

            details[str(index)] = {
                'prompt': origin_prompt[index] if origin_prompt else refer.get('prompt', ''),
                'pred': pred,
                'refer': refer,
                'is_strict_correct': is_strict_correct,
                'is_loose_correct': is_loose_correct,
                'is_correct': is_strict_correct,
                'grade': grade,
            }

        results = {
            'Prompt-level-strict-accuracy':
                prompt_strict_correct / prompt_strict_total * 100 if prompt_strict_total else 0,
            'Inst-level-strict-accuracy':
                inst_strict_correct / inst_strict_total * 100 if inst_strict_total else 0,
            'Prompt-level-loose-accuracy':
                prompt_loose_correct / prompt_loose_total * 100 if prompt_loose_total else 0,
            'Inst-level-loose-accuracy':
                inst_loose_correct / inst_loose_total * 100 if inst_loose_total else 0,
            'details': details,
        }

        logger.info("=" * 60)
        logger.info("[ifbench] Evaluation Results:")
        logger.info(f"  Prompt-level-strict-accuracy:  {results['Prompt-level-strict-accuracy']:.2f}%")
        logger.info(f"  Inst-level-strict-accuracy:    {results['Inst-level-strict-accuracy']:.2f}%")
        logger.info(f"  Prompt-level-loose-accuracy:   {results['Prompt-level-loose-accuracy']:.2f}%")
        logger.info(f"  Inst-level-loose-accuracy:     {results['Inst-level-loose-accuracy']:.2f}%")
        logger.info(f"  Total samples:                 {prompt_strict_total}")
        logger.info("=" * 60)

        return results
