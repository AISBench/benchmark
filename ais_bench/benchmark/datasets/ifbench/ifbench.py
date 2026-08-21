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
    logger.debug(
        f"[ifbench][strict] ========== strict check start ==========\n"
        f"  inp.key: {inp.key}  instruction_id_list: {inp.instruction_id_list}\n"
        f"  inp.prompt: {inp.prompt}\n"
        f"  inp.kwargs: {inp.kwargs}\n"
        f"  response: {response}\n"
        f"  instruction_list: {instruction_list}"
    )

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = inp.kwargs[index]

        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()

        if args and 'prompt' in args:
            instruction.build_description(prompt=inp.prompt)

        check_result = instruction.check_following(response)

        if response.strip() and check_result:
            is_following_list.append(True)
        else:
            is_following_list.append(False)

        logger.debug(
            f"[ifbench][strict] --- instruction[{index}] '{instruction_id}' ---\n"
            f"  instruction_cls: {instruction_cls}  instruction: {instruction}\n"
            f"  kwargs[{index}]: {kwargs}  args: {args}\n"
            f"  response.strip(): '{response.strip()}'  check_result: {check_result}\n"
            f"  passed: {is_following_list[-1]}"
        )

    result = OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )
    logger.debug(
        f"[ifbench][strict] ========== strict check end ==========\n"
        f"  result.instruction_id_list: {result.instruction_id_list}\n"
        f"  result.prompt: {result.prompt}\n"
        f"  result.response: {result.response}\n"
        f"  result.follow_all_instructions: {result.follow_all_instructions}\n"
        f"  result.follow_instruction_list: {result.follow_instruction_list}"
    )
    return result


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

    all_responses_str = '\n'.join(f"    all_responses[{i}]: '{ar}'" for i, ar in enumerate(all_responses))
    logger.debug(
        f"[ifbench][loose] ========== loose check start ==========\n"
        f"  inp.key: {inp.key}  instruction_id_list: {inp.instruction_id_list}\n"
        f"  inp.prompt: {inp.prompt}\n"
        f"  inp.kwargs: {inp.kwargs}\n"
        f"  response: {response}\n"
        f"  r (split lines): {r}\n"
        f"  response_remove_first: '{response_remove_first}'\n"
        f"  response_remove_last: '{response_remove_last}'\n"
        f"  response_remove_both: '{response_remove_both}'\n"
        f"  revised_response: '{revised_response}'\n"
        f"  revised_response_remove_first: '{revised_response_remove_first}'\n"
        f"  revised_response_remove_last: '{revised_response_remove_last}'\n"
        f"  revised_response_remove_both: '{revised_response_remove_both}'\n"
        f"  all_responses (8 variants):\n{all_responses_str}\n"
        f"  instruction_list: {instruction_list}"
    )

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()

        if args and 'prompt' in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        matched_vi = None
        matched_ar = None
        for vi, ar in enumerate(all_responses):
            if ar.strip() and instruction.check_following(ar):
                matched_vi = vi
                matched_ar = ar
                is_following = True
                break

        is_following_list.append(is_following)

        match_info = f"  matched at all_responses[{matched_vi}]: '{matched_ar}'\n" if is_following else ""
        logger.debug(
            f"[ifbench][loose] --- instruction[{index}] '{instruction_id}' ---\n"
            f"  instruction_cls: {instruction_cls}  instruction: {instruction}\n"
            f"  kwargs[{index}]: {inp.kwargs[index]}  args: {args}\n"
            f"{match_info}"
            f"  passed: {is_following}"
        )

    result = OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )
    logger.debug(
        f"[ifbench][loose] ========== loose check end ==========\n"
        f"  result.instruction_id_list: {result.instruction_id_list}\n"
        f"  result.prompt: {result.prompt}\n"
        f"  result.response: {result.response}\n"
        f"  result.follow_all_instructions: {result.follow_all_instructions}\n"
        f"  result.follow_instruction_list: {result.follow_instruction_list}"
    )
    return result


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
        logger.info(f"[ifbench][load] Loading IFBench dataset from: {path}")
        from datasets import Dataset
        dataset = Dataset.from_parquet(path)
        raw_data = []
        for i in range(len(dataset)):
            item = dataset[i]
            prompt = item['prompt']
            logger.debug(
                f"[ifbench][load] Sample[{i}] prompt: {prompt}\n"
                f"  item: {item}"
            )
            raw_data.append({
                'prompt': prompt,
                'reference': item,
            })
        logger.info(f"[ifbench][load] IFBench dataset loaded: {len(raw_data)} samples")
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
        logger.info(f"[ifbench][score] ========== Score Start, total samples: {len(predictions)} ==========")

        prompt_strict_correct, prompt_strict_total = 0, 0
        prompt_loose_correct, prompt_loose_total = 0, 0
        inst_strict_ratios = []
        inst_loose_ratios = []
        details = {}

        for index, (pred, refer) in enumerate(zip(predictions, references)):
            inp = InputExample(
                key=refer['key'],
                instruction_id_list=refer['instruction_id_list'],
                prompt=refer['prompt'],
                kwargs=[
                    {k: v for k, v in kwarg.items() if v is not None}
                    for kwarg in refer.get('kwargs', [])
                ])

            prompt_text = refer.get('prompt', '')
            pred_text = pred or ''

            logger.debug(
                f"[ifbench][score] --- Sample[{index}] ---\n"
                f"  pred: {pred}\n"
                f"  refer keys: {list(refer.keys())}\n"
                f"  refer['key']: {refer.get('key')}\n"
                f"  refer['instruction_id_list']: {refer.get('instruction_id_list')}\n"
                f"  refer['prompt']: {refer.get('prompt')}\n"
                f"  refer['kwargs']: {refer.get('kwargs')}\n"
                f"  InputExample.key: {inp.key}\n"
                f"  InputExample.instruction_id_list: {inp.instruction_id_list}\n"
                f"  InputExample.prompt: {inp.prompt}\n"
                f"  InputExample.kwargs (cleaned): {inp.kwargs}\n"
                f"  prompt_text: {prompt_text}\n"
                f"  pred_text: {pred_text}"
            )

            # strict evaluation
            example_strict = test_instruction_following_strict(inp, pred_text)

            follow_instruction_list_strict = example_strict.follow_instruction_list
            instruction_id_list_strict = example_strict.instruction_id_list
            prompt_strict_total += 1
            is_strict_correct = all(follow_instruction_list_strict)
            prompt_strict_correct += is_strict_correct
            inst_strict_ratios.append(
                sum(follow_instruction_list_strict) / len(instruction_id_list_strict)
                if instruction_id_list_strict else 0.0
            )

            # loose evaluation
            example_loose = test_instruction_following_loose(inp, pred_text)

            follow_instruction_list_loose = example_loose.follow_instruction_list
            instruction_id_list_loose = example_loose.instruction_id_list
            prompt_loose_total += 1
            is_loose_correct = all(follow_instruction_list_loose)
            prompt_loose_correct += is_loose_correct
            inst_loose_ratios.append(
                sum(follow_instruction_list_loose) / len(instruction_id_list_loose)
                if instruction_id_list_loose else 0.0
            )

            if is_strict_correct:
                grade = 'strict'
            elif is_loose_correct:
                grade = 'loose'
            else:
                grade = 'none'

            details[str(index)] = {
                'prompt': origin_prompt[index] if origin_prompt else refer.get('prompt', ''),
                'pred': pred,
                'refer': refer,
                'is_strict_correct': is_strict_correct,
                'is_loose_correct': is_loose_correct,
                'is_correct': is_strict_correct,
                'grade': grade,
            }

            logger.debug(
                f"[ifbench][score] Sample[{index}] grade={grade} "
                f"strict={is_strict_correct} loose={is_loose_correct} "
                f"accum: prompt_s={prompt_strict_correct}/{prompt_strict_total} "
                f"inst_s={inst_strict_ratios[-1]:.2f} "
                f"prompt_l={prompt_loose_correct}/{prompt_loose_total} "
                f"inst_l={inst_loose_ratios[-1]:.2f}\n"
                f"  strict OutputExample: instruction_id_list={example_strict.instruction_id_list}\n"
                f"    prompt={example_strict.prompt}\n"
                f"    response={example_strict.response}\n"
                f"    follow_all_instructions={example_strict.follow_all_instructions}\n"
                f"    follow_instruction_list={example_strict.follow_instruction_list}\n"
                f"  strict: sum/count={sum(follow_instruction_list_strict)}/{len(follow_instruction_list_strict)}\n"
                f"  loose OutputExample: instruction_id_list={example_loose.instruction_id_list}\n"
                f"    prompt={example_loose.prompt}\n"
                f"    response={example_loose.response}\n"
                f"    follow_all_instructions={example_loose.follow_all_instructions}\n"
                f"    follow_instruction_list={example_loose.follow_instruction_list}\n"
                f"  loose: sum/count={sum(follow_instruction_list_loose)}/{len(follow_instruction_list_loose)}\n"
                f"  details: {details[str(index)]}"
            )

        results = {
            'Prompt-level-strict-accuracy':
                prompt_strict_correct / prompt_strict_total * 100 if prompt_strict_total else 0,
            'Inst-level-strict-accuracy':
                sum(inst_strict_ratios) / len(inst_strict_ratios) * 100 if inst_strict_ratios else 0,
            'Prompt-level-loose-accuracy':
                prompt_loose_correct / prompt_loose_total * 100 if prompt_loose_total else 0,
            'Inst-level-loose-accuracy':
                sum(inst_loose_ratios) / len(inst_loose_ratios) * 100 if inst_loose_ratios else 0,
            'details': details,
        }

        # Phase-level final results: 1 line
        logger.info(
            f"[ifbench][score] Results: "
            f"PromptS={results['Prompt-level-strict-accuracy']:.2f}% "
            f"InstS={results['Inst-level-strict-accuracy']:.2f}% "
            f"PromptL={results['Prompt-level-loose-accuracy']:.2f}% "
            f"InstL={results['Inst-level-loose-accuracy']:.2f}% "
            f"Total={prompt_strict_total}"
        )
        logger.info(f"[ifbench][score] ========== Score End ==========")

        return results
