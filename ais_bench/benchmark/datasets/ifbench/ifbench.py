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
    logger.debug(f"[ifbench][strict] ========== strict check start ==========")
    logger.debug(f"[ifbench][strict] inp.key: {inp.key}")
    logger.debug(f"[ifbench][strict] inp.instruction_id_list: {inp.instruction_id_list}")
    logger.debug(f"[ifbench][strict] inp.prompt: {inp.prompt}")
    logger.debug(f"[ifbench][strict] inp.kwargs: {inp.kwargs}")
    logger.debug(f"[ifbench][strict] response: {response}")

    instruction_list = inp.instruction_id_list
    is_following_list = []
    logger.debug(f"[ifbench][strict] instruction_list: {instruction_list}")

    for index, instruction_id in enumerate(instruction_list):
        logger.debug(f"[ifbench][strict] --- instruction[{index}] '{instruction_id}' ---")
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        logger.debug(f"[ifbench][strict] instruction_cls: {instruction_cls}")
        instruction = instruction_cls(instruction_id)
        logger.debug(f"[ifbench][strict] instruction: {instruction}")

        kwargs = inp.kwargs[index]
        logger.debug(f"[ifbench][strict] kwargs[{index}]: {kwargs}")

        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        logger.debug(f"[ifbench][strict] args: {args}")

        if args and 'prompt' in args:
            logger.debug(f"[ifbench][strict] building description with prompt: {inp.prompt}")
            instruction.build_description(prompt=inp.prompt)

        check_result = instruction.check_following(response)
        logger.debug(f"[ifbench][strict] response.strip(): '{response.strip()}'")
        logger.debug(f"[ifbench][strict] check_following result: {check_result}")

        if response.strip() and check_result:
            is_following_list.append(True)
        else:
            is_following_list.append(False)
        logger.debug(f"[ifbench][strict] instruction[{index}] passed: {is_following_list[-1]}")

    result = OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )
    logger.debug(f"[ifbench][strict] result.instruction_id_list: {result.instruction_id_list}")
    logger.debug(f"[ifbench][strict] result.prompt: {result.prompt}")
    logger.debug(f"[ifbench][strict] result.response: {result.response}")
    logger.debug(f"[ifbench][strict] result.follow_all_instructions: {result.follow_all_instructions}")
    logger.debug(f"[ifbench][strict] result.follow_instruction_list: {result.follow_instruction_list}")
    logger.debug(f"[ifbench][strict] ========== strict check end ==========")
    return result


def test_instruction_following_loose(inp: InputExample, response: str) -> OutputExample:
    """Tests response for an upper bound for following instructions (loose mode)."""
    logger.debug(f"[ifbench][loose] ========== loose check start ==========")
    logger.debug(f"[ifbench][loose] inp.key: {inp.key}")
    logger.debug(f"[ifbench][loose] inp.instruction_id_list: {inp.instruction_id_list}")
    logger.debug(f"[ifbench][loose] inp.prompt: {inp.prompt}")
    logger.debug(f"[ifbench][loose] inp.kwargs: {inp.kwargs}")
    logger.debug(f"[ifbench][loose] response: {response}")

    r = response.split('\n')
    response_remove_first = '\n'.join(r[1:]).strip()
    response_remove_last = '\n'.join(r[:-1]).strip()
    response_remove_both = '\n'.join(r[1:-1]).strip()
    revised_response = response.replace('*', '')
    revised_response_remove_first = response_remove_first.replace('*', '')
    revised_response_remove_last = response_remove_last.replace('*', '')
    revised_response_remove_both = response_remove_both.replace('*', '')

    logger.debug(f"[ifbench][loose] r (split lines): {r}")
    logger.debug(f"[ifbench][loose] response_remove_first: '{response_remove_first}'")
    logger.debug(f"[ifbench][loose] response_remove_last: '{response_remove_last}'")
    logger.debug(f"[ifbench][loose] response_remove_both: '{response_remove_both}'")
    logger.debug(f"[ifbench][loose] revised_response: '{revised_response}'")
    logger.debug(f"[ifbench][loose] revised_response_remove_first: '{revised_response_remove_first}'")
    logger.debug(f"[ifbench][loose] revised_response_remove_last: '{revised_response_remove_last}'")
    logger.debug(f"[ifbench][loose] revised_response_remove_both: '{revised_response_remove_both}'")

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
    logger.debug(f"[ifbench][loose] all_responses (8 variants):")
    for i, ar in enumerate(all_responses):
        logger.debug(f"[ifbench][loose]   all_responses[{i}]: '{ar}'")

    instruction_list = inp.instruction_id_list
    is_following_list = []
    logger.debug(f"[ifbench][loose] instruction_list: {instruction_list}")

    for index, instruction_id in enumerate(instruction_list):
        logger.debug(f"[ifbench][loose] --- instruction[{index}] '{instruction_id}' ---")
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        logger.debug(f"[ifbench][loose] instruction_cls: {instruction_cls}")
        instruction = instruction_cls(instruction_id)
        logger.debug(f"[ifbench][loose] instruction: {instruction}")

        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        logger.debug(f"[ifbench][loose] kwargs[{index}]: {inp.kwargs[index]}")
        logger.debug(f"[ifbench][loose] args: {args}")

        if args and 'prompt' in args:
            logger.debug(f"[ifbench][loose] building description with prompt: {inp.prompt}")
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for vi, ar in enumerate(all_responses):
            if ar.strip() and instruction.check_following(ar):
                logger.debug(f"[ifbench][loose] instruction[{index}] matched at all_responses[{vi}]: '{ar}'")
                is_following = True
                break

        is_following_list.append(is_following)
        logger.debug(f"[ifbench][loose] instruction[{index}] passed: {is_following}")

    result = OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )
    logger.debug(f"[ifbench][loose] result.instruction_id_list: {result.instruction_id_list}")
    logger.debug(f"[ifbench][loose] result.prompt: {result.prompt}")
    logger.debug(f"[ifbench][loose] result.response: {result.response}")
    logger.debug(f"[ifbench][loose] result.follow_all_instructions: {result.follow_all_instructions}")
    logger.debug(f"[ifbench][loose] result.follow_instruction_list: {result.follow_instruction_list}")
    logger.debug(f"[ifbench][loose] ========== loose check end ==========")
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
            logger.debug(f"[ifbench][load] Sample[{i}] prompt: {prompt}")
            logger.debug(f"[ifbench][load] Sample[{i}] item: {item}")
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
            logger.debug(f"[ifbench][score] --- Sample[{index}] ---")
            logger.debug(f"[ifbench][score] Sample[{index}] pred: {pred}")
            logger.debug(f"[ifbench][score] Sample[{index}] refer keys: {list(refer.keys())}")
            logger.debug(f"[ifbench][score] Sample[{index}] refer['key']: {refer.get('key')}")
            logger.debug(f"[ifbench][score] Sample[{index}] refer['instruction_id_list']: {refer.get('instruction_id_list')}")
            logger.debug(f"[ifbench][score] Sample[{index}] refer['prompt']: {refer.get('prompt')}")
            logger.debug(f"[ifbench][score] Sample[{index}] refer['kwargs']: {refer.get('kwargs')}")

            inp = InputExample(
                key=refer['key'],
                instruction_id_list=refer['instruction_id_list'],
                prompt=refer['prompt'],
                kwargs=[
                    {k: v for k, v in kwarg.items() if v is not None}
                    for kwarg in refer.get('kwargs', [])
                ])
            logger.debug(f"[ifbench][score] Sample[{index}] InputExample.key: {inp.key}")
            logger.debug(f"[ifbench][score] Sample[{index}] InputExample.instruction_id_list: {inp.instruction_id_list}")
            logger.debug(f"[ifbench][score] Sample[{index}] InputExample.prompt: {inp.prompt}")
            logger.debug(f"[ifbench][score] Sample[{index}] InputExample.kwargs (cleaned): {inp.kwargs}")

            prompt_text = refer.get('prompt', '')
            pred_text = pred or ''
            logger.debug(f"[ifbench][score] Sample[{index}] prompt_text: {prompt_text}")
            logger.debug(f"[ifbench][score] Sample[{index}] pred_text: {pred_text}")

            # strict evaluation
            example_strict = test_instruction_following_strict(inp, pred_text)
            logger.debug(f"[ifbench][score] Sample[{index}] strict OutputExample.instruction_id_list: {example_strict.instruction_id_list}")
            logger.debug(f"[ifbench][score] Sample[{index}] strict OutputExample.prompt: {example_strict.prompt}")
            logger.debug(f"[ifbench][score] Sample[{index}] strict OutputExample.response: {example_strict.response}")
            logger.debug(f"[ifbench][score] Sample[{index}] strict OutputExample.follow_all_instructions: {example_strict.follow_all_instructions}")
            logger.debug(f"[ifbench][score] Sample[{index}] strict OutputExample.follow_instruction_list: {example_strict.follow_instruction_list}")

            follow_instruction_list_strict = example_strict.follow_instruction_list
            instruction_id_list_strict = example_strict.instruction_id_list
            prompt_strict_total += 1
            is_strict_correct = all(follow_instruction_list_strict)
            prompt_strict_correct += is_strict_correct
            inst_strict_ratios.append(
                sum(follow_instruction_list_strict) / len(instruction_id_list_strict)
                if instruction_id_list_strict else 0.0
            )

            logger.debug(f"[ifbench][score] Sample[{index}] follow_instruction_list_strict: {follow_instruction_list_strict}")
            logger.debug(f"[ifbench][score] Sample[{index}] instruction_id_list_strict: {instruction_id_list_strict}")
            logger.debug(f"[ifbench][score] Sample[{index}] is_strict_correct: {is_strict_correct}")
            logger.debug(f"[ifbench][score] Sample[{index}] strict sum/count: {sum(follow_instruction_list_strict)}/{len(follow_instruction_list_strict)}")

            # loose evaluation
            example_loose = test_instruction_following_loose(inp, pred_text)
            logger.debug(f"[ifbench][score] Sample[{index}] loose OutputExample.instruction_id_list: {example_loose.instruction_id_list}")
            logger.debug(f"[ifbench][score] Sample[{index}] loose OutputExample.prompt: {example_loose.prompt}")
            logger.debug(f"[ifbench][score] Sample[{index}] loose OutputExample.response: {example_loose.response}")
            logger.debug(f"[ifbench][score] Sample[{index}] loose OutputExample.follow_all_instructions: {example_loose.follow_all_instructions}")
            logger.debug(f"[ifbench][score] Sample[{index}] loose OutputExample.follow_instruction_list: {example_loose.follow_instruction_list}")

            follow_instruction_list_loose = example_loose.follow_instruction_list
            instruction_id_list_loose = example_loose.instruction_id_list
            prompt_loose_total += 1
            is_loose_correct = all(follow_instruction_list_loose)
            prompt_loose_correct += is_loose_correct
            inst_loose_ratios.append(
                sum(follow_instruction_list_loose) / len(instruction_id_list_loose)
                if instruction_id_list_loose else 0.0
            )

            logger.debug(f"[ifbench][score] Sample[{index}] follow_instruction_list_loose: {follow_instruction_list_loose}")
            logger.debug(f"[ifbench][score] Sample[{index}] instruction_id_list_loose: {instruction_id_list_loose}")
            logger.debug(f"[ifbench][score] Sample[{index}] is_loose_correct: {is_loose_correct}")
            logger.debug(f"[ifbench][score] Sample[{index}] loose sum/count: {sum(follow_instruction_list_loose)}/{len(follow_instruction_list_loose)}")

            if is_strict_correct:
                grade = 'strict'
            elif is_loose_correct:
                grade = 'loose'
            else:
                grade = 'none'
            logger.debug(f"[ifbench][score] Sample[{index}] grade: {grade}")

            details[str(index)] = {
                'prompt': origin_prompt[index] if origin_prompt else refer.get('prompt', ''),
                'pred': pred,
                'refer': refer,
                'is_strict_correct': is_strict_correct,
                'is_loose_correct': is_loose_correct,
                'is_correct': is_strict_correct,
                'grade': grade,
            }
            logger.debug(f"[ifbench][score] Sample[{index}] details: {details[str(index)]}")

            # Phase-level progress: 1-line summary per sample
            logger.info(
                f"[ifbench][score] Sample[{index}] grade={grade} "
                f"strict={is_strict_correct} loose={is_loose_correct} "
                f"accum: prompt_s={prompt_strict_correct}/{prompt_strict_total} "
                f"inst_s={inst_strict_ratios[-1]:.2f} "
                f"prompt_l={prompt_loose_correct}/{prompt_loose_total} "
                f"inst_l={inst_loose_ratios[-1]:.2f}"
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
