import ast
import json
import subprocess
import sys
import os.path as osp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from ais_bench.benchmark.openicl.icl_evaluator import BaseEvaluator
from ais_bench.benchmark.registry import ICL_EVALUATORS
from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.error_codes import DSET_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchDataContentError, ParameterValueError

from ais_bench.benchmark.datasets.livecodebench.execute_utils import BASE_IMPORTS, codeexecute_check_correctness
from ais_bench.benchmark.datasets.livecodebench.extract_utils import (extract_code_execution, extract_code_generation,
                            extract_code_generation_v2,
                            extract_test_output_code)
from ais_bench.benchmark.datasets.livecodebench.livecodebench import LCBCodeGenerationDataset
from ais_bench.benchmark.datasets.livecodebench.pass_k_utils import compute_metrics_from_results


logger = AISLogger()

def codegen_check_correctness(sample, generation, timeout, debug=True):
    per_case = len(json.loads(sample['input_output'])['inputs'])
    total_timeout = (timeout + 1) * per_case + 5

    payload = {
        'sample': sample,
        'generation': generation,
        'debug': debug,
        'timeout': timeout
    }

    try:
        # use text=True to get str, capture_output=True capture stdout/stderr
        current_dir = osp.dirname(osp.abspath(__file__))
        runner_path = osp.join(current_dir, "test_runner.py")
        proc = subprocess.run(
            [sys.executable, runner_path],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=total_timeout
        )
    except subprocess.TimeoutExpired as e:
        logger.info(f'Global timeout (subprocess.TimeoutExpired): {e}')
        # return all failed placeholder results
        return ([-1] * per_case), {}
    except Exception as e:
        logger.info(f'Failed to spawn test_runner subprocess: {e}')
        return ([-1] * per_case), {}

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    if proc.returncode != 0:
        logger.warning('test_runner exited with non-zero code %s', proc.returncode)
        logger.debug('test_runner stdout: %s', stdout)
        logger.debug('test_runner stderr: %s', stderr)
        # try to parse JSON from stdout (may contain error field)
    try:
        if not stdout:
            raise ValueError("empty stdout from test_runner")
        data = json.loads(stdout)
        # expected structure: {'res': ..., 'meta': ..., 'error': ...}
        if data.get('error'):
            logger.warning('test_runner returned error: %s', data.get('error'))
        res = data.get('res')
        meta = data.get('meta')
        if res is None:
            raise ValueError("result 'res' missing or None")
        return res, (meta or {})
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # parse failed or data illegal
        logger.debug(f'Failed to parse test_runner output: {e}')
        logger.info('---- test_runner stdout ----\n%s', stdout)
        return ([-1] * per_case), {}


def evaluate_generations_by_problem(problem_generations: list, sample: list,
                                    debug: bool, timeout: int):
    """Evaluate each problem.

    Args:
        problem_generations:
        sample:
        debug:
        timeout
    """
    # problem_generations: list[str] = args[0]
    # sample = args[1]
    # debug: bool = args[2]
    # timeout: int = args[3]
    logger = AISLogger()

    res = []
    metadata = []
    for o_idx, o in enumerate(problem_generations):
        curr_res = [-2]
        try:
            curr_res, curr_metadata = codegen_check_correctness(
                sample, o, timeout=timeout, debug=debug)
            if debug:
                logger.info(f'\nSuccessful compilation of task {o_idx}!')
            fixed = []
            for e in curr_res:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_res = fixed
            if not np.all(curr_res):
                if debug:
                    logger.info(
                        f'Results were not True for all test cases'  # noqa: F541, E501
                        f' {curr_res=}\n')
        except Exception as e:
            logger.debug(f'Compilation failed, test framework exception: {repr(e)}')
            # break
            curr_metadata = {}
        finally:
            if not isinstance(curr_res, list):
                raise AISBenchDataContentError(
                    DSET_CODES.INVALID_DATA_TYPE,
                    f"curr_res must be a list, got {type(curr_res)}: {repr(curr_res)}"
                )
            if not isinstance(curr_metadata, dict):
                raise AISBenchDataContentError(
                    DSET_CODES.INVALID_DATA_TYPE,
                    f"curr_metadata must be a dict, got {type(curr_metadata)}: {repr(curr_metadata)}"
                )
            res.append(curr_res)
            metadata.append(curr_metadata)
    if debug:
        for i, r in enumerate(problem_generations):
            logger.info(f'Sample\n{r}\nResult\n{res[i]}')
            logger.info('*' * 30 + '\n\n')
    return res, metadata


def evaluate_generations(
    samples_list: list,
    generations_list: list[list[str]],
    debug: bool = False,
    num_process_evaluate: int = 16,
    timeout=6,
):
    """We take the list of code generations and try to compile them and the run
    their corresponding unit tests which are retrieved from the APPS dataset.

    Args:
        generations: list of code generations (same order as samples in APPS
            dataset)
        level: difficulty level used in the generation, can be "all",
            "introductory", "interview" or "competition"

    Returns:
        results: dictionary of results, key is the problem index, value is
            a list of results for each generation
        [-2] = compile error, [-1] = runtime error [False] = failed test
            case [True] = passed test case
    """

    # generations are code generations in the same order of the dataset

    inputs = [[(generations_list[index], samples_list[index], debug, timeout),
               index] for index in range(len(generations_list))]

    with tqdm(total=len(inputs)) as pbar:
        with ProcessPoolExecutor(
                max_workers=1 if debug else num_process_evaluate) as executor:
            futures = {
                executor.submit(evaluate_generations_by_problem,
                                problem_generations, sample, debug, timeout):
                index
                for (problem_generations, sample, debug,
                     timeout), index in inputs
            }

            results = {}
            metadata = {}
            for future in as_completed(futures):
                index = futures[future]
                results[index], metadata[index] = future.result()
                pbar.update(1)

    if len(results) != len(inputs):
        raise AISBenchDataContentError(
            DSET_CODES.PREDICTION_LENGTH_MISMATCH,
            f'Results and inputs have different lengths: results={len(results)}, inputs={len(inputs)}'
        )
    # results = {i: r for r, (_, i) in zip(results, inputs)}

    return results, metadata


def codegen_metrics(
    samples_list,
    generations_list,
    k_list=[1, 5, 10, 20, 40, 50, 75, 100, 125, 150, 200, 500, 1000],
    num_process_evaluate=16,
    timeout=6,
    debug=False,
):
    samples_linear = []
    generations_linear = []
    remap_index = []
    results = defaultdict(list)
    metadatas = defaultdict(list)
    for idx, (sample,
              generation_list) in enumerate(zip(samples_list,
                                                generations_list)):
        if not isinstance(generation_list, list):
            raise AISBenchDataContentError(
                DSET_CODES.INVALID_DATA_TYPE,
                f"generation_list must be a list, got {type(generation_list)}. First element in generations_list: {repr(generations_list[0]) if generations_list else 'N/A'}"
            )
        for generation in generation_list:
            if not isinstance(generation, str):
                raise AISBenchDataContentError(
                    DSET_CODES.INVALID_DATA_TYPE,
                    f"generation must be a str, got {type(generation)}: {repr(generation)}"
                )
            samples_linear.append(sample)
            generations_linear.append([generation])
            remap_index.append(idx)

    logger.info(f'LCBCodeGeneration: Evaluating {len(samples_linear)}...')

    results_linear, metadatas_linear = evaluate_generations(
        samples_linear,
        generations_linear,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout,
    )

    for idx, sub_results in sorted(results_linear.items(), key=lambda x: x[0]):
        results[remap_index[idx]].append(sub_results[0])

    for idx, sub_metadatas in sorted(metadatas_linear.items(),
                                     key=lambda x: x[0]):
        metadatas[remap_index[idx]].append(sub_metadatas[0])

    metrics = compute_metrics_from_results(results, k_list=k_list)

    final_metadata = []
    for key in sorted(list(metadatas.keys())):
        final_metadata.append(metadatas[key])
    for i in range(len(final_metadata)):
        if type(final_metadata[i]) is not list:
            final_metadata[i] = [json.dumps(final_metadata[i])]
        else:
            final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]

        if len(final_metadata[i]) != len(generations_list[0]):
            raise AISBenchDataContentError(
                DSET_CODES.PREDICTION_LENGTH_MISMATCH,
                f'Metadata length mismatch: {len(final_metadata[i])} != {len(generations_list[0])}'
            )

    return [metrics, results, final_metadata]


@ICL_EVALUATORS.register_module()
class LCBCodeGenerationEvaluator(BaseEvaluator):

    def __init__(self,
                 num_process_evaluate,
                 timeout=6,
                 release_version='release_v1',
                 extractor_version='v1'):
        super().__init__()
        self.num_process_evaluate = num_process_evaluate
        self.timeout = timeout
        self.extractor_version = extractor_version

    def score(self, predictions, references, test_set):
        if self.extractor_version == 'v1':
            predictions = [[extract_code_generation(item)]
                           for item in predictions]
        elif self.extractor_version == 'v2':
            predictions = [[extract_code_generation_v2(item)]
                           for item in predictions]

        evaluation_samples = {
            item['question_id']: item['evaluation_sample']
            for item in test_set
        }

        references = [evaluation_samples[item] for item in references]

        references = [{'input_output': item} for item in references]

        BaseEvaluator.is_num_equal(predictions, references)

        extracted_predictions = {}
        for idx, content in enumerate(predictions):
            extracted_predictions[idx] = content

        metrics, eval_results, final_metadata = codegen_metrics(
            references,
            predictions,
            k_list=[1],
            num_process_evaluate=self.num_process_evaluate,
            timeout=self.timeout,
        )

        def is_equal(pred, refer):
            try:
                if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
                    return True
            except (ValueError, TypeError) as e:
                logger.debug(f"Failed to compare pred={pred} and refer={refer}: {e}")
                pass
            return False
        details = []
        pass_at_one_detail_dict = metrics.get('detail', {}).get('pass@1', {})
        for idx in range(len(predictions)):
            detail = {'correct': False}
            if is_equal(pass_at_one_detail_dict.get(idx, 0.0), 100.0):
                detail['correct'] = True
            details.append(detail)

        results = {
            'extracted_predictions': extracted_predictions,
            'eval_results': eval_results,
            'details': details
        }
        results.update(metrics)

        return results


def evaluate_score(args) -> list[bool]:
    gs, (c, i, o) = args

    execution_results = []
    for g in gs:
        if i in g:
            pass
        else:
            code_to_execute = f'{BASE_IMPORTS}\n{c}\nassert {o} == {g}'
            execution_results.append(
                codeexecute_check_correctness(code_to_execute, 3))
    if len(execution_results) == 0:
        execution_results = [False] * len(gs)
    return execution_results


def code_execution_metrics(
    samples,
    generations,
):

    def pass_at_k(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    # execute the code
    references = [(doc['code'], doc['input'], doc['output'])
                  for doc in samples]
    with ProcessPoolExecutor() as executor:
        args_list = zip(generations, references)
        results = executor.map(evaluate_score, args_list)
    all_results = list(results)

    # serial version
    # all_results = []
    # for i in range(len(generations)):
    #     generation = generations[i]
    #     result = evaluate_score([generation, references[i]])
    #     all_results.append(result)

    # compute pass@1
    pass_at_1s = []
    for execution_result in all_results:
        c, n = execution_result.count(True), len(execution_result)
        pass_at_1s.append(pass_at_k(n, c, 1))
    metrics = {'pass@1': sum(pass_at_1s) / len(pass_at_1s) * 100}

    results = {}
    for i, r in enumerate(all_results):
        r_new = []
        for _r in r:
            r_new.append([_r])
        results[i] = r_new
    return [metrics, results]


@ICL_EVALUATORS.register_module()
class LCBCodeExecutionEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()
        # self.num_process_evaluate = num_process_evaluate
        # self.timeout = timeout

    def score(self, predictions, references):
        predictions = [[extract_code_execution(item)] for item in predictions]
        references = [json.loads(item) for item in references]
        metrics, results = code_execution_metrics(references, predictions)
        return metrics


def parse_assert_statement(statement):
    """Parse a Python assert statement and extract the expected output from the
    right side of the '==' operator as a string.

    :param statement: A string containing the assert statement.
    :return: The expected output from the assert statement as a string.
    """
    try:
        parsed = ast.parse(statement, mode='exec')
    except SyntaxError:
        return 'Invalid syntax'

    if len(parsed.body) == 0:
        return 'Empty statement'

    if not isinstance(parsed.body[0], ast.Assert):
        return 'Not an assert statement'

    comparison = parsed.body[0].test

    if not isinstance(comparison, ast.Compare) or not isinstance(
            comparison.ops[0], ast.Eq):
        return 'Not an equality assertion'

    # Extract and return the right side of the '==' operator as a string
    return ast.get_source_segment(statement, comparison.comparators[0])


def check_testcase_output(testcase_str, expected_output):

    if len(testcase_str.splitlines()) > 1:
        for line in testcase_str.splitlines():
            if line.startswith('#'):
                continue
            if 'assert' in line:
                testcase_str = line
                break

    testcase_str = testcase_str.strip()

    if 'assert' in testcase_str:
        testcase_output_str = str(parse_assert_statement(testcase_str))

    else:
        testcase_output_str = testcase_str

    global_result = None

    try:
        testcase_output_eval = eval(testcase_output_str)
    except Exception as e:
        print(e)
        global_result = False
        # print("Failed to eval testcase output", testcase_output_str)

    try:
        expected_output_eval = json.loads(expected_output)
    except Exception as e:
        print(e)
        global_result = False
        print('Failed to eval expected testcase output', expected_output)

    if global_result is None:
        global_result = testcase_output_eval == expected_output_eval

    return global_result


def test_output_metrics(
    samples,
    generations,
    k_list=[1, 5],
):
    num_samples = len(samples)
    results = []
    for idx in tqdm(list(range(num_samples))):
        idx_results = []
        sample = samples[idx]
        extracted_generation_list = generations[idx]
        for extracted_generation in extracted_generation_list:
            global_result = check_testcase_output(extracted_generation,
                                                  sample['output'])
            idx_results.append([global_result])
        results.append(idx_results)

    results = {
        result_idx: results[result_idx]
        for result_idx in range(len(results))
    }

    metrics = compute_metrics_from_results(results, k_list=k_list)

    return [metrics, results]


@ICL_EVALUATORS.register_module()
class LCBTestOutputEvaluator(BaseEvaluator):

    def __init__(self):
        super().__init__()

    def score(self, predictions, references):

        predictions = [[extract_test_output_code(item)]
                       for item in predictions]
        references = [json.loads(item) for item in references]
        metrics, results = test_output_metrics(references,
                                               predictions,
                                               k_list=[1])
        return metrics
