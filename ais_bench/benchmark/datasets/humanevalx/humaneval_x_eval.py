#  Copyright Huawei Technologies Co., Ltd. 2024. All rights reserved.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distsafe_openributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Counter, Tuple, Iterable, Dict

from dataclasses import dataclass
import regex
import numpy as np
from tqdm.auto import tqdm

from .file_utils import safe_open
from .humaneval_x_utils import read_dataset, IMPORT_HELPER, estimate_pass_at_k, check_correctness

LANGUAGE_NAME = {
    "cpp": "CPP",
    "go": "Go",
    "java": "Java",
    "js": "JavaScript",
    "python": "Python",
}

COMPLETION_ID_KEY = "completion_id"
TEST_CODE_KEY = "test_code"
TASK_ID_KEY = "task_id"


@dataclass
class EvalConfig:
    input_file: str = None
    tmp_dir: str = "./"
    n_workers: int = 32
    timeout: float = 500.0
    problem_file: str = "./benchmark/ais_bench//datasets/humanevalx/humanevalx_python.jsonl.gz"
    out_dir: str = None
    k: Tuple[int, int, int] = (1, 10, 100)
    test_groundtruth: bool = False
    example_test: bool = False
    go_dir: str = None


def process_humaneval_test(sample, problems, example_test=False):
    task_id = sample["task_id"]
    language = task_id.split("/")[0].lower()
    example_test_key = "example_test"

    prompt = sample["prompt"]
    if example_test and example_test_key in problems[task_id] and problems[task_id][example_test_key] != "":
        test = problems[task_id][example_test_key]
    else:
        test = problems[task_id]["test"]

    code = sample["generation"]

    # Pre-process for different languages
    if language == "python":
        code_ = []
        for line in code.split("\n"):
            if (len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t'):
                break
            code_.append(line)
        code = "\n".join(code_)
        test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
        test_string = test_setup + prompt + code + "\n" + test + "\n"
    elif language == "cpp":
        test_set_up = ""
        for s in IMPORT_HELPER["cpp"]:
            if s not in prompt:
                test_set_up += s + "\n"
        test_string = test_set_up + "\n" + prompt + code + "\n" + test
    elif language == "java":
        test_string = prompt + code + "\n" + test
    elif language == "js" or language == "javascript":
        test_string = prompt + code + "\n" + test
    elif language == "go":
        import_string = problems[task_id]["import"]
        prompt = prompt.replace(import_string, "")
        if example_test and example_test_key in problems[task_id]:
            test = problems[task_id][example_test_key]
        else:
            test = problems[task_id]["test"]
        test_setup = problems[task_id]["test_setup"]
        other_pkgs = []
        for pkg in IMPORT_HELPER["go"]:
            if pkg not in test_setup:
                p = pkg.split("/")[-1]
                if p + "." in code:
                    other_pkgs.append(f"\"{pkg}\"")
        if other_pkgs:
            import_other_pkgs = "import (\n" + "    ".join([p + "\n" for p in other_pkgs]) + ")"
            test_string = test_setup + "\n" + import_other_pkgs + "\n" + prompt + code + "\n" + test
        else:
            test_string = test_setup + "\n" + prompt + code + "\n" + test
    elif language == "rust":
        main = "\nfn main(){ \n } \n"
        declaration = problems[task_id]["declaration"]
        test_string = main + declaration + prompt + code + test

    return test_string


def stream_jsonl_all(filename: str) -> Iterable[Dict]:
    results = []
    fp = safe_open(filename, "r")
    for line in fp:
        if any(not x.isspace() for x in line):
            results.append(json.loads(line))
    fp.close()

    return results


def evaluate_functional_correctness(config: EvalConfig):
    completion_id_key = COMPLETION_ID_KEY
    test_code_key = TEST_CODE_KEY
    task_id_key = TASK_ID_KEY

    if config.example_test:
        pass

    problems = read_dataset(config.problem_file,
                            dataset_type="humaneval")
    sample_jsonl = stream_jsonl_all(config.input_file)

    if config.example_test:
        suffix = "_example_test.jsonl"
    else:
        suffix = "_results.jsonl"
    if config.out_dir is not None:
        if not os.path.exists(config.out_dir):
            os.makedirs(config.out_dir)
        out_file = os.path.join(config.out_dir, config.input_file.split('/')[-1].replace(".jsonl", suffix))
    else:
        out_file = os.path.join(config.input_file.replace(".jsonl", suffix))

    if "/codegeex/benchmark/humaneval-x/" in config.input_file:
        config.test_groundtruth = True

    if "-to-" in config.input_file:
        translation_mode = True
    else:
        translation_mode = False

    with ThreadPoolExecutor(max_workers=config.n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        if config.test_groundtruth:
            for sample in tqdm(problems.values()):
                task_id = sample[task_id_key]
                lang = task_id.split("/")[0].lower()
                if lang == "javascript":
                    lang = "js"
                tmp_dir_ = os.path.join(config.tmp_dir, lang, "evaluation")
                sample["generation"] = sample["canonical_solution"]
                sample[test_code_key] = process_humaneval_test(sample, problems, config.example_test)
                if sample[test_code_key] is None:
                    print(f"Skipping task {task_id} due to missing test code.")  # 跳过的任务
                    continue
                config_dict = {
                    "language_type": lang,
                    "timeout": config.timeout,
                    "tmp_dir": tmp_dir_,
                    "completion_id": completion_id[task_id],
                    "go_dir": config.go_dir
                }
                args = (task_id, sample, config_dict)
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1
        else:
            for sample in tqdm(sample_jsonl):
                task_id = sample[task_id_key]

                lang = task_id.split("/")[0].lower()
                if translation_mode:
                    task_id = sample[task_id_key].split("/")[-1]
                    lang = regex.findall("-to-.*-", config.input_file)[0].split("-to-")[-1].rstrip("-")
                    for language in LANGUAGE_NAME:
                        if language in lang:
                            lang = language
                            break
                    task_id = f"{LANGUAGE_NAME[lang]}/{task_id}"
                if lang == "javascript":
                    lang = "js"
                tmp_dir_ = os.path.join(config.tmp_dir, lang, "evaluation")
                sample[task_id_key] = task_id
                sample[test_code_key] = process_humaneval_test(sample, problems, config.example_test)
                if sample[test_code_key] is None:
                    continue
                if completion_id_key in sample:
                    completion_id_ = sample[completion_id_key]
                else:
                    completion_id_ = completion_id[task_id]
                config_dict = {
                    "language_type": lang,
                    "timeout": config.timeout,
                    "tmp_dir": tmp_dir_,
                    "completion_id": completion_id_,
                    "go_dir": config.go_dir
                }
                args = (task_id, sample, config_dict)
                future = executor.submit(check_correctness, *args)
                futures.append(future)
                completion_id[task_id] += 1
                n_samples += 1

        if len(completion_id) == len(problems):
            evaluate_pass_at_k = True
        else:
            evaluate_pass_at_k = False

        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result[task_id_key]].append((result[completion_id_key], result))

    # Calculate accuracy
    total, correct, details = [], [], []
    for result in results.values():
        for r in result:
            passed = r[1].get('passed', False)
            total.append(1)
            correct.append(1 if passed else 0)
            details.append({'task_id': r[0], 'passed': passed, 'result': r[1]})

    accuracy = 100 * sum(correct) / sum(total) if total else 0

    result = {'accuracy': accuracy, 'details': details}

    fp = safe_open(out_file, 'w')
    for res in results.values():
        for r in res:
            fp.write(json.dumps(r[1], indent=4) + "\n")
    fp.close()

    with safe_open(out_file, "ab") as fp:
        fp.write((json.dumps(result) + "\n").encode('utf-8'))

    return result  # Only return the required result

