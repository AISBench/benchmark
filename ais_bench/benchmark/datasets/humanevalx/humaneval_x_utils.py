#  Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import contextlib
import faulthandler
import json
import io
import itertools
import multiprocessing
import os
import sys
import shutil
import platform
import random
import signal
import subprocess
import tempfile
import builtins
import resource
import gzip
import json
from typing import Iterable, Dict
import importlib.util
from typing import Iterable, Dict, Union, List, Optional
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers.generation.stopping_criteria import StoppingCriteria

from .file_utils import safe_open

# from atb_llm.utils.log.logging import logger

LANGUAGE_TAG = {
    "c": "// language: C",
    "c++": "// language: C++",
    "cpp": "// language: C++",
    "c#": "// language: C#",
    "csharp": "// language: C#",
    "css": "/* language: CSS */",
    "cuda": "// language: Cuda",
    "dart": "// language: Dart",
    "lua": "// language: Lua",
    "objectivec": "// language: Objective-C",
    "objective-c": "// language: Objective-C",
    "objective-c++": "// language: Objective-C++",
    "python": "# language: Python",
    "perl": "# language: Perl",
    "prolog": "% language: Prolog",
    "swift": "// language: swift",
    "lisp": "; language: Lisp",
    "java": "// language: Java",
    "scala": "// language: Scala",
    "tex": "% language: TeX",
    "vue": "<!--language: Vue-->",
    "markdown": "<!--language: Markdown-->",
    "html": "<!--language: HTML-->",
    "php": "// language: PHP",
    "js": "// language: JavaScript",
    "javascript": "// language: JavaScript",
    "typescript": "// language: TypeScript",
    "go": "// language: Go",
    "shell": "# language: Shell",
    "rust": "// language: Rust",
    "sql": "-- language: SQL",
    "kotlin": "// language: Kotlin",
    "vb": "' language: Visual Basic",
    "ruby": "# language: Ruby",
    "pascal": "// language: Pascal",
    "r": "# language: R",
    "fortran": "!language: Fortran",
    "lean": "-- language: Lean",
    "matlab": "% language: Matlab",
    "delphi": "{language: Delphi}",
    "scheme": "; language: Scheme",
    "basic": "' language: Basic",
    "assembly": "; language: Assembly",
    "groovy": "// language: Groovy",
    "abap": "* language: Abap",
    "gdscript": "# language: GDScript",
    "haskell": "-- language: Haskell",
    "julia": "# language: Julia",
    "elixir": "# language: Elixir",
    "excel": "' language: Excel",
    "clojure": "; language: Clojure",
    "actionscript": "// language: ActionScript",
    "solidity": "// language: Solidity",
    "powershell": "# language: PowerShell",
    "erlang": "% language: Erlang",
    "cobol": "// language: Cobol",
}

IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "go": [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp": [
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<math.h>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
    ],
}


class StoppingCriteriaWithHumanEvalX(StoppingCriteria):
    def __init__(self, lang: str = None, original_input_len: int = None, tokenizer=None):
        self.lang = lang
        self.original_input_len = original_input_len
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        output_ids = input_ids[0]
        if output_ids[-1].detach().cpu().numpy() in [self.tokenizer.eos_token_id]:
            return True
        text = self.tokenizer.decode(output_ids[self.original_input_len:], skip_special_tokens=False)
        return is_code_generation_finished(
            text,
            language_type=self.lang,
            dataset="humaneval",
        )


class HumanEvalXDataset(Dataset):
    def __init__(self, task_dict):
        self.task_dict = task_dict
        self.keys = list(task_dict.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        return self.task_dict[self.keys[index]]


def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        if len(num_samples) != len(num_correct):
            raise RuntimeError("The lengths of num_samples and num_correct do not match.")
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary, assuming the file is gzipped if needed.
    """

    if filename.endswith('.gz'):
        with gzip.open(filename, 'rt', encoding='utf-8') as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)
    else:
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def read_dataset(
        data_file: str = None,
        dataset_type: str = "humaneval",
        num_shot=None,
) -> Dict:
    if num_shot is not None:
        pass
    if "humaneval" in dataset_type.lower():
        if data_file is None:
            current_path = os.path.dirname(os.path.abspath(__file__))
            data_file = os.path.join(current_path, "..", "humaneval-x", "python", "data", "humaneval_python.jsonl.gz")
        dataset = {task["task_id"]: task for task in stream_jsonl(data_file)}
    else:
        raise f"Dataset: {dataset_type} not supported."

    return dataset


def process_extra_prompt(prompt: str, language_type: str = None) -> str:
    """
    Processes the extra prompt.
    """
    language = language_type.lower()
    if language in LANGUAGE_TAG:
        extra_prompt = LANGUAGE_TAG[language] + "\n"
    else:
        extra_prompt = ""

    return extra_prompt + prompt


def is_code_generation_finished(
        code: str,
        language_type: str = None,
        dataset: str = None,
):
    """
    Checks whether the generated code is finished.
    """
    if language_type is None or dataset is None:
        return False

    if "humaneval" in dataset.lower():
        if language_type.lower() == "python":
            for line in code.split("\n"):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    return True
            end_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
            for w in end_words:
                if w in code:
                    return True
        elif language_type.lower() == "java":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "go":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "js":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "cpp":
            if code.count("{") + 1 == code.count("}"):
                return True

    return False


def cleanup_code(
        code: str,
        language_type: str = None,
        dataset: str = None,
):
    """
    Cleans up the generated code.
    """
    if language_type is None or dataset is None:
        return code

    if "humaneval" in dataset.lower():
        if language_type.lower() == "python":
            end_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint", "\nassert"]
            for w in end_words:
                if w in code:
                    code = code[:code.rfind(w)]
        elif language_type.lower() == "java":
            main_pos = code.find("public static void main")
            if main_pos != -1:
                code = code[:main_pos] + '}'
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
            if code.count('{') + 1 == code.count('}'):
                code += "\n}"
        elif language_type.lower() == "go":
            end_words = ["\n//", "\nfunc main("]
            for w in end_words:
                if w in code:
                    code = code[:code.rfind(w)]
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
        elif language_type.lower() == "cpp":
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
        elif language_type.lower() == "js":
            if '}' in code:
                code = code[:code.rfind('}')] + '}'

    return code


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class RedirectStdin:
    def __init__(self, new_target):
        self._new_target = new_target
        self._old_target = None

    def __enter__(self):
        self._old_target = sys.stdin
        sys.stdin = self._new_target
        return self._new_target

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdin = self._old_target


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


@contextlib.contextmanager
def create_tempdir():
    original_unlink = os.unlink
    original_remove = os.remove

    def get_safe_unlink():
        return getattr(shutil, '_orig_unlink', original_unlink)

    def get_safe_remove():
        return getattr(shutil, '_orig_remove', original_remove)

    try:
        os.unlink = get_safe_unlink()
        os.remove = get_safe_remove()
        with tempfile.TemporaryDirectory() as dirname:
            yield dirname
    finally:
        os.unlink = original_unlink
        os.remove = original_remove


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with RedirectStdin(stream):
                yield


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    original_unlink = os.unlink
    original_remove = os.remove
    try:
        if maximum_memory_bytes is not None:
            resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
            resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
            if not platform.uname().system == 'Darwin':
                resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))
    except Exception as e:
        pass

        faulthandler.disable()

        builtins.exit = None
        builtins.quit = None

        os.environ['OMP_NUM_THREADS'] = '1'

        os.kill = None
        os.system = None
        os.putenv = None
        os.remove = None
        os.removedirs = None
        os.rmdir = None
        os.fchdir = None
        os.setuid = None
        os.fork = None
        os.forkpty = None
        os.killpg = None
        os.rename = None
        os.renames = None
        os.truncate = None
        os.replace = None
        os.unlink = None
        os.fchmod = None
        os.fchown = None
        os.chmod = None
        os.chown = None
        os.chroot = None
        os.fchdir = None
        os.lchflags = None
        os.lchmod = None
        os.lchown = None
        os.getcwd = None
        os.chdir = None

        shutil.move = None
        shutil.chown = None

        subprocess.Popen = None  # type: ignore

        __builtins__['help'] = None

        sys.modules['ipdb'] = None
        sys.modules['joblib'] = None
        sys.modules['resource'] = None
        sys.modules['psutil'] = None

    finally:
        os.unlink = original_unlink
        os.remove = original_remove


def check_correctness(
        task_id: str,
        sample: dict,
        config: dict,
) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem.
    """

    language_type = config.get('language_type', 'python')
    timeout = config.get('timeout', 300.0)
    tmp_dir = config.get('tmp_dir', None)
    completion_id = config.get('completion_id', None)
    go_dir = config.get('go_dir', None)

    finish = "finish"
    file = "file"
    output = "output"
    test_code = "test_code"
    passed = "passed"
    timed_out = "timed out"
    tmp = "tmp"
    test_cpp = "test.cpp"

    def unsafe_execute(tmp_dir):
        random_id = random.uniform(1, 1000)
        if "python" in language_type.lower():
            with create_tempdir() as tmp_dir:

                # These system calls are needed when cleaning up tempdir.
                rmtree = shutil.rmtree
                rmdir = os.rmdir
                change_dir = os.chdir

                # Disable functionalities that can make destructive changes to the test.
                reliability_guard()
                code_file = os.path.join(tmp_dir, f"test_code_{random_id}.py")

                with open(code_file, 'w') as f:
                    f.write(sample[test_code])

                exec_globals = {}
                try:
                    with swallow_io():
                        with time_limit(timeout):
                            # WARNING
                            # This program exists to execute untrusted model-generated code. Although
                            # it is highly unlikely that model-generated code will do something overtly
                            # malicious in response to this test suite, model-generated code may act
                            # destructively due to a lack of model capability or alignment.
                            # Users are strongly encouraged to sandbox this evaluation suite so that it
                            # does not perform destructive actions on their host or network.
                            spec = importlib.util.spec_from_file_location("test_code_module", code_file)
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                        result.append(passed)
                except TimeoutException:
                    result.append(timed_out)
                except AssertionError:
                    result.append("failed: AssertionError")
                except BaseException as e:
                    result.append(f"failed: {e}")

                # Needed for cleaning up.
                shutil.rmtree = rmtree
                os.rmdir = rmdir
                os.chdir = change_dir

        elif "go" in language_type.lower():
            if tmp_dir is None:
                raise RuntimeError("Go should be evaluated in a dir where necessary module files installed.")

            if tmp not in tmp_dir:
                tmp_dir = os.path.join(tmp_dir, tmp)
            tmp_dir = os.path.join(tmp_dir, f"{task_id.replace('/', '-')}-{random_id}")
            tmp_dir = os.path.abspath(tmp_dir)
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)

            os.chdir(tmp_dir)
            files_in_tmp_dir = os.listdir(tmp_dir)
            shutil.copytree(go_dir, tmp_dir, dirs_exist_ok=True)
            files_in_tmp_dir = os.listdir(tmp_dir)
            safe_open("main_test.go", 'w').write(sample[test_code])
            exec_result = None
            try:
                # '/usr/local/go/bin/go', 'test', '-timeout=6s', 'main_test.go'
                go_executable = shutil.which("go")
                if go_executable is None:
                    raise FileNotFoundError("Go executable not found in the system PATH.")
                with time_limit(timeout):
                    # WARNING
                    # This program exists to execute untrusted model-generated code. Although
                    # it is highly unlikely that model-generated code will do something overtly
                    # malicious in response to this test suite, model-generated code may act
                    # destructively due to a lack of model capability or alignment.
                    # Users are strongly encouraged to sandbox this evaluation suite so that it
                    # does not perform destructive actions on their host or network.
                    exec_result = subprocess.run([go_executable, "test", f"-timeout={timeout}s", "main_test.go"],
                                                 timeout=timeout, capture_output=True)

                if exec_result.returncode == 0:
                    result.append(passed)
                else:
                    if exec_result.stderr:
                        try:
                            err = exec_result.stderr.decode()
                        except Exception:
                            err = exec_result.stderr
                    else:
                        try:
                            err = exec_result.stdout.decode()
                        except Exception:
                            err = exec_result.stdout
                    result.append(f"failed: {err}")

            except TimeoutException:
                result.append(timed_out)

            tmp_dir = os.path.abspath(tmp_dir)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

        elif "js" in language_type.lower():

            if tmp not in tmp_dir:
                tmp_dir = os.path.join(tmp_dir, tmp)
            tmp_dir = os.path.join(tmp_dir, f"{task_id.replace('/', '-')}-{random_id}")
            tmp_dir = os.path.abspath(tmp_dir)

            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)

            os.chdir(tmp_dir)
            safe_open("test.js", 'w').write(sample[test_code])
            exec_result = None
            try:
                node_executable = shutil.which("node")
                if node_executable is None:
                    raise FileNotFoundError("Node executable not found in the system PATH.")
                with time_limit(timeout):
                    # WARNING
                    # This program exists to execute untrusted model-generated code. Although
                    # it is highly unlikely that model-generated code will do something overtly
                    # malicious in response to this test suite, model-generated code may act
                    # destructively due to a lack of model capability or alignment.
                    # Users are strongly encouraged to sandbox this evaluation suite so that it
                    # does not perform destructive actions on their host or network.
                    exec_result = subprocess.run([node_executable, "test.js"], timeout=timeout, capture_output=True)

                if exec_result.stderr.decode():
                    err = exec_result.stderr.decode()
                    result.append(f"failed: {err}")
                elif exec_result.stdout.decode():
                    err = exec_result.stdout.decode()
                    result.append(f"failed: {err}")
                else:
                    result.append(passed)

            except TimeoutException:
                result.append(timed_out)

            tmp_dir = os.path.abspath(tmp_dir)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

        elif "cpp" in language_type.lower():

            if tmp not in tmp_dir:
                tmp_dir = os.path.join(tmp_dir, tmp)
            tmp_dir = os.path.join(tmp_dir, f"{task_id.replace('/', '-')}-{random_id}")
            tmp_dir = os.path.abspath(tmp_dir)
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)

            os.chdir(tmp_dir)
            safe_open(test_cpp, 'w').write(sample[test_code])
            if "162" in task_id:
                gpp_executable = shutil.which("g++")
                compilation_result = subprocess.run([gpp_executable, "-std=c++11", test_cpp, "-lcrypto", "-lssl"],
                                                    timeout=10,
                                                    capture_output=True)
                compilation_result = subprocess.run([gpp_executable, "-std=c++11", test_cpp, "-lcrypto", "-lssl"],
                                                    timeout=10,
                                                    capture_output=True)
            else:
                gpp_executable = shutil.which("g++")
                compilation_result = subprocess.run([gpp_executable, "-std=c++11", test_cpp], timeout=timeout,
                                                    capture_output=True)
            if compilation_result.returncode != 0:
                if compilation_result.stderr:
                    err = compilation_result.stderr.decode()
                else:
                    err = compilation_result.stdout.decode()
                result.append(f"failed: compilation error: {err}")
            else:
                exec_result = None
                try:
                    with time_limit(timeout):
                        # WARNING
                        # This program exists to execute untrusted model-generated code. Although
                        # it is highly unlikely that model-generated code will do something overtly
                        # malicious in response to this test suite, model-generated code may act
                        # destructively due to a lack of model capability or alignment.
                        # Users are strongly encouraged to sandbox this evaluation suite so that it
                        # does not perform destructive actions on their host or network.
                        exec_result = subprocess.run(["./a.out"], timeout=timeout, capture_output=True)

                    if exec_result.returncode == 0:
                        result.append(passed)
                    else:
                        if exec_result.stderr:
                            try:
                                err = exec_result.stderr.decode()
                            except Exception:
                                err = exec_result.stderr
                        else:
                            try:
                                err = exec_result.stdout.decode()
                            except Exception:
                                err = exec_result.stdout
                        result.append(f"failed: {err}")
                except TimeoutException:
                    result.append(timed_out)

            tmp_dir = os.path.abspath(tmp_dir)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

        elif "java" in language_type.lower():
            if tmp_dir is None:
                raise RuntimeError("Java should be evaluated in a temporary dir.")

            if tmp not in tmp_dir:
                tmp_dir = os.path.join(tmp_dir, tmp)
            tmp_dir = os.path.join(tmp_dir, f"{task_id.replace('/', '-')}-{random_id}")
            tmp_dir = os.path.abspath(tmp_dir)

            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir, exist_ok=True)
            os.chdir(tmp_dir)
            with safe_open(os.path.join(tmp_dir, "Main.java"), mode='w', permission_mode=0o644) as f:
                f.write(sample[test_code])

            res = "failed: unknown error"
            compile_returncode = -1
            for _ in range(5):
                try:
                    javac_executable = shutil.which("javac")
                    if javac_executable is None:
                        raise FileNotFoundError("JavaC executable not found in the system PATH.")
                    compilation_result = subprocess.run([javac_executable, os.path.join(tmp_dir, "Main.java")],
                                                        timeout=5, capture_output=True)
                    compile_returncode = compilation_result.returncode
                    break
                except subprocess.TimeoutExpired:
                    continue
            if compile_returncode != 0:
                res = "failed: compilation error"
            else:
                exec_result = None
                try:
                    java_executable = shutil.which("java")
                    if java_executable is None:
                        raise FileNotFoundError("Java executable not found in the system PATH.")
                    # WARNING
                    # This program exists to execute untrusted model-generated code. Although
                    # it is highly unlikely that model-generated code will do something overtly
                    # malicious in response to this test suite, model-generated code may act
                    # destructively due to a lack of model capability or alignment.
                    # Users are strongly encouraged to sandbox this evaluation suite so that it
                    # does not perform destructive actions on their host or network.
                    exec_result = subprocess.run([java_executable, '-cp', tmp_dir, 'Main'],
                                                 timeout=timeout, capture_output=True)
                    if exec_result.returncode == 0:
                        res = passed
                    elif exec_result.returncode == 1:
                        if "AssertionError" in exec_result.stderr.decode('unicode-escape'):
                            res = "failed: wrong answer"
                        else:
                            res = f"failed: {exec_result.stderr.decode()}"
                except subprocess.TimeoutExpired:
                    res = "time out"
                except BaseException as e:
                    res = f"failed: {e}"
            result.append(res)

            tmp_dir = os.path.abspath(tmp_dir)
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(tmp_dir,))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append(timed_out)

    return {
        "task_id": task_id,
        "completion_id": completion_id,
        test_code: sample[test_code],
        "prompt": sample["prompt"],
        "generation": sample["generation"],
        "result": result[0],
        passed: result[0] == passed,
        finish: -1 if finish not in sample else sample[finish],
        file: "" if file not in sample else sample[file],
        output: [] if output not in sample else sample[output],
    }
