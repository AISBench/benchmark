from ais_bench.benchmark.utils.postprocess import model_postprocessors as mp


class DummyPool:
    def __init__(self, n):
        self.n = n
        self._batches = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, func, batches):
        # Capture batches for assertions and run synchronously
        self._batches.extend(list(batches))
        return [func(batch) for batch in self._batches]


def iter_identity(x):
    return x

def test_extract_non_reasoning_content_only_end_token():
    t = "This is a test.</think> How are you?"
    assert mp.extract_non_reasoning_content(t) == "How are you?"


def test_extract_non_reasoning_content_both_tokens():
    t = "Start<think>reasoning here</think> End"
    assert mp.extract_non_reasoning_content(t) == "Start End"


def test_extract_non_reasoning_content_no_tokens():
    t = "Plain text"
    assert mp.extract_non_reasoning_content(t) == "Plain text"


def test_extract_non_reasoning_content_list_input():
    inputs = [
        "Start<think>reasoning</think> End",
        "Test</think> Result",
    ]
    assert mp.extract_non_reasoning_content(inputs) == ["Start End", "Result"]


def test_extract_non_reasoning_content_custom_tokens():
    single = "A <r>reason</r> B"
    assert (
        mp.extract_non_reasoning_content(
            single, think_start_token="<r>", think_end_token="</r>"
        )
        == "A  B".strip()
    )

    lst = ["P <r>a</r> Q", "X <r>b</r> Y"]
    assert mp.extract_non_reasoning_content(
        lst, think_start_token="<r>", think_end_token="</r>"
    ) == ["P  Q".strip(), "X  Y".strip()]


def test_extract_non_reasoning_content_multiple_pairs():
    t = "A<think>x</think>B<think>y</think>C"
    assert mp.extract_non_reasoning_content(t) == "ABC"


def test_extract_non_reasoning_content_only_start_token():
    t = "Hello <think>unfinished"
    assert mp.extract_non_reasoning_content(t) == t.strip()
