# Retriever

In dataset configuration files, there is a `retriever` field that indicates how to retrieve samples from the dataset as in-context examples. The most commonly used one is `FixKRetriever`, which means fixed use of k samples, thus it is k-shot. There is also `ZeroRetriever`, which means no samples are used, which in most cases means 0-shot.

On the other hand, in-context samples can also be directly specified in the dataset template. In this case, `ZeroRetriever` will also be used, but the evaluation at this time is not 0-shot, and needs to be determined according to the specific template. For details, please refer to [prompt_template](./prompt_template.md).

Currently, AISBench supports the following `Retriever` types:

- **`ZeroRetriever`**: Does not use any samples as in-context examples
- **`FixKRetriever`**: Fixed use of k samples as in-context examples
- **`RandomRetriever`**: Random use of k samples as in-context examples

## ZeroRetriever

`ZeroRetriever` is a zero-shot retriever that does not retrieve any samples from the training set as in-context. For each test sample, it returns an empty index list, so it is usually used to implement 0-shot evaluation.

### Configuration Method

```python
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever

infer_cfg = dict(
    retriever=dict(type=ZeroRetriever),
    # ... Other configurations
)
```

### Function Description

- **Return Value**: For all test samples, returns an empty index list `[]`
- **Use Cases**:
  - 0-shot evaluation scenarios
  - When in-context samples are already hardcoded in the prompt template (in this case, although `ZeroRetriever` is used, it is actually not 0-shot)

### Actual Example

Assume we have a question-answering dataset with the following samples in the training set:

**Training Set (train)**:

- Sample 0: `{"question": "What is artificial intelligence?", "answer": "Artificial intelligence is a branch of computer science"}`
- Sample 1: `{"question": "What is Python?", "answer": "Python is a programming language"}`

**Test Set (test)**:

- Sample 0: `{"question": "What is machine learning?", "answer": "Machine learning is a subfield of AI"}`

When using `ZeroRetriever`, for test sample 0, no training samples will be retrieved, and the generated prompt will not contain any in-context examples.

## FixKRetriever

`FixKRetriever` is a fixed K-sample retriever that will use a fixed K samples from the training set as in-context examples for all test samples. This is the most commonly used way to implement k-shot evaluation.

### Configuration Method

```python
from ais_bench.benchmark.openicl.icl_retriever import FixKRetriever

infer_cfg = dict(
    retriever=dict(
        type=FixKRetriever,
        fix_id_list=[0, 1, 2, 3, 4]  # Specify to use samples with indices 0,1,2,3,4 from training set
    ),
    # ... Other configurations
)
```

### Parameter Description

- **`fix_id_list`** (`List[int]`): Required parameter, specifies the list of training sample indices to use. All test samples will use the same these samples as in-context.

### Function Description

- **Return Value**: For all test samples, returns the same index list (i.e., `fix_id_list`)
- **Use Cases**:
  - k-shot evaluation scenarios (such as 5-shot, 10-shot, etc.)
  - When it is necessary to ensure that all test samples use the same in-context examples to ensure evaluation consistency

### Actual Example

Assume we have a reading comprehension dataset:

**Training Set (train)**:

- Sample 0: `{"article": "Article A...", "question": "Question 1", "answer": "A"}`
- Sample 1: `{"article": "Article B...", "question": "Question 2", "answer": "B"}`
- Sample 2: `{"article": "Article C...", "question": "Question 3", "answer": "C"}`
- Sample 3: `{"article": "Article D...", "question": "Question 4", "answer": "D"}`
- Sample 4: `{"article": "Article E...", "question": "Question 5", "answer": "A"}`

**Test Set (test)**:

- Sample 0: `{"article": "Article X...", "question": "Test Question 1", "answer": "B"}`
- Sample 1: `{"article": "Article Y...", "question": "Test Question 2", "answer": "C"}`

Configuration example (5-shot):

```python
retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4])
```

**Workflow**:

1. For test sample 0:
   - Retrieve training samples [0, 1, 2, 3, 4]
   - Use `ice_template` to format these samples as in-context examples
   - Insert in-context examples into the test sample's prompt

2. For test sample 1:
   - Also retrieve training samples [0, 1, 2, 3, 4] (same as test sample 0)
   - Use the same in-context examples

**Generated Prompt Example** (assuming using a simple template):

```text
Read the article, and answer the question by replying A, B, C or D.

Article:
Article A...

Q: Question 1
Answer: A
Read the article, and answer the question by replying A, B, C or D.

Article:
Article B...

Q: Question 2
Answer: B
... (more examples)
Read the article, and answer the question by replying A, B, C or D.

Article:
Article X...

Q: Test Question 1
Answer:
```

### Configuration Examples

The following are some usage examples from actual configuration files:

#### Example 1: 5-shot Configuration

```python
# ais_bench/benchmark/configs/datasets/race/race_middle_gen_5_shot_chat.py
retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4])
```

#### Example 2: Using range to Generate Index List

```python
# ais_bench/benchmark/configs/datasets/triviaqa/triviaqa_gen_5_shot_chat_prompt.py
k = 5
retriever=dict(type=FixKRetriever, fix_id_list=list(range(k)))
```

#### Example 3: 10-shot Configuration

```python
# ais_bench/benchmark/configs/datasets/hellaswag/hellaswag_gen_10_shot_chat_prompt.py
retriever=dict(type=FixKRetriever, fix_id_list=list(range(10)))
```

#### Example 4: 25-shot Configuration

```python
# ais_bench/benchmark/configs/datasets/ARC_c/ARC_c_gen_25_shot_chat_prompt.py
retriever=dict(type=FixKRetriever, fix_id_list=[i for i in range(25)])
```

### Notes

1. **Index Range Check**: Indices in `fix_id_list` must be within the valid range of the training set (`[0, len(train))`), otherwise an `AISBenchValueError` exception will be thrown.

2. **Index Order**: The order in `fix_id_list` determines the order in which in-context examples appear in prompts.

3. **Use with ice_template**: When using `FixKRetriever`, it is usually necessary to configure `ice_template` to format retrieved samples.

## RandomRetriever

`RandomRetriever` is a random retriever that randomly selects K samples from the training set as in-context examples for each test sample. Unlike `FixKRetriever`, each test sample uses different in-context examples, all randomly selected.

### Configuration Method

```python
from ais_bench.benchmark.openicl.icl_retriever.icl_random_retriever import RandomRetriever

infer_cfg = dict(
    retriever=dict(
        type=RandomRetriever,
        ice_num=5,  # Specify the number of samples to retrieve for each test sample
        seed=43     # Random seed, used to ensure result reproducibility, default is 43
    ),
    # ... Other configurations
)
```

### Parameter Description

- **`ice_num`** (`int`): Required parameter, specifies the number of samples to retrieve for each test sample. Default is 1.
- **`seed`** (`Optional[int]`): Optional parameter, random seed, used to ensure result reproducibility. Default is 43. If the same seed is set, multiple runs will get the same result.

### Function Description

- **Return Value**: For each test sample, returns a randomly selected index list with length `ice_num`
- **Randomness**: In-context examples for each test sample are independently randomly selected
- **Reproducibility**: By setting the `seed` parameter, reproducible results can be guaranteed under the same configuration
- **Use Cases**:
  - When different in-context examples need to be used for each test sample
  - Research on the impact of different in-context examples on model performance
  - When random sampling is needed to reduce overfitting risk

### Actual Example

Assume we have a classification dataset:

**Training Set (train)**:

- Sample 0: `{"text": "This is an article about technology", "label": "Technology"}`
- Sample 1: `{"text": "This is an article about sports", "label": "Sports"}`
- Sample 2: `{"text": "This is an article about entertainment", "label": "Entertainment"}`
- Sample 3: `{"text": "This is an article about finance", "label": "Finance"}`
- Sample 4: `{"text": "This is an article about education", "label": "Education"}`
- Sample 5: `{"text": "This is an article about health", "label": "Health"}`

**Test Set (test)**:

- Sample 0: `{"text": "This is an article about AI", "label": "Technology"}`
- Sample 1: `{"text": "This is an article about football", "label": "Sports"}`

Configuration example (3-shot, seed=123):

```python
retriever=dict(type=RandomRetriever, ice_num=3, seed=123)
```

**Workflow**:

1. For test sample 0:
   - Randomly select 3 samples from training set (e.g., [1, 3, 5])
   - Use `ice_template` to format these samples as in-context examples
   - Insert in-context examples into the test sample's prompt

2. For test sample 1:
   - Randomly select 3 samples from training set (e.g., [0, 2, 4], may be different from test sample 0)
   - Use `ice_template` to format these samples as in-context examples
   - Insert in-context examples into the test sample's prompt

**Generated Prompt Example** (assuming test sample 0 randomly selected training samples [1, 3, 5]):

```text
</E>
Text: This is an article about sports
Label: Sports
</E>
Text: This is an article about finance
Label: Finance
</E>
Text: This is an article about health
Label: Health
</E>
Text: This is an article about AI
Label:
```

**Reproducibility Note**:

If the same `seed` value is used, multiple runs will get the same random result. For example:

```python
# First run
retriever1 = RandomRetriever(dataset, ice_num=3, seed=123)
result1 = retriever1.retrieve()

# Second run (same configuration)
retriever2 = RandomRetriever(dataset, ice_num=3, seed=123)
result2 = retriever2.retrieve()

# result1 and result2 are identical
```

### Notes

1. **Not Fully Tested**: The `RandomRetriever` class has a warning in the code indicating that it has not been fully tested and should be used with caution.

2. **Difference from FixKRetriever**:
   - `FixKRetriever`: All test samples use the same in-context examples
   - `RandomRetriever`: Each test sample uses different randomly selected in-context examples

3. **Random Seed**: If `seed` is not specified or different `seed` values are used each time, results will be different each run, which may affect evaluation result reproducibility.

4. **Use with ice_template**: When using `RandomRetriever`, it is usually necessary to configure `ice_template` to format retrieved samples.

5. **Import Path**: `RandomRetriever` is not exported in `__init__.py` and needs to be imported directly from the module path:

   ```python
   from ais_bench.benchmark.openicl.icl_retriever.icl_random_retriever import RandomRetriever
   ```

## Complete Configuration Example

The following is a complete dataset configuration example showing how to use both `ice_template` and `FixKRetriever`:

```python
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import FixKRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

reader_cfg = dict(
    input_columns=['article', 'question', 'A', 'B', 'C', 'D'],
    output_column='answer',
)

infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt='Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:'),
                dict(role='BOT', prompt='{answer}'),
            ]
        ),
        ice_token='</E>',  # Used to identify the position of in-context examples
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),  # 5-shot
    inferencer=dict(type=GenInferencer),
)
```

## Complete Configuration Example (RandomRetriever)

The following is a complete configuration example using `RandomRetriever`:

```python
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever.icl_random_retriever import RandomRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer

reader_cfg = dict(
    input_columns=['text'],
    output_column='label',
)

infer_cfg =dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin='</E>',
            round=[
                dict(role='HUMAN', prompt='Text: {text}'),
                dict(role='BOT', prompt='Label: {label}'),
            ]
        ),
        ice_token='</E>',
    ),
    retriever=dict(type=RandomRetriever, ice_num=3, seed=123),  # 3-shot, random selection
    inferencer=dict(type=GenInferencer),
)
```

