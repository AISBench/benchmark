# Harbor normalized result export

HarborTask (including its HarborAgentTask subclass) consumes a published
`summary/normalized_result.json` under the Harbor job directory. To consume a
report stored elsewhere, set the dataset's `args.normalized_result_path` to its
absolute path. The producer must finish publishing the report before export;
this setting does not generate or score a report.

```python
args = dict(
    # Other Harbor dataset arguments remain unchanged.
    normalized_result_path='/workspace/job/summary/normalized_result.json',
)
```

The sibling `reporting_complete.json` must contain `result_sha256`, computed
with SHA256 over UTF-8 `json.dumps(result, sort_keys=True, ensure_ascii=False,
allow_nan=False)`. Schema version 1.0, integrity/acceptance states, accuracy
range and state consistency are checked before consumption. Missing or corrupt
publication is exported as `accuracy=null`, `avg_score=null`, invalid/blocked
with diagnostic reasons. It is never converted to a numeric zero.

The normalized metrics, counts, reasons and status are preserved. `avg_score`
is an unrounded compatibility alias of normalized `metrics.accuracy`. Existing
reward-only benchmarks retain their legacy export behavior. The generic Harbor
table and CSV expose integrity and acceptance; a null score remains empty in
CSV. No dataset-specific summarizer is used.

## Regression commands

Run at the AISBench repository root with pytest, mmengine-lite, tabulate, tqdm
and orjson installed in an isolated Python environment:

```bash
python -m pytest \
  tests/UT/tasks/custom_tasks/test_harbor_normalized_results.py \
  tests/UT/tasks/custom_tasks/test_harbor_task.py \
  tests/UT/tasks/custom_tasks/test_harbor_agent_task.py \
  tests/UT/summarizers/test_harbor.py \
  -v --tb=short --junitxml=harbor-export-regression.xml
```

The normalized suite imports the actual HarborTask and generic summarizer. It
covers successful/failed/blocked results, publication failures, unrounded
scores, repeat export, legacy rewards, and export-to-table/CSV. Existing task,
agent-task and summarizer tests check compatibility. These deterministic tests
make no model API calls and do not certify live multi-task scheduling/resume
or official model baseline parity.
