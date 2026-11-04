"""Standalone config.py compatibility or one resolved per-attempt JSON input."""
import json
import os
from pathlib import Path
from config import *

_payload = None
if os.environ.get('BENCH_CASE_CONFIG'):
    _payload = json.loads(Path(os.environ['BENCH_CASE_CONFIG']).read_text(encoding='utf-8'))
    globals().update(_payload['service'])
    DEFAULT_PERFORMANCE_TEST = 'stable_stage' if _payload['case'].get('steady_state',False) else 'default_perf'

def case_arguments():
    if _payload is None: return {}
    case=_payload['case']
    fields={'input_len':'input_tokens','output_len':'output_tokens','data_num':'formal_requests','concurrency':'concurrency','request_rate':'request_rate','prefix_num':'prefix_num','repeat_rate':'prefix_ratio','seed':'seed'}
    result={key:case[value] for key,value in fields.items()}
    for key in ('output_len','concurrency','request_rate','repeat_rate'): result[key]=str(result[key])
    result.update(dp=_payload['dp'],npu_num=_payload['npu_num'],test_type='stream',dataset_type='prefix_cache',prefix_test=True)
    return result
