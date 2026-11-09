"""Validate standard AISBench perf artifacts without plugin dependencies."""
import json
from cache_reset import validate_reset


def validate(folder, case):
    execution = json.loads((folder/'execution.json').read_text(encoding='utf-8'))
    if execution.get('returncode') != 0:
        raise ValueError('AISBench did not exit successfully')
    files = list((folder/'outputs').glob('*/performances/*/*_details.jsonl'))
    if len(files) != 1:
        raise ValueError('Expected one formal request detail file')
    seen = set()
    zero_outputs = 0
    input_min = input_max = None
    output_min = output_max = None
    with files[0].open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line)
            if row['id'] in seen:
                raise ValueError('Duplicate request ID')
            seen.add(row['id'])
            if row.get('success') is not True or row.get('error_info'):
                raise ValueError('AISBench request failed')
            inputs, outputs = row.get('input_tokens'), row.get('output_tokens')
            if type(inputs) is not int or inputs < 1 or type(outputs) is not int or outputs < 0 or outputs > case['output_tokens']:
                raise ValueError('Invalid request token lengths')
            zero_outputs += outputs == 0
            if case.get('input_tokens') is not None and inputs != case['input_tokens']:
                raise ValueError('Dataset input token length differs from explicit expectation')
            input_min = inputs if input_min is None else min(input_min, inputs)
            input_max = inputs if input_max is None else max(input_max, inputs)
            output_min = outputs if output_min is None else min(output_min, outputs)
            output_max = outputs if output_max is None else max(output_max, outputs)
    if len(seen) != case['formal_requests']:
        raise ValueError('Formal request count mismatch')
    if case.get('reset_cache', False):
        validate_reset(folder)
    logs = sorted(str(path) for path in (folder/'outputs').glob('*/logs/**/*') if path.is_file())
    if not logs:
        raise ValueError('Missing AISBench on-disk worker logs')
    result = dict(status='PASSED', case_id=case['id'], dataset_id=case.get('dataset_id','default'),
                  formal={'requests':len(seen),'details':str(files[0]),'input_min':input_min,'input_max':input_max,
                          'output_min':output_min,'output_max':output_max,'zero_output_requests':zero_outputs}, logs=logs)
    (folder/'validated.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
