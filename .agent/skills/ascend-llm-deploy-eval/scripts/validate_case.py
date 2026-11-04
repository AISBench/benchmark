"""Validate native AISBench records and Prefix Cache analysis artifacts."""
import json


def validate(folder, case):
    analyses = list(folder.glob('prefix_*/result/*.analysis.json'))
    if len(analyses) != 1:
        raise ValueError('Expected one native Prefix Cache analysis')
    analysis = json.loads(analyses[0].read_text(encoding='utf-8'))
    runtime = analysis.get('runtime', {})
    if analysis.get('status') != 'complete' or runtime.get('aisbench_exit_code') != 0:
        raise ValueError('Native benchmark did not complete successfully')
    if case.get('warmup', True):
        warmup = runtime.get('warmup', [])
        pairs = {(row['group_id'], row['dp_rank']) for row in warmup}
        groups = {row['group_id'] for row in warmup}
        expected = {(group, rank) for group in groups for rank in range(case.get('prefill_dp', 1))}
        if (len(groups) != case.get('prefix_num', 1) or pairs != expected or
                len(warmup) != len(expected) or not all(row.get('success') is True for row in warmup)):
            raise ValueError('Incomplete native warmup coverage')
    files = list((folder/'outputs').glob('*/performances/*/*_details.jsonl'))
    if len(files) != 1:
        raise ValueError('Expected one formal request detail file')
    rows = [json.loads(line) for line in files[0].read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != case['formal_requests'] or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Formal request count or unique ID mismatch')
    if any(row.get('success') is not True or row.get('error_info') or
           row.get('input_tokens') != case['input_tokens'] or row.get('output_tokens') != case['output_tokens'] for row in rows):
        raise ValueError('Request failure or token length mismatch')
    engines = analysis.get('actual', {}).get('by_dp', {})
    if set(engines) != {str(i) for i in range(case.get('prefill_dp', 1))}:
        raise ValueError('Missing DP metrics')
    if any(not row['queries'] > 0 or not 0 <= row['hits'] <= row['queries'] for row in engines.values()):
        raise ValueError('Invalid prefix metrics delta')
    queries = sum(row['queries'] for row in engines.values())
    hits = sum(row['hits'] for row in engines.values())
    for port in case.get('reset_ports', []):
        resets = sorted(folder.glob(f'reset-{port}-*.json'))
        if not resets or json.loads(resets[-1].read_text()).get('success') is not True:
            raise ValueError('Cache reset not verified')
    result = dict(status='PASSED', case_id=case['id'], target_prefix_ratio=case['prefix_ratio'],
                  measured_prefix_ratio=hits/queries, formal={'requests':len(rows), 'details':str(files[0])},
                  warmup=runtime.get('warmup', []), prefix_metrics=engines,
                  warnings=analysis.get('warnings', []), analysis=str(analyses[0]))
    (folder/'validated.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
