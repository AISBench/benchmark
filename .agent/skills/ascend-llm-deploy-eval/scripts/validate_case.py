"""Validate raw AISBench request records, output lengths, and cache evidence."""
import ast
import json
from pathlib import Path
import re


def validate(folder, case):
    files = sorted((folder/'tool/outputs').glob('*/performances/*/*_details.jsonl'))
    if len(files) != 2:
        raise ValueError('Expected independent warmup and formal detail files')
    reports = []
    for path, count, output in zip(files, [case.get('prefill_dp',2)*case.get('prefix_num',1), case['formal_requests']], [1, case['output_tokens']]):
        seen, inputs, outputs = set(), [], []
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['id'] in seen or row.get('success') is not True or row.get('error_info'):
                    raise ValueError('Duplicate or unsuccessful request in '+str(path))
                seen.add(row['id'])
                inputs.append(row['input_tokens'])
                outputs.append(row['output_tokens'])
        if len(seen) != count or any(value != output for value in outputs):
            raise ValueError('Request count or per-request output length mismatch')
        if len(reports) == 1 and any(value < case.get('input_tokens',1) for value in inputs):
            raise ValueError('Actual input shorter than requested target')
        reports.append(dict(requests=count, input_min=min(inputs), input_max=max(inputs),
                            input_mean=sum(inputs)/count, output_tokens=output, details=str(path)))
    log = (folder/'tool/aisbench.log').read_text()
    queries = [ast.literal_eval(x) for x in re.findall(r'queries: normal=(\{[^\n]+?\}), external=', log)]
    hits = [ast.literal_eval(x) for x in re.findall(r'hits: normal=(\{[^\n]+?\}), external=', log)]
    def unavailable(reason):
        (folder/'validated.json').write_text(json.dumps({'status':'REVIEW_REQUIRED','case_id':case['id'],
            'formal':reports[1],'warmup':reports[0],'measured_prefix_ratio':None,
            'prefix_metrics':{'status':'unavailable','reason':reason}},indent=2))
        raise ValueError(reason)
    if len(queries) != 4 or len(hits) != 4:
        unavailable('Expected four metric boundaries: warmup and formal before/after')
    engines = set(range(case.get('prefill_dp',2)))
    if any(set(x) != engines for x in queries+hits):
        unavailable('Missing prefill DP engine metrics')
    metrics = {}
    for engine in engines:
        warmup_queries = queries[1][engine]-queries[0][engine]
        q = queries[3][engine]-queries[2][engine]
        h = hits[3][engine]-hits[2][engine]
        if warmup_queries <= 0 or q <= 0 or not 0 <= h <= q:
            unavailable('Invalid metrics delta or incomplete warmup coverage')
        metrics[str(engine)] = dict(warmup_queries=warmup_queries, formal_queries=q,
                                    formal_hits=h, hit_ratio=h/q)
    for port in case.get('reset_ports',[9000,9001]) if case.get('reset_cache',True) else []:
        resets = sorted(folder.glob(f'reset-{port}-*.json'))
        if not resets or json.loads(resets[-1].read_text()).get('success') is not True:
            raise ValueError('Cache reset not verified')
    result = dict(status='PASSED', case_id=case['id'], target_prefix_ratio=case['prefix_ratio'],
                  warmup=reports[0], formal=reports[1], prefix_metrics=metrics,
                  measured_prefix_ratio=sum(x['formal_hits'] for x in metrics.values())/sum(x['formal_queries'] for x in metrics.values()))
    (folder/'validated.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    import sys
    folder = Path(sys.argv[1])
    print(json.dumps(validate(folder, json.loads((folder/'case.json').read_text())), indent=2))
