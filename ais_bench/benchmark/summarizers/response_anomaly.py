import json
from collections import Counter
from pathlib import Path


RESULT_NAMES = (
    'normal', 'rare_character', 'garbled', 'repetition', 'nan_value',
    'skipped', 'failed', 'unavailable', 'unknown',
)


def read_anomaly_counts(path, logger):
    path = Path(path)
    if not path.is_file():
        return None
    records = {}
    try:
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict) or 'id' not in record or 'uuid' not in record:
                        raise ValueError('missing case identity')
                    records[(record['id'], record['uuid'])] = record
                except (ValueError, TypeError) as exc:
                    logger.warning('Invalid anomaly result %s:%d: %s', path, line_number, exc)
                    return None
    except OSError as exc:
        logger.warning('Cannot read anomaly results %s: %s', path, exc)
        return None
    counts = Counter()
    for record in records.values():
        status = record.get('detection_status')
        if status == 'completed':
            name = record.get('anomaly_type_name', 'unknown')
            counts[name if name in RESULT_NAMES[:5] else 'unknown'] += 1
            counts['completed'] += 1
            counts['count'] += int(bool(record.get('is_anomaly')))
        else:
            counts[status if status in RESULT_NAMES[5:8] else 'unknown'] += 1
    counts['total'] = len(records)
    return counts


def append_anomaly_rows(table, work_dir, model_pairs, datasets, logger):
    """Append persisted detection metrics without changing the summary schema."""
    metrics = ('total',) + RESULT_NAMES + ('count', 'rate')
    model_columns = table[0][table[0].index('mode') + 1:]
    if model_columns and model_columns[0] == 'total_count':
        model_columns = model_columns[1:]
    has_total_count = 'total_count' in table[0]
    for dataset in dict.fromkeys(datasets):
        results = {}
        for storage_name, display_name in model_pairs:
            counts = read_anomaly_counts(
                Path(work_dir) / 'response_anomaly' / storage_name / f'{dataset}.jsonl', logger)
            if counts is not None:
                if display_name in results:
                    results[display_name].update(counts)
                else:
                    results[display_name] = counts
        if not results:
            continue
        for metric in metrics:
            row = [dataset, '-', f'response_anomaly_{metric}', 'response_anomaly']
            if has_total_count:
                row.append('-')
            for model in model_columns:
                counts = results.get(model)
                if counts is None:
                    value = '-'
                elif metric == 'rate':
                    value = (f"{100 * counts['count'] / counts['completed']:.2f}"
                             if counts['completed'] else '-')
                else:
                    value = str(counts[metric])
                row.append(value)
            table.append(row)
    return table
