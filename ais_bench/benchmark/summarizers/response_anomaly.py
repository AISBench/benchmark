import json
from collections import Counter
from pathlib import Path


RESULT_NAMES = (
    'normal', 'rare_character', 'garbled', 'repetition', 'nan_value',
    'skipped', 'failed', 'unavailable', 'unknown',
)

ANOMALY_NAMES = RESULT_NAMES[:5]
NON_DETECTION_NAMES = RESULT_NAMES[5:8]


def _case_key(record):
    return (record.get('id'), record.get('uuid'))


def load_prediction_keys(path, logger):
    """Return the ``(id, uuid)`` pairs of the current predictions.

    Returns ``None`` when the file cannot be read, so callers can decide
    whether counting without the filter is acceptable.
    """
    path = Path(path)
    if not path.is_file():
        return None
    keys = set()
    try:
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError('not a JSON object')
                    keys.add(_case_key(record))
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        'Invalid prediction record %s:%d: %s',
                        path, line_number, exc,
                    )
                    return None
    except OSError as exc:
        logger.warning('Cannot read predictions %s: %s', path, exc)
        return None
    return keys


def read_anomaly_counts(path, logger, valid_keys=None):
    """Count one dataset's persisted detection results.

    ``valid_keys`` restricts counting to the cases of the current
    predictions: results are append-only, so a case re-inferred in the same
    work directory leaves a stale record behind and would otherwise be
    counted twice.
    """
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
                    records[_case_key(record)] = record
                except (ValueError, TypeError) as exc:
                    logger.warning('Invalid anomaly result %s:%d: %s', path, line_number, exc)
                    return None
    except OSError as exc:
        logger.warning('Cannot read anomaly results %s: %s', path, exc)
        return None
    if valid_keys is not None:
        records = {key: value for key, value in records.items() if key in valid_keys}
        if not records:
            return None
    counts = Counter()
    for record in records.values():
        status = record.get('detection_status')
        if status == 'completed':
            name = record.get('anomaly_type_name', 'unknown')
            counts[name if name in ANOMALY_NAMES else 'unknown'] += 1
            counts['completed'] += 1
            counts['count'] += int(bool(record.get('is_anomaly')))
        else:
            counts[status if status in NON_DETECTION_NAMES else 'unknown'] += 1
    counts['total'] = len(records)
    return counts


def _format_value(metric, counts):
    if counts is None:
        return '-'
    if metric == 'rate':
        if not counts['completed']:
            return '-'
        return f"{100 * counts['count'] / counts['completed']:.2f}"
    return str(counts[metric])


def append_anomaly_rows(table, work_dir, model_pairs, datasets, logger):
    """Append persisted detection metrics without changing the summary schema.

    ``model_pairs`` are ``(storage_abbr, display_abbr)`` tuples and
    ``datasets`` are ``(storage_abbr, display_abbr)`` tuples: judge-based
    datasets are renamed for display while their artifacts stay under the
    inference abbreviation.
    """
    metrics = ('total',) + RESULT_NAMES + ('count', 'rate')
    header = table[0]
    mode_index = header.index('mode')
    model_columns = header[mode_index + 1:]
    has_total_count = 'total_count' in header
    if has_total_count and model_columns and model_columns[0] == 'total_count':
        model_columns = model_columns[1:]
    work_dir = Path(work_dir)
    for storage_dataset, display_dataset in datasets:
        results = {}
        for storage_model, display_model in model_pairs:
            predictions = work_dir / 'predictions' / storage_model / f'{storage_dataset}.jsonl'
            valid_keys = load_prediction_keys(predictions, logger)
            if valid_keys is None:
                logger.warning(
                    'Predictions %s are unavailable; response anomaly metrics '
                    'for %s are skipped.', predictions, display_dataset,
                )
                continue
            counts = read_anomaly_counts(
                work_dir / 'response_anomaly' / storage_model / f'{storage_dataset}.jsonl',
                logger,
                valid_keys,
            )
            if counts is None:
                continue
            if display_model in results:
                merged = results[display_model]
                merged.update(counts)
            else:
                results[display_model] = counts
        if not results:
            continue
        for metric in metrics:
            row = [display_dataset, '-', f'response_anomaly_{metric}', 'response_anomaly']
            if has_total_count:
                row.append('-')
            for model in model_columns:
                row.append(_format_value(metric, results.get(model)))
            table.append(row)
    return table
