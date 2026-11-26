"""Small, model-free timing records for CI and incomplete smoke runs."""
import argparse
import json
import math
from pathlib import Path
import time
import uuid


class CaseTimings:
    def __init__(self, workspace, case_count, fast=False):
        self.path = Path(workspace) / 'case_timings.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.suite_id = uuid.uuid4().hex
        self.records = []
        self.has_errors = False
        self.case_count = case_count
        self._append(dict(event='suite_start', case_count=case_count, fast=fast))

    def _append(self, record):
        record = dict(record, suite_id=self.suite_id, timestamp=time.time())
        # Reopen/close each short record so completed cases survive cancellation.
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')

    def record(self, config, result):
        duration = result.get('execution_time')
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            duration = None
        record = dict(event='case_end', case_name=config.get('case_name', 'unknown'),
                      rank=config.get('rank_size'), success=bool(result.get('success')),
                      elapsed_s=duration)
        self.records.append(record)
        self._append(record)
        return duration

    def error(self, case_name):
        self.has_errors = True
        self._append(dict(event='processing_error', case_name=case_name))

    def slowest(self, count=10):
        timed = [record for record in self.records if record['elapsed_s'] is not None]
        return sorted(timed, key=lambda record: record['elapsed_s'], reverse=True)[:count]

    def finish(self, elapsed_s, interrupted=False):
        success = (not interrupted and not self.has_errors and len(self.records) == self.case_count
                   and all(record['success'] for record in self.records))
        self._append(dict(event='suite_end', completed=len(self.records), expected=self.case_count,
                          elapsed_s=elapsed_s, interrupted=interrupted, success=success))


def unfinished_names(futures, limit=10):
    """Call only after submission; unfinished includes queued and active tasks."""
    names = [config.get('case_name', 'unknown') for future, config in futures.items()
             if not future.done()]
    suffix = ', ... (+{})'.format(len(names) - limit) if len(names) > limit else ''
    return ', '.join(names[:limit]) + suffix


def completed_successfully(path):
    """Reject absent, truncated, failed or stale-only results, not just failure dirs."""
    start = end = None
    cases = set()
    failed = False
    try:
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('event') == 'suite_start':
                    start, end, cases, failed = row, None, set(), False
                elif start and row.get('suite_id') == start.get('suite_id'):
                    if end is not None:
                        return False  # No case/error/end may follow a terminal record.
                    if row.get('event') == 'case_end':
                        name = row.get('case_name')
                        failed |= (row.get('success') is not True or name in cases
                                   or not isinstance(name, str) or not name)
                        cases.add(name)
                    elif row.get('event') == 'processing_error':
                        failed = True
                    elif row.get('event') == 'suite_end':
                        end = row
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return bool(start and end and not failed and end.get('success') is True
                and isinstance(start.get('suite_id'), str) and start['suite_id']
                and end.get('interrupted') is False and type(start.get('case_count')) is int
                and start['case_count'] > 0
                and type(end.get('expected')) is int and type(end.get('completed')) is int
                and start['case_count'] == end['expected'] == end['completed'] == len(cases))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', type=Path, required=True)
    args = parser.parse_args()
    if completed_successfully(args.check):
        print('[timing] complete successful suite confirmed')
    else:
        parser.exit(1, '[timing] missing, incomplete or failed suite results\n')
