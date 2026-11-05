"""Readable terminal observer; closing it never stops the task."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from task_events import read_events


def format_event(record):
    source = record.get('source_event', record)
    identity = '/'.join(str(v) for v in [record.get('task_id'), source.get('subcase_id'),
                                        source.get('stage')] if v)
    attempt = ' attempt='+str(source['attempt']) if 'attempt' in source else ''
    detail = source.get('error', source.get('reason', ''))
    timestamp = datetime.fromisoformat(source['time'].replace('Z', '+00:00'))
    timestamp = timestamp.astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S +08:00')
    return f"[{timestamp}] {identity} {source['status']}{attempt} {detail}"


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--once', action='store_true')
    args = p.parse_args()
    offset = 0
    seen = set()
    while True:
        result = read_events(args.run_dir, offset)
        for event in result['events']:
            identity = (event.get('source_stream'), event.get('source_event', event)['event_id'])
            if identity not in seen:
                print(format_event(event), flush=True)
                seen.add(identity)
        advanced = result['offset'] != offset
        offset = result['offset']
        if args.once and not advanced:
            break
        if not advanced:
            time.sleep(1)
