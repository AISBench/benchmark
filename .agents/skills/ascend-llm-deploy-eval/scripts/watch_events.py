"""Read current status timelines or historical event streams without controlling tasks."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from task_events import read_events


def format_event(record):
    source = record.get('source_event', record)
    if isinstance(source['time'], (int, float)):
        timestamp = datetime.fromtimestamp(source['time'], timezone.utc).astimezone()
    else:
        timestamp = datetime.fromisoformat(source['time'].replace('Z', '+00:00')).astimezone()
    identity = '/'.join(str(v) for v in [record.get('task_id'), source.get('subcase_id'),
                       source.get('stage'), source.get('kind'), source.get('id')] if v)
    attempt = ' attempt='+str(source['attempt']) if 'attempt' in source else ''
    detail = source.get('error', source.get('reason', ''))
    return f"[{timestamp:%Y-%m-%d %H:%M:%S %z}] {identity} {source['status']}{attempt} {detail}"


def poll(root, cursor=0):
    status = root/'status.json'
    if status.is_file():
        state = json.loads(status.read_text(encoding='utf-8'))
        if 'timeline' in state:
            events = state['timeline']
            if cursor > len(events):
                raise ValueError('Timeline truncated; restart observer explicitly')
            return len(events), events[cursor:]
    result = read_events(root, cursor)
    return result['offset'], result['events']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    cursor = 0
    try:
        while True:
            next_cursor, events = poll(args.run_dir, cursor)
            for event in events:
                print(format_event(event), flush=True)
            advanced = next_cursor != cursor
            cursor = next_cursor
            if args.once and not advanced:
                break
            if not advanced:
                time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
