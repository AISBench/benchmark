"""Read-only follower for current and historical task logs."""
import argparse
import codecs
import os
from pathlib import Path
import time


def discover(root, kind):
    patterns = (['logs/**/*.log', 'deployments/*/*.log'] if kind == 'service' else [
        'results/*/aisbench.log', 'results/*/outputs/*/logs/**/*',
        'sub_tasks/*/attempt-*/benchmark.log', 'sub_tasks/*/attempt-*/runner.log',
        'sub_tasks/*/attempt-*/outputs/*/logs/**/*'])
    return sorted({p for base in [root, *sorted((root/'nodes').glob('*'))]
                   for pattern in patterns for p in base.glob(pattern)
                   if p.is_file() and p.suffix in ('.log', '.out')})


def poll(root, kind, cursors, paths=None):
    """Drain each observed snapshot in bounded chunks; new bytes wait one poll."""
    for path in discover(root, kind) if paths is None else paths:
        try:
            with path.open('rb') as stream:
                stat = os.fstat(stream.fileno())
                previous = cursors.get(path)
                if previous is None or previous[0] != stat.st_ino or stat.st_size < previous[1]:
                    previous = (stat.st_ino, 0, codecs.getincrementaldecoder('utf-8')('replace'))
                inode, offset, decoder = previous
                stream.seek(offset)
                remaining = max(0, stat.st_size-offset)
                if remaining:
                    print('\n--- '+str(path)+' ---', flush=True)
                while remaining:
                    data = stream.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    print(decoder.decode(data), end='', flush=True)
                cursors[path] = (inode, stream.tell(), decoder)
        except FileNotFoundError:
            cursors.pop(path, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--kind', choices=['service', 'test'], required=True)
    args = parser.parse_args()
    cursors, paths, next_scan = {}, [], 0
    print('Waiting for logs; Ctrl+C closes this observer only.', flush=True)
    try:
        while True:
            now = time.monotonic()
            if now >= next_scan:
                paths = discover(args.run_dir, args.kind)
                for stale in cursors.keys()-set(paths):
                    del cursors[stale]
                next_scan = now+5
            poll(args.run_dir, args.kind, cursors, paths)
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
