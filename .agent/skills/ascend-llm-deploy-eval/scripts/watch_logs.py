"""Read-only follower that discovers new deployment and attempt logs."""
import argparse
import codecs
from pathlib import Path
import time


def poll(root, kind, cursors):
    patterns = ['deployments/*/*.log'] if kind == 'service' else [
        'sub_tasks/*/attempt-*/benchmark.log',
        'sub_tasks/*/attempt-*/runner.log',
        'sub_tasks/*/attempt-*/outputs/*/logs/**/*.out',
        'sub_tasks/*/attempt-*/outputs/*/logs/**/*.log',
    ]
    roots = [root] + list((root/'nodes').glob('*'))
    for base in roots:
        paths = {path for pattern in patterns for path in base.glob(pattern) if path.is_file()}
        for path in sorted(paths):
            stat = path.stat()
            previous = cursors.get(path)
            if previous is None or previous[0] != stat.st_ino or stat.st_size < previous[1]:
                previous = (stat.st_ino, 0, codecs.getincrementaldecoder('utf-8')('replace'))
            inode, offset, decoder = previous
            with path.open('rb') as stream:
                stream.seek(offset)
                data = stream.read(65536)
                cursors[path] = (inode, stream.tell(), decoder)
            if data:
                print('\n--- '+str(path)+' ---', flush=True)
                print(decoder.decode(data), end='', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--kind', choices=['service', 'test'], required=True)
    args = parser.parse_args()
    cursors = {}
    print('Waiting for logs; Ctrl+C closes this observer only.', flush=True)
    try:
        while True:
            poll(args.run_dir, args.kind, cursors)
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
