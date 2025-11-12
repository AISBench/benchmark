# test_runner.py
import sys
import json
import traceback

from testing_util import run_test

def main():
    try:
        payload = json.load(sys.stdin)
        sample = payload.get('sample')
        generation = payload.get('generation')
        debug = payload.get('debug', False)
        timeout = payload.get('timeout', 10)

        res, meta = run_test(sample, test=generation, debug=debug, timeout=timeout)

        out = {'res': res, 'meta': meta, 'error': None}
        # always use single line JSON output, for main process parsing
        print(json.dumps(out), flush=True)
        sys.exit(0)
    except Exception as e:
        tb = traceback.format_exc()
        err_out = {
            'res': None,
            'meta': None,
            'error': str(e),
            'traceback': tb
        }
        # send exception back to main process (and write stderr for manual debugging)
        print(json.dumps(err_out), flush=True)
        print("----stderr-traceback----", file=sys.stderr)
        print(tb, file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    main()
