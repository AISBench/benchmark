"""Run the generated serial task after attended preflight has completed."""

if __name__ == '__main__':
    raise SystemExit('Retired lifecycle entry. Use scripts/run.py --mode FULL|scripts; historical frozen jobs retain their own code.')
import argparse
from pathlib import Path
from stage_runtime import record
from task_runtime import stage_entry

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    args = p.parse_args()
    if not args.run_dir.is_absolute():
        p.error('run-dir must be absolute')
    raise SystemExit(record(args.run_dir, 'evaluate', lambda: stage_entry('evaluate', args.run_dir)))
