import json
import fcntl
import os


def safe_write(results_dict: dict, filename):
    """
    use fcntl file lock to implement mutual exclusion writing
    """
    with open(filename, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            results=[json.dumps(result, ensure_ascii=False) + "\n" for result in results_dict.values()]
            f.writelines(results)
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def dump_results_dict(results_dict, filename, formatted=True):
    with open(filename, "w", encoding="utf-8") as json_file:
        if formatted:
            json.dump(results_dict, json_file, indent=4, ensure_ascii=False)
        else:
            json.dump(results_dict, json_file, ensure_ascii=False)
