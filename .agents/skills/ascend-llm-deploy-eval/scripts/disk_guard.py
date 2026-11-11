"""Initial host-local storage budget; retained artifacts count toward the limit."""
from pathlib import Path
import shutil

GIB = 1024 ** 3
SUBTASK_BYTES = 4 * GIB
RESERVE_BYTES = 10 * GIB
INTERVAL = 30

class DiskLimitError(RuntimeError):
    pass

def used_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file() and not p.is_symlink())

def check(root, count=1, case_dir=None, admission=False):
    root = Path(root)
    used = used_bytes(root)
    limit = (max(1, count) * 4 + 1) * GIB
    free = shutil.disk_usage(root).free
    case_used = used_bytes(case_dir) if case_dir and Path(case_dir).exists() else 0
    result = dict(used_bytes=used, limit_bytes=limit, free_bytes=free,
                  reserve_bytes=RESERVE_BYTES, subtask_bytes=case_used,
                  subtask_limit_bytes=SUBTASK_BYTES)
    from stage_runtime import save
    save(root/'storage-state.json', result)
    required = RESERVE_BYTES + max(0, limit-used) if admission else RESERVE_BYTES
    if used >= limit or case_used >= SUBTASK_BYTES or free < required:
        raise DiskLimitError('Storage budget exceeded: '+str(result))
    return result
