"""Record exact links created by an attempt; never delete their targets."""
import json
import os
from pathlib import Path


def record(link):
    manifest = os.environ.get('BENCH_LINK_MANIFEST')
    if not manifest:
        return
    link = Path(link).absolute()
    stat = link.lstat()
    entry = {'target': os.readlink(link), 'device': stat.st_dev, 'inode': stat.st_ino}
    path = Path(manifest)
    entries = json.loads(path.read_text()) if path.exists() else {}
    entries[str(link)] = entry
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(entries, indent=2))
    os.replace(temporary, path)


def cleanup(links, manifest):
    try:
        entries = json.loads(Path(manifest).read_text()) if Path(manifest).exists() else {}
    except Exception as exc:
        return ['Tool link manifest unreadable: ' + str(exc)]
    errors = []
    for link in map(Path, links):
        try:
            if not link.exists() and not link.is_symlink():
                continue
            stat = link.lstat()
            actual = {'target': os.readlink(link), 'device': stat.st_dev, 'inode': stat.st_ino} if link.is_symlink() else None
            if actual is None or entries.get(str(link.absolute())) != actual:
                raise ValueError('Tool link ownership changed: ' + str(link))
            link.unlink()
        except Exception as exc:
            errors.append(str(exc))
    return errors
