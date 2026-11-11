"""Linux receiver: publish a verified bundle under an exclusive file lock."""

PROGRAM = r'''
import base64, fcntl, hashlib, json, pathlib, sys, tempfile
p = json.load(sys.stdin)
root = pathlib.Path(p['root'])
files, hashes = p['files'], p['hashes']
root.parent.mkdir(parents=True, exist_ok=True)
def verified(path):
    return all((path/name).is_file() and
               hashlib.sha256((path/name).read_bytes()).hexdigest() == digest
               for name, digest in hashes.items())
# Keep this inode stable across receivers. Never unlink a shared lock file.
with (root.parent/('.'+root.name+'.publish.lock')).open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if root.exists():
        if not verified(root):
            raise SystemExit('Existing task directory differs; retain evidence and use a new job ID')
    else:
        with tempfile.TemporaryDirectory(prefix='.'+root.name+'.upload-', dir=root.parent) as temp:
            staged = pathlib.Path(temp)/'bundle'
            staged.mkdir()
            for name, data in files.items():
                relative = pathlib.PurePosixPath(name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise SystemExit('Invalid bundle member path')
                path = staged/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(base64.b64decode(data, validate=True))
            if not verified(staged):
                raise SystemExit('Uploaded bundle digest mismatch')
            staged.rename(root)
print('Bundle verified')
'''
