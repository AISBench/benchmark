"""Task-scoped immutable datasets with content verification and atomic publication."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()

def cached(root, parameters, model, generate):
    model = Path(model)
    tokenizer = {p.name:sha(p) for p in sorted(model.iterdir())
                 if p.is_file() and (p.suffix in ('.json', '.model', '.txt', '.py', '.tiktoken')
                                    and 'safetensors' not in p.name)}
    sources = {name:sha(Path(__file__).parent/name) for name in
               ('generate_dataset.py', 'data_picker.py', 'GSM8K.jsonl', 'dataset_cache.py')}
    key = hashlib.sha256(json.dumps([parameters, tokenizer, sources], sort_keys=True).encode()).hexdigest()
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    target = root/key
    # Linux execution nodes: serialize identical generation across deployments.
    import fcntl
    with (root/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            temporary = Path(tempfile.mkdtemp(prefix=key+'.', dir=root))
            try:
                paths = generate(str(temporary))
                names = [Path(p).name if p else '' for p in paths]
                files = {name:sha(temporary/name) for name in names if name}
                (temporary/'manifest.json').write_text(json.dumps(dict(parameters=parameters, files=files, paths=names)))
                for name in files: (temporary/name).chmod(0o444)
                os.replace(temporary, target)
            finally:
                if temporary.exists(): shutil.rmtree(temporary)
        manifest = json.loads((target/'manifest.json').read_text())
        for name, expected in manifest['files'].items():
            if sha(target/name) != expected:
                raise ValueError('Dataset cache content changed: '+str(target/name))
        return tuple(str(target/name) if name else '' for name in manifest['paths'])
