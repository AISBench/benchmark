"""Explicit, deterministic container mount profiles; no local filesystem probing."""
PROFILES = {
    'ascend-runtime': ('/usr/local/Ascend/driver', '/usr/local/dcmi'),
    'ascend-hixl': ('/etc/hixlep',),
}


def resolve_mounts(binding):
    explicit = [dict(m) for m in binding.get('mounts', [])]
    targets = [m['target'] for m in explicit]
    if len(set(targets)) != len(targets):
        raise ValueError('Duplicate mount targets')
    result = []
    for profile in binding.get('mount_profiles', []):
        if profile not in PROFILES:
            raise ValueError('Unknown mount profile: ' + profile)
        for path in PROFILES[profile]:
            if path not in targets and not any(m['target'] == path for m in result):
                result.append(dict(source=path, target=path, read_only=True))
    return result + explicit
