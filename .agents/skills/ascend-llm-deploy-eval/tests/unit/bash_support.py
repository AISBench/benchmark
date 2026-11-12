"""Shared executable Bash probe; an installed WSL launcher is not sufficient."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest


def require_bash():
    candidates = []
    if os.name == 'nt':
        git_bash = Path('C:/Program Files/Git/bin/bash.exe')
        if git_bash.is_file():
            candidates.append(str(git_bash))
    found = shutil.which('bash')
    if found:
        candidates.append(found)
    reasons = []
    for candidate in dict.fromkeys(candidates):
        try:
            result = subprocess.run([candidate, '-c', 'printf "%s\\n" "$BASH_VERSION"'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and re.fullmatch(r'\d+\.\d+[^\r\n]*', result.stdout.strip()):
                return candidate
            reasons.append(f'{candidate}: exit {result.returncode}, no usable Bash version')
        except (OSError, subprocess.TimeoutExpired) as error:
            reasons.append(f'{candidate}: {type(error).__name__}')
    raise unittest.SkipTest('Bash unavailable: '+('; '.join(reasons) or 'no executable found'))
