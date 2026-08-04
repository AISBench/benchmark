"""Generate msProbe response anomaly model configs from a local model.

msProbe's official ``gen_model_config.py`` always writes into the installed
msProbe package (``response_anomaly/configs`` and ``response_anomaly/token2category``)
and overwrites ``mtype_config.json`` on every run. This wrapper runs the
official script inside ``<output_dir>/tools`` so its cwd-relative outputs land
in ``<output_dir>/configs`` and ``<output_dir>/token2category`` instead, merges
new model entries into an existing ``mtype_config.json``, and copies the default
algorithm-threshold ``config.yaml`` when it is not already present.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def _normalize_name(name: str) -> str:
    """Keep in sync with msProbe's model-name normalization rules."""
    return "-".join(re.split(r"\.|-|_", name.lower()))


def _msprobe_response_anomaly_dir() -> Path:
    try:
        import msprobe.response_anomaly as response_anomaly
    except ImportError as exc:
        raise RuntimeError(
            "mindstudio-probe is required to generate response anomaly model "
            "configs. Install the AISBench response_anomaly extra first."
        ) from exc
    return Path(response_anomaly.__file__).resolve().parent


def _official_script_path() -> Path:
    script = _msprobe_response_anomaly_dir() / "tools" / "gen_model_config.py"
    if not script.exists():
        raise RuntimeError(
            f"msProbe response anomaly generator not found at {script}. "
            "Please reinstall the pinned mindstudio-probe version."
        )
    return script


def generate_model_config(
    model_path: str,
    model_name: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Generate msProbe model files into a user-owned directory.

    Args:
        model_path: Local model/tokenizer directory.
        model_name: msProbe model name; defaults to the directory basename and
            is normalized by msProbe (lowercase, ``-_.`` -> ``-``).
        output_dir: Destination directory. Generated layout:
            ``<output_dir>/configs/config.yaml``,
            ``<output_dir>/configs/mtype_config.json``,
            ``<output_dir>/token2category/<model>_<vocab>.json``.

    Returns:
        Dict with ``msprobe_config_path``, ``msprobe_mtype_path`` and
        ``msprobe_token2category_dir``.
    """
    output_dir = Path(output_dir or "msprobe_configs").resolve()
    script = _official_script_path()
    effective_model_name = _normalize_name(
        model_name or Path(model_path).name
    )

    configs_dir = output_dir / "configs"
    token2category_dir = output_dir / "token2category"
    configs_dir.mkdir(parents=True, exist_ok=True)
    token2category_dir.mkdir(parents=True, exist_ok=True)

    mtype_path = configs_dir / "mtype_config.json"
    existing_mtype = {}
    if mtype_path.exists():
        existing_mtype = json.loads(mtype_path.read_text(encoding="utf-8"))

    # The official script derives its output paths from the current working
    # directory (".."), so run it from <output_dir>/tools to redirect outputs.
    tools_dir = output_dir / "tools"
    tools_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(script), "--model-path", str(model_path)]
    if model_name:
        command += ["--model-name", model_name]
    try:
        proc = subprocess.run(
            command,
            cwd=str(tools_dir),
            text=True,
            capture_output=True,
        )
    except Exception as exc:
        # Keep the subprocess workspace so failures can be inspected.
        raise RuntimeError(
            f"msProbe gen_model_config failed to start: {exc}. "
            f"Inspection files are kept at {tools_dir}."
        ) from exc

    if proc.returncode != 0:
        raise RuntimeError(
            "msProbe gen_model_config failed "
            f"(return code {proc.returncode}): {proc.stderr or proc.stdout}. "
            f"Inspection files are kept at {tools_dir}."
        )
    shutil.rmtree(tools_dir, ignore_errors=True)

    generated_mtype = {}
    if mtype_path.exists():
        generated_mtype = json.loads(mtype_path.read_text(encoding="utf-8"))
    merged_mtype = {**existing_mtype, **generated_mtype}
    mtype_path.write_text(
        json.dumps(merged_mtype, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config_yaml = configs_dir / "config.yaml"
    if not config_yaml.exists():
        default_yaml = _msprobe_response_anomaly_dir() / "configs" / "config.yaml"
        shutil.copy2(default_yaml, config_yaml)

    return {
        "model_name": effective_model_name,
        "msprobe_config_path": str(config_yaml),
        "msprobe_mtype_path": str(mtype_path),
        "msprobe_token2category_dir": str(token2category_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate msProbe response anomaly configs for a local model."
    )
    parser.add_argument("--model-path", required=True, help="Local model directory.")
    parser.add_argument(
        "--model-name",
        default=None,
        help="msProbe model name; defaults to the directory basename.",
    )
    parser.add_argument(
        "--output-dir",
        default="msprobe_configs",
        help="Output directory for configs/ and token2category/.",
    )
    args = parser.parse_args()

    generated = generate_model_config(
        model_path=args.model_path,
        model_name=args.model_name,
        output_dir=args.output_dir,
    )
    for key, path in generated.items():
        print(f"{key}={path}")


if __name__ == "__main__":
    main()
