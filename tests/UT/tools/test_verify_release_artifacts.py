import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "tools" / "verify_release_artifacts.py"
SPEC = importlib.util.spec_from_file_location("verify_release_artifacts", MODULE_PATH)
verify_release_artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify_release_artifacts)


def _write_wheel(path, members):
    with zipfile.ZipFile(path, mode="w") as archive:
        for member in members:
            archive.writestr(member, "# packaged resource\n")


def _write_sdist(path, members):
    with tarfile.open(path, mode="w:gz") as archive:
        for member in members:
            content = b"# packaged resource\n"
            info = tarfile.TarInfo(
                f"ais_bench_benchmark-3.1.0/{member}"
            )
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_verify_wheel_with_cmmlu_resources(tmp_path):
    wheel = tmp_path / "ais_bench_benchmark-3.1.0-py3-none-any.whl"
    _write_wheel(wheel, verify_release_artifacts.REQUIRED_CMMLU_PATHS)

    verify_release_artifacts.verify_release_artifact(wheel)


def test_verify_sdist_with_cmmlu_resources(tmp_path):
    sdist = tmp_path / "ais_bench_benchmark-3.1.0.tar.gz"
    _write_sdist(sdist, verify_release_artifacts.REQUIRED_CMMLU_PATHS)

    verify_release_artifacts.verify_release_artifact(sdist)


def test_verify_reports_missing_cmmlu_resource(tmp_path):
    wheel = tmp_path / "ais_bench_benchmark-3.1.0-py3-none-any.whl"
    missing = "ais_bench/benchmark/datasets/cmmlu.py"
    members = verify_release_artifacts.REQUIRED_CMMLU_PATHS - {missing}
    _write_wheel(wheel, members)

    with pytest.raises(
        verify_release_artifacts.ReleaseArtifactError,
        match="datasets/cmmlu.py",
    ):
        verify_release_artifacts.verify_release_artifact(wheel)


def test_verify_rejects_unsupported_artifact(tmp_path):
    artifact = tmp_path / "ais_bench_benchmark.zip"
    artifact.write_bytes(b"not a release archive")

    with pytest.raises(
        verify_release_artifacts.ReleaseArtifactError,
        match="Unsupported release artifact",
    ):
        verify_release_artifacts.verify_release_artifact(artifact)
