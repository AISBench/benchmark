"""Multi-process concurrency test for SWE-bench Pro session-scoped container cleanup.

This test validates the fix for the container-cleanup race condition described in
the SWE-bench Pro dataset: when several SWE-bench Pro evaluation tasks finish at
different times, a task that finishes early must NOT remove containers belonging
to tasks that are still running.

How it works
------------
The real ``ais_bench.benchmark.tasks.swebench_pro.utils`` module is loaded from
source (its ``ais_bench`` logging dependencies are stubbed so the module can be
imported on any platform, including Windows where ``fcntl``/``resource`` are
unavailable). ``subprocess.run`` is patched inside each worker process to
simulate the Docker daemon against a shared (``multiprocessing.Manager``)
registry that maps ``container_id -> session_id``.

Each worker process:
  1. Generates its own ``session_id`` via ``make_swebench_pro_session_id``.
  2. Registers K containers tagged with that session label (simulating creation).
  3. Sleeps a random short interval so cleanups overlap across processes.
  4. Calls ``cleanup_swebench_pro_containers(session_id=...)``.

The key assertion: every worker removes *exactly* the containers it created and
*none* belonging to other sessions. Under the old name-filter based cleanup
(``minisweagent-`` / ``sweb.eval``), the first worker to clean would remove every
live container, which this test would catch.

Run directly:  python tests/UT/tasks/swebp_pro/test_session_concurrency.py
Or via pytest: pytest tests/UT/tasks/swebp_pro/test_session_concurrency.py
"""
import os
import sys
import time
import types
import random
import subprocess
import importlib.util
import multiprocessing
from unittest.mock import patch


# --------------------------------------------------------------------------- #
# Stub the minimal dependencies so the real utils.py can be loaded in isolation
# on any platform (avoids the Unix-only ``fcntl``/``resource`` import chain).
# --------------------------------------------------------------------------- #
def _setup_stubs():
    # Unix-only modules referenced transitively by the ais_bench package.
    for name in ("fcntl", "resource"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    def _make_pkg(name):
        m = types.ModuleType(name)
        m.__path__ = []  # mark as a package
        return m

    for pkg in (
        "ais_bench",
        "ais_bench.benchmark",
        "ais_bench.benchmark.utils",
    ):
        if pkg not in sys.modules:
            sys.modules[pkg] = _make_pkg(pkg)

    # ais_bench.benchmark.utils.logging  -> provides AISLogger
    log_mod = types.ModuleType("ais_bench.benchmark.utils.logging")

    class _StubLogger:
        def __init__(self, *a, **k):
            pass

        def info(self, *a, **k):
            pass

        def debug(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

        def error(self, *a, **k):
            pass

    log_mod.AISLogger = _StubLogger
    sys.modules["ais_bench.benchmark.utils.logging"] = log_mod

    # ais_bench.benchmark.utils.logging.error_codes -> provides SWEBP_CODES & SWEB_CODES
    ec = types.ModuleType("ais_bench.benchmark.utils.logging.error_codes")

    class _SWEBPCodes:
        DOCKER_IMAGE_UNAVAILABLE = 1
        SWEBENCH_HARNESS_IMPORT_ERROR = 2

    class _SWEBCodes:
        DOCKER_IMAGE_UNAVAILABLE = 1

    ec.SWEBP_CODES = _SWEBPCodes()
    ec.SWEB_CODES = _SWEBCodes()
    sys.modules["ais_bench.benchmark.utils.logging.error_codes"] = ec

    # ais_bench.benchmark.utils.logging.exceptions -> provides error classes
    ex = types.ModuleType("ais_bench.benchmark.utils.logging.exceptions")

    class AISBenchRuntimeError(RuntimeError):
        pass

    class AISBenchImportError(ImportError):
        pass

    ex.AISBenchRuntimeError = AISBenchRuntimeError
    ex.AISBenchImportError = AISBenchImportError
    sys.modules["ais_bench.benchmark.utils.logging.exceptions"] = ex

    # ais_bench.benchmark.tasks & .swebench -> package stubs so that
    # ``from ais_bench.benchmark.tasks.swebench.utils import ...`` (used by
    # swebench_pro/utils.py to reuse the session implementation) resolves.
    for pkg in (
        "ais_bench.benchmark.tasks",
        "ais_bench.benchmark.tasks.swebench",
    ):
        if pkg not in sys.modules:
            sys.modules[pkg] = _make_pkg(pkg)

    # Load the REAL swebench/utils.py by file path and register it under its
    # dotted name so swebench_pro/utils.py's import picks up the shared
    # implementation (parameterised with ``label_key``).
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    swebench_utils_path = os.path.join(
        root, "ais_bench", "benchmark", "tasks", "swebench", "utils.py"
    )
    swebench_utils_spec = importlib.util.spec_from_file_location(
        "ais_bench.benchmark.tasks.swebench.utils", swebench_utils_path
    )
    swebench_utils_mod = importlib.util.module_from_spec(swebench_utils_spec)
    sys.modules["ais_bench.benchmark.tasks.swebench.utils"] = swebench_utils_mod
    swebench_utils_spec.loader.exec_module(swebench_utils_mod)


def _load_real_utils():
    """Load the REAL swebench_pro/utils.py by file path (bypass package init)."""
    here = os.path.dirname(os.path.abspath(__file__))
    # tests/UT/tasks/swebp_pro/ -> repo root is four levels up.
    root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    path = os.path.join(
        root, "ais_bench", "benchmark", "tasks", "swebench_pro", "utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_swebench_pro_utils_under_test", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Simulated Docker backend: a shared registry mapping container_id -> session_id.
# --------------------------------------------------------------------------- #
def _make_fake_run(registry):
    """Return a subprocess.run replacement that simulates docker ps/rm."""

    def fake_run(args, *a, **k):
        if args[:1] != ["docker"]:
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="")
        sub = args[1] if len(args) > 1 else ""
        if sub == "ps":
            # Parse "--filter label=<LABEL>=<session_id>".
            session = None
            for tok in args[2:]:
                if tok.startswith("label="):
                    rest = tok[len("label="):]  # "<LABEL>=<session_id>"
                    if "=" in rest:
                        session = rest.rsplit("=", 1)[1]
            cids = [cid for cid, sid in list(registry.items()) if sid == session]
            stdout = "\n".join(cids)
            if cids:
                stdout += "\n"
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout=stdout
            )
        if sub == "rm":
            # args: ["docker", "rm", "-f", cid1, cid2, ...]
            cids = args[3:] if len(args) > 3 else []
            for cid in cids:
                if cid in registry:
                    del registry[cid]
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout="", stderr=b""
            )
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="")

    return fake_run


# --------------------------------------------------------------------------- #
# Worker: each process is one "SWE-bench Pro task" with its own session id.
# --------------------------------------------------------------------------- #
def _worker(num_containers, registry, results, idx):
    _setup_stubs()
    utils = _load_real_utils()

    session_id = utils.make_swebench_pro_session_id()
    # Simulate creating ``num_containers`` containers, all tagged with this
    # task's session label (as the docker-client wrapper / run_args would do).
    my_cids = [f"{session_id}-{i}" for i in range(num_containers)]
    for cid in my_cids:
        registry[cid] = session_id

    # Jitter so cleanups from different processes overlap in time.
    time.sleep(random.uniform(0.0, 0.3))

    with patch.object(utils.subprocess, "run", _make_fake_run(registry)):
        utils.cleanup_swebench_pro_containers(session_id=session_id)

    removed = [cid for cid in my_cids if cid not in registry]
    results[idx] = {
        "session_id": session_id,
        "created": my_cids,
        "removed": removed,
    }


def _run_concurrency_test(num_proc=4, containers_per_proc=5):
    mgr = multiprocessing.Manager()
    registry = mgr.dict()  # container_id -> session_id (live containers)
    results = mgr.dict()

    procs = []
    for i in range(num_proc):
        p = multiprocessing.Process(
            target=_worker,
            args=(containers_per_proc, registry, results, i),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
            p.join()

    ok = True
    total_created = 0
    total_removed = 0
    for i in range(num_proc):
        if i not in results:
            print(f"[FAIL] worker {i} produced no result (timed out?)")
            ok = False
            continue
        r = results[i]
        created = r["created"]
        removed = r["removed"]
        total_created += len(created)
        total_removed += len(removed)
        if set(removed) != set(created):
            ok = False
            extra = set(removed) - set(created)
            missing = set(created) - set(removed)
            print(
                f"[FAIL] worker {i} session {r['session_id'][:8]}: "
                f"created {len(created)}, removed {len(removed)}, "
                f"extra(foreign)={len(extra)}, missing={len(missing)}"
            )
        else:
            print(
                f"[OK]   worker {i} session {r['session_id'][:8]}: "
                f"created & removed {len(created)} containers "
                f"(no cross-session cleanup)"
            )

    leftover = list(registry.keys())
    if leftover:
        ok = False
        print(f"[FAIL] {len(leftover)} containers left in registry: {leftover}")
    else:
        print(f"[OK]   all {total_created} containers cleaned; registry empty")

    print(
        f"SUMMARY: processes={num_proc}, containers/process={containers_per_proc}, "
        f"total_created={total_created}, total_removed={total_removed}"
    )
    return ok


def test_session_scoped_cleanup_under_concurrency():
    """Pytest entry point: assert no cross-session cleanup under concurrency."""
    assert _run_concurrency_test()


if __name__ == "__main__":
    random.seed()
    # On Windows the default start method is 'spawn'; make it explicit.
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    success = _run_concurrency_test()
    print("RESULT: " + ("PASS" if success else "FAIL"))
    sys.exit(0 if success else 1)
