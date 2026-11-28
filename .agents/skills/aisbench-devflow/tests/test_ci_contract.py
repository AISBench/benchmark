"""Contract test: gates we cannot decide locally must never be reported as PASS.

The fixture records the check names observed on a real upstream pull request.
It is metadata only (no diffs, no tokens) and is used to keep the local report
honest about its own coverage.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pr-653-checks.json"

REMOTE_ONLY_GATES = ("coverage_scope", "review_present", "threads_resolved", "rtd_and_smoke")


class CiContract(unittest.TestCase):
    def setUp(self) -> None:
        self.data = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_fixture_is_metadata_only(self) -> None:
        self.assertEqual(self.data["schema"], 1)
        self.assertTrue(self.data["checks"])
        for check in self.data["checks"]:
            self.assertEqual(set(check) - {"name", "conclusion", "decidable_locally", "local_gates"}, set())

    def test_remote_only_gates_are_backed_by_some_check(self) -> None:
        referenced = {gate for check in self.data["checks"] for gate in check["local_gates"]}
        for gate in REMOTE_ONLY_GATES:
            self.assertIn(gate, referenced, f"{gate} 未关联任何真实 CI 检查")

    def test_remote_only_gates_are_never_declared_locally_decidable(self) -> None:
        for check in self.data["checks"]:
            if check["decidable_locally"] != "yes":
                continue
            for gate in check["local_gates"]:
                self.assertNotIn(
                    gate,
                    REMOTE_ONLY_GATES,
                    f"{check['name']} 被声明为完全本地可判，但包含只能在远端判定的门禁 {gate}",
                )

    def test_no_gate_claims_full_local_decidability_for_remote_checks(self) -> None:
        for check in self.data["checks"]:
            if check["decidable_locally"] != "no":
                continue
            self.assertTrue(
                set(check["local_gates"]).issubset(set(REMOTE_ONLY_GATES)),
                f"{check['name']} 只能在远端判定，但被映射到本地可判定门禁",
            )


if __name__ == "__main__":
    unittest.main()
