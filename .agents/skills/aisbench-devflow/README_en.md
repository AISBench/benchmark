# aisbench-devflow quick start

[中文](README.md) | English

## Purpose and boundaries

Turn the AISBench loop — issue, implementation, pull request, CI, review closure — into a resumable, evidence-backed workflow, and move the merge gates that **can** be decided locally in front of the push, shortening a 20–40 minute round trip.

Out of scope: merging, approving, signing DCO/CLA on the user's behalf, creating issues, editing the PR template or label workflow, reproducing NPU evaluations, and running the full test suite locally.

## What it buys you

| Situation | Without the skill | With the skill |
|---|---|---|
| Gate check before pushing | wait 20–40 min for a self-hosted runner | verdict in seconds, locally |
| Coverage | discover the failure in CI | get the test directory and suggested file path |
| PR labels | box ticked, label missing, cause unknown | names the template/workflow mismatch |
| Bilingual docs | English counterpart missing, rejected later | reported before pushing |
| DCO | forgotten `-s` | lists the commits missing sign-off |

## Requirements

Git and Python 3.10+ (standard library only at runtime). Remote reads/writes need an authenticated GitHub CLI (`gh`) or host-provided GitHub tooling.

## Quick start

```bash
python .agents/skills/aisbench-devflow/scripts/gate_check.py --profile

python .agents/skills/aisbench-devflow/scripts/gate_check.py \
  --base origin/master --pr-type feature --issue 651 \
  --title "[feat][datasets]support X" --pr-body-file /tmp/pr-body.md --out /tmp/gates.json

python .agents/skills/aisbench-devflow/scripts/ut_map.py suggest \
  --changed ais_bench/benchmark/datasets/hle.py

python .agents/skills/aisbench-devflow/scripts/pr_body.py render \
  --template .github/pull_request_template.md \
  --workflow .github/workflows/auto_label_pr.yml \
  --pr-type feature --issue 651 --sections sections.json --out /tmp/pr-body.md

python .agents/skills/aisbench-devflow/scripts/pr_body.py check \
  --body-file /tmp/pr-body.md --workflow .github/workflows/auto_label_pr.yml --issue 651
```

`--sections` takes JSON keyed by a heading fragment: `{"Motivation": "...", "Modification": "..."}`.

## Verdict semantics

| Value | Meaning |
|---|---|
| `PASS` | evidence shows the gate holds |
| `FAIL` | evidence shows it does not, with a `remedy` |
| `UNKNOWN` | not decidable locally (remote reviews, thread state, full coverage, external builds, self-hosted smoke) |
| `NOT_APPLICABLE` | does not apply, with the reason |

Exit code `0` means "no known FAIL" only. Promotion additionally requires an empty `promotion_blockers` list. `PASS` is never used for "not checked".

## Known limits

1. Coverage is scoped: no dependency installation, no full run; results carry `scope=subset` and the full gate comes from CI.
2. Documentation **content** parity is not automatic: the script checks path pairing and reports content sync as `UNKNOWN`.
3. Merging is never executed; `merge` only prints a `ready-to-merge` checklist.
4. The existing CI/CD and Dependency label mismatches are reported, never patched in place.
5. Self-hosted runner and NPU smoke differences cannot be reproduced locally: always `UNKNOWN`.

## Layout

```
aisbench-devflow/
├── SKILL.md              # rules: action routing, contracts, state machine
├── README.md / README_en.md
├── references/           # per-action details, loaded on demand (bilingual single files)
├── scripts/              # judge only, never modify
└── tests/                # unittest, real temporary git repositories
```

## Division of labour

`pr-checklist` owns norm checks, `ci-failure-triage` owns read-only CI diagnosis, and this skill owns the repair after diagnosis plus the write path itself.

## Development

```bash
python tests/run_unit_tests.py --output /tmp/devflow-tests.json
python -m ruff check --config ruff.toml scripts tests   # needs ruff==0.17.4
```
