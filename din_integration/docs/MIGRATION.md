# Migration Guide: old scripts → `swebench-dind`

The old workflow lived under `/home/zengziyu/mini_matrix/scripts/` and
`/home/zengziyu/mini_matrix/jobs/launch_*.sh`. They still work, but the
`swebench-dind` CLI is now the canonical entry point and the only one we
add features to.

This page is a 1-to-1 mapping from each old script to the equivalent
`git`-able command.

## Top-level scripts (mini_matrix/scripts/)

| Old script | Equivalent CLI |
|---|---|
| `scripts/start_orchestrator.sh` | `swebench-dind orchestrator start` |
| `scripts/start_orchestrator.sh --recreate` | `swebench-dind orchestrator start --recreate` |
| `scripts/stop_orchestrator.sh` | `swebench-dind orchestrator stop` |
| `scripts/stop_orchestrator.sh rm` | `swebench-dind orchestrator stop --remove` |
| `scripts/summarize.py` | `swebench-dind summarize` |
| `scripts/summarize.sh` | `swebench-dind summarize` |
| `scripts/simple_summary.py` | `swebench-dind summarize` (md/csv/json in one pass) |
| `scripts/show_results.py` | `swebench-dind summarize --include <pattern>` |
| `scripts/filter_matrix.py` | `swebench-dind summarize --include <csv-pattern>` |
| `scripts/patch_qwen_code.py` | `swebench-dind patch harbor --agent qwen-code` |
| `scripts/monitor_3x3.sh` | `swebench-dind watch <job-name>` (per-trial) or loop `watch` over `launch 3x3` |
| `scripts/build_baked_v2.sh` | `swebench-dind build all` |
| `scripts/build_baked_images.sh` | `swebench-dind build l1 --case ...` + `build l2 --agent ...` |
| `scripts/run_3x3_matrix.sh` | `swebench-dind launch 3x3 --wait` |
| `scripts/run_matrix.sh` | `swebench-dind launch matrix --case ... --agent ...` |

## Per-agent launchers (mini_matrix/jobs/)

These were the most duplicated scripts — one per agent, each repeating the
`COMMON_AE` / `AIDER_AE` / `MSWEA_AE` env blocks. Now there is one
command that handles every supported agent:

| Old script | Equivalent CLI |
|---|---|
| `jobs/launch_aider_only.sh` | `swebench-dind launch trial --case X --agent aider` |
| `jobs/launch_msa_only.sh` | `swebench-dind launch trial --case X --agent mini-swe-agent` |
| `jobs/launch_qwen.sh` | `swebench-dind launch trial --case X --agent qwen-code` |
| `jobs/launch_oh.sh` | `swebench-dind launch trial --case X --agent openhands-sdk` |
| `jobs/launch_3new.sh` | `swebench-dind launch matrix --case 10097 --case 10554 --case 10880 --agent aider` |
| `jobs/launch_v2_test.sh` | `swebench-dind launch trial --case 11099 --agent aider --job-name v2-test-11099` |

If you want to reproduce the exact `docker exec` command line that the
old scripts used, pass `--help` to `launch trial` and add any extra
`--ae` flags you had (the CLI already sets `OPENAI_API_BASE`,
`OPENAI_API_KEY`, `AIDER_API_KEY`, etc. for you).

## Hardcoded values that used to live in many places

| Was scattered across | Now in |
|---|---|
| `COMMON_AE_KEYS` (6 launchers) | `config.COMMON_AE_KEYS` |
| `AIDER_AE`, `MSWEA_AE`, `QWEN_AE`, `OH_AE` | `config.AGENT_AE` (one dict, all agents) |
| `--agent qwen-coder` (Harbor flag) | `config.HARBOR_AGENT_FLAG["qwen-code"] = "qwen-coder"` |
| Image tags (`swebench/django-{case}-with-{agent}:latest`) | `config.l2_image_tag()` |
| Model / API base / API key path | `config.DEFAULT_MODEL`, `DEFAULT_API_BASE`, `API_KEY_ENV` |
| Timeout multipliers (`--agent-timeout-multiplier 4`) | `config.DEFAULT_MULTIPLIERS` |
| Task directory naming (`django__django-11099-aider`) | `config.task_dir_name()` |

To add a new case: append to `DEFAULT_CASES` (or `NEW_CASES`). To add a
new agent: add entries to `AGENT_AE`, `HARBOR_AGENT_FLAG`,
`DOCKERFILE_L2_BY_AGENT`, and (if it ships as a Harbor module) the
`patcher.AGENT_TO_MODULE` / `patcher.AGENT_CLI` dicts.

## Status / health checks

| Old habit | Equivalent CLI |
|---|---|
| `docker ps \| grep swebench-orchestrator` | `swebench-dind orchestrator status` |
| `docker exec swebench-orchestrator docker info` | `swebench-dind orchestrator status` |
| `tail -f jobs/<name>/verifier.log` (manual polling) | `swebench-dind watch <job-name>` |

## Removing the old scripts

The plan is to keep the old scripts on disk for now (they have many
small in-place tweaks over months of debugging). If you want to clean
them up after switching:

1. Verify the CLI covers all the commands you actually run by reading
   through `docs/CLI-USAGE.md`.
2. For any one-off script that isn't on the mapping table, file a
   follow-up task; we don't want to silently drop functionality.
3. Add `# DEPRECATED: use swebench-dind <cmd>` headers to the surviving
   old scripts and remove them in a separate commit.

## Why we built the CLI

After 12 trials (9 PASS / 75%) the workflow had stabilized into four
repeating steps. Each new case or agent meant copy-pasting a launch
script, mutating four env-var blocks, and re-reading every shell file
to find the right multiplier. The CLI collapses that into:

```bash
swebench-dind build l2 --agent <new-agent> --case <new-case>
swebench-dind launch trial --case <new-case> --agent <new-agent>
```

with the model, API base, and timeout multipliers all coming from
`config.py`. New (case, agent) combinations stop being a copy-paste
exercise.