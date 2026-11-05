# VABench T2AV (GPU, eval-only)

For the end-to-end Chinese guide, official CLI comparison, and the selected
78-prompt subset, see `docs/source_zh_cn/extended_benchmark/lmm_generate/vabench.md`.
`mini10pct.json` pins that project-selected 78-prompt subset with provenance;
it contains 78 prompts from the pinned 778. The guide exports complete
metadata from the pinned upstream source and checks all 78 videos explicitly.
Section 3 of that guide documents the category quotas, per-category TF-IDF,
medoid/farthest-point selection, deterministic tie-breaking and an offline
reproduction check. It also distinguishes the pinned subset from the later
one-prompt replacement recorded in the project README.

Use an external pinned VABench checkout and one Conda Python 3.11 scoring
environment for all 15 dimensions. Keep the AISBench main environment separate.

From the AISBench source root, using the scoring Conda's interpreter:

```bash
python3 -m ais_bench.tools.vabench.environment install \
  --source-root /workspace/code/VABench-main \
  --cache-dir /workspace/vabench-cache \
  --work-dir /workspace/vabench-env-audit
```

This uses `requirements/datasets/vabench*.txt`, pinned Git revisions, and a
metadata-only NISQA wheel patch. Download/provide model weights separately.
Use `plan` instead of `install` for a read-only command preview.
On Ubuntu/Debian containers, root may add `--install-system-deps` to prepare
FFmpeg, git, libsndfile and the C/C++ build tools through apt before pip runs.
Without that explicit option, the tool does not invoke the system package manager.

Then edit the paths and `BACKEND_PYTHON` in `eval_vabench.py` (78/778 samples) or
`eval_vabench_smoke.py` (one CLAP sample). Follow
`docs/source_zh_cn/get_started/vabench_gpu_environment.md` for installation and
`docs/source_zh_cn/get_started/vabench_t2av.md` for media preparation/scoring.

Environment checks and mock-scoring tests do not establish real GPU score
alignment. The evaluator exports all 15 native means and fixed 0–100 display scores,
coverage, and a complete/preview/unavailable equal-weight total for the selected
samples. Scene-level error attribution and checkpoint resume remain separate work.
