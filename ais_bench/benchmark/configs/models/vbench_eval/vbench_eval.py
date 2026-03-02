# Placeholder model config for VBench 1.0 eval-only.
# Usage: ais_bench --mode eval --models vbench_eval --datasets vbench_standard
# No real model is loaded; this is only for task naming and result paths.

models = [
    dict(
        attr='local',
        type='VBenchEvalPlaceholder',  # placeholder, not built in eval
        abbr='vbench_eval',
    )
]
