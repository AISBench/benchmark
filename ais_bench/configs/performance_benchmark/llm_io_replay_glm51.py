"""Replay a complete llm_io log with the core LLMIOReplayTask.

Run with ``ais_bench <this-file> --mode infer``. ``LLMIOReplayTask`` reads
``AISBENCH_REPLAY_*`` environment overrides at runtime, independently of
MMEngine's lazy configuration parsing.
"""


datasets = [
    dict(
        abbr="llm-io-replay",
        # The args marker tells AISBench this custom task owns data loading.
        args=dict(
            input_log_file=(
                "s9fkf_input_14-25round_150users_2500_redacted.txt"
            ),
            # Zero sends every loaded record once.
            requests=0,
            repair_redacted=True,
            on_error="raise",
            # Zero means no loading cap.
            max_records=0,
        ),
    )
]


models = [
    dict(
        attr="service",
        type="LLMIOReplayService",
        abbr="glm51-llm-io-replay",
        path="",
        model="glm51",
        url="http://172.27.13.87:8900/v1/chat/completions",
        concurrent=100,
        timeout=900,
        mode="stream",
        stream=True,
        temperature=0.7,
        x_app_id="1111",
        x_app_key="22222",
        send_legacy_app_headers=True,
        add_timestamp_prefix=True,
    )
]


infer = dict(
    partitioner=dict(type="ais_bench.benchmark.partitioners.NaivePartitioner"),
    runner=dict(
        type="ais_bench.benchmark.runners.local.LocalRunner",
        max_num_workers=1,
        task=dict(
            type=(
                "ais_bench.benchmark.tasks.llm_io_replay."
                "LLMIOReplayTask"
            )
        ),
    ),
)

work_dir = "outputs/llm_io_replay/"
