"""Replay a complete llm_io log with AISBench performance reporting.

Run with ``ais_bench <this-file> --mode perf``. All replay values are configured
directly below; AISBench's generic CLI behavior remains unchanged.
"""


datasets = [
    dict(
        abbr="llm-io-replay",
        # These markers let the standard perf workflow group this custom
        # dataset without constructing an OpenICL dataset or inferencer.
        type="LLMIOReplayDataset",
        infer_cfg=dict(inferencer=dict(type="LLMIOReplayInferencer")),
        # The args marker tells AISBench this custom task owns data loading.
        args=dict(
            input_log_file=(
                "/data/s9fkf_input_14-25round_150users_2500_redacted_fix.txt"
            ),
            # Set to zero to send every loaded record once.
            requests=2500,
            repair_redacted=False,
            on_error="raise",
            # Zero means no loading cap.
            max_records=0,
        ),
    )
]


summarizer = dict(
    attr="performance",
    type=(
        "ais_bench.benchmark.tasks.llm_io_replay."
        "LLMIOReplayPerfSummarizer"
    ),
    # Prevent PerfViz from injecting the tokenizer-dependent default
    # calculator; replay already records and summarizes token metrics.
    calculator={},
)


models = [
    dict(
        attr="service",
        type="LLMIOReplayService",
        abbr="llm-io-replay",
        path="",
        model="",
        url="http://127.0.0.1:8000/v1/chat/completions",
        concurrent=100,
        timeout=900,
        mode="stream",
        stream=True,
        temperature=0.7,
        # Set an integer to override every source payload, e.g. 4096.
        # None preserves each replay record's max_tokens value.
        max_tokens=None,
        # Set True/False to send vLLM's ignore_eos extension explicitly.
        # None preserves the source value or omits the field when absent.
        ignore_eos=None,
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
