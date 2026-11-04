from ais_bench.benchmark.models import test_type_for_replace

models = [
    dict(
        attr="service",
        type=test_type_for_replace,
        abbr='test_abbr_for_replace',
        path="model_path_for_replace",
        model="model_name_for_replace",
        api_key="api_key_for_replace",
        request_rate=rr_for_replace,
        retry=2,
        host_ip="ip_for_replace",
        host_port=port_for_replace,
        url="url_for_replace",
        max_out_len=outputlen_for_replace,
        batch_size=concurrency_for_replace,
        generation_kwargs=dict(
            temperature=0,
            ignore_eos=True,
        )
    )
]
