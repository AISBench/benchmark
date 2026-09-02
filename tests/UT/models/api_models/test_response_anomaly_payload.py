import unittest
from unittest.mock import patch

from ais_bench.benchmark.models import VLLMCustomAPI
from ais_bench.benchmark.models.api_models import base_api
from ais_bench.benchmark.models.output import Output


class TestResponseAnomalyPayload(unittest.TestCase):
    def setUp(self):
        self._get_service_model_path_patcher = patch.object(
            base_api.BaseAPIModel, "_get_service_model_path"
        )
        self.mock_get_model_path = self._get_service_model_path_patcher.start()
        self.mock_get_model_path.return_value = "mocked-model-path"
        self._get_url_patcher = patch.object(
            VLLMCustomAPI,
            "_get_url",
            return_value="http://localhost:8080/v1/completions",
        )
        self._get_url_patcher.start()

    def tearDown(self):
        self._get_url_patcher.stop()
        self._get_service_model_path_patcher.stop()

    def _make_model(self, enabled=True):
        generation_kwargs = {"temperature": 0.7}
        if enabled:
            generation_kwargs["response_anomaly_enabled"] = True
        return VLLMCustomAPI(
            path="test-model",
            model="test-model-name",
            generation_kwargs=generation_kwargs,
        )

    def test_enable_flag_is_consumed_and_not_forwarded(self):
        model = self._make_model(enabled=True)

        self.assertTrue(model.response_anomaly_enabled)
        self.assertNotIn("response_anomaly_enabled", model.generation_kwargs)

    def test_disabled_does_not_capture_payload(self):
        model = self._make_model(enabled=False)
        output = Output()

        model._record_response_anomaly_payload(
            {"token_ids": [1], "topk_logprobs": [{"1": -0.1}]}, output
        )

        self.assertNotIn("response_anomaly_payload", output.extra_details_data)

    def test_records_vllm_openai_logprobs_payload(self):
        model = self._make_model(enabled=True)
        output = Output()

        model._record_response_anomaly_payload(
            {
                "choices": [
                    {
                        "token_ids": [10, 11],
                        "logprobs": {
                            "content": [
                                {
                                    "token": "token_id:10",
                                    "logprob": -0.1,
                                    "top_logprobs": [
                                        {"token": "token_id:10", "logprob": -0.1},
                                        {"token": "token_id:12", "logprob": -1.2},
                                    ],
                                },
                                {
                                    "token": "token_id:11",
                                    "logprob": -0.2,
                                    "top_logprobs": [
                                        {"token": "token_id:11", "logprob": -0.2},
                                        {"token": "token_id:13", "logprob": -1.3},
                                    ],
                                },
                            ]
                        },
                    }
                ]
            },
            output,
        )

        payload = output.extra_details_data["response_anomaly_payload"]
        self.assertEqual(payload["tokens"], [10, 11])
        self.assertEqual(
            payload["topk_logprobs"],
            [{10: -0.1, 12: -1.2}, {11: -0.2, 13: -1.3}],
        )

    def test_vllm_openai_logprobs_can_supply_sampled_token_ids(self):
        model = self._make_model(enabled=True)
        output = Output()

        model._record_response_anomaly_payload(
            {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {
                                    "token": "token_id:21",
                                    "logprob": -0.1,
                                    "top_logprobs": [
                                        {"token": "token_id:21", "logprob": -0.1}
                                    ],
                                }
                            ]
                        }
                    }
                ]
            },
            output,
        )

        payload = output.extra_details_data["response_anomaly_payload"]
        self.assertEqual(payload["tokens"], [21])
        self.assertEqual(payload["topk_logprobs"], [{21: -0.1}])

    def test_stream_accumulates_vllm_openai_logprobs_chunks(self):
        model = self._make_model(enabled=True)
        output = Output()

        for token_id, logprob in ((31, -0.1), (32, -0.2)):
            model._accumulate_response_anomaly_payload(
                {
                    "choices": [
                        {
                            "token_ids": [token_id],
                            "logprobs": {
                                "content": [
                                    {
                                        "token": f"token_id:{token_id}",
                                        "logprob": logprob,
                                        "top_logprobs": [
                                            {
                                                "token": f"token_id:{token_id}",
                                                "logprob": logprob,
                                            }
                                        ],
                                    }
                                ]
                            },
                        }
                    ]
                },
                output,
            )

        payload = output.extra_details_data["response_anomaly_payload"]
        self.assertEqual(payload["tokens"], [31, 32])
        self.assertEqual(payload["topk_logprobs"], [{31: -0.1}, {32: -0.2}])

    def test_stream_accumulates_incremental_chunks(self):
        model = self._make_model(enabled=True)
        output = Output()

        model._accumulate_response_anomaly_payload(
            {"token_ids": [10], "topk_logprobs": [{"10": -0.1}]}, output
        )
        model._accumulate_response_anomaly_payload(
            {"token_ids": [11], "topk_logprobs": [{"11": -0.2}]}, output
        )

        payload = output.extra_details_data["response_anomaly_payload"]
        self.assertEqual(payload["tokens"], [10, 11])
        self.assertEqual(payload["topk_logprobs"], [{"10": -0.1}, {"11": -0.2}])

    def test_stream_full_token_ids_with_single_current_topk_appends_incrementally(self):
        model = self._make_model(enabled=True)
        output = Output()

        model._accumulate_response_anomaly_payload(
            {"token_ids": [10], "topk_logprobs": [{"10": -0.1}]}, output
        )
        model._accumulate_response_anomaly_payload(
            {
                "token_ids": [10, 11],
                "topk_logprobs": [{"11": -0.2}],
            },
            output,
        )
        model._accumulate_response_anomaly_payload(
            {
                "token_ids": [10, 11, 12],
                "topk_logprobs": [{"12": -0.3}],
            },
            output,
        )

        payload = output.extra_details_data["response_anomaly_payload"]
        self.assertEqual(payload["tokens"], [10, 11, 12])
        self.assertEqual(
            payload["topk_logprobs"],
            [{"10": -0.1}, {"11": -0.2}, {"12": -0.3}],
        )

    def test_stream_consecutive_identical_tokens_are_not_dropped(self):
        """两个连续相同 token 不应被全量快照覆盖。"""
        model = self._make_model(enabled=True)
        output = Output()

        model._accumulate_response_anomaly_payload(
            {"token_ids": [7], "topk_logprobs": [{"7": -0.1}]}, output
        )
        model._accumulate_response_anomaly_payload(
            {"token_ids": [7], "topk_logprobs": [{"7": -0.2}]}, output
        )

        payload = output.extra_details_data["response_anomaly_payload"]
        self.assertEqual(payload["tokens"], [7, 7])
        self.assertEqual(payload["topk_logprobs"], [{"7": -0.1}, {"7": -0.2}])

    def test_stream_mismatched_snapshot_does_not_corrupt_payload(self):
        model = self._make_model(enabled=True)
        output = Output()

        model._accumulate_response_anomaly_payload(
            {
                "token_ids": [10, 11],
                "topk_logprobs": [{"10": -0.1}, {"11": -0.2}],
            },
            output,
        )
        model._accumulate_response_anomaly_payload(
            {
                "token_ids": [10, 11, 12],
                "topk_logprobs": [],
            },
            output,
        )
        model._accumulate_response_anomaly_payload(
            {"token_ids": [12], "topk_logprobs": [{"12": -0.3}]},
            output,
        )

        payload = output.extra_details_data["response_anomaly_payload"]
        self.assertEqual(payload["tokens"], [10, 11, 12])
        self.assertEqual(
            payload["topk_logprobs"],
            [{"10": -0.1}, {"11": -0.2}, {"12": -0.3}],
        )

    def test_stream_misaligned_chunk_logs_debug(self):
        """长度不匹配的 chunk 应丢弃并留下 debug 日志。"""
        model = self._make_model(enabled=True)
        output = Output()

        with patch.object(model.logger, "debug") as mock_debug:
            model._accumulate_response_anomaly_payload(
                {
                    "token_ids": [10, 11],
                    "topk_logprobs": [{"10": -0.1}],
                },
                output,
            )

        mock_debug.assert_called_once()
        self.assertNotIn("response_anomaly_payload", output.extra_details_data)

    def test_stream_same_length_snapshot_with_different_prefix_replaces_previous_state(self):
        model = self._make_model(enabled=True)
        output = Output()

        model._accumulate_response_anomaly_payload(
            {
                "token_ids": [10, 11],
                "topk_logprobs": [{"10": -0.1}, {"11": -0.2}],
            },
            output,
        )
        model._accumulate_response_anomaly_payload(
            {
                "token_ids": [20, 21],
                "topk_logprobs": [{"20": -0.3}, {"21": -0.4}],
            },
            output,
        )
        model._accumulate_response_anomaly_payload(
            {"token_ids": [22], "topk_logprobs": [{"22": -0.5}]},
            output,
        )

        payload = output.extra_details_data["response_anomaly_payload"]
        self.assertEqual(payload["tokens"], [20, 21, 22])

    def test_stream_snapshot_chunk_replaces_previous_state(self):
        model = self._make_model(enabled=True)
        output = Output()

        model._accumulate_response_anomaly_payload(
            {"token_ids": [10], "topk_logprobs": [{"10": -0.1}]}, output
        )
        model._accumulate_response_anomaly_payload(
            {
                "token_ids": [10, 11],
                "topk_logprobs": [{"10": -0.1}, {"11": -0.2}],
            },
            output,
        )

        payload = output.extra_details_data["response_anomaly_payload"]
        self.assertEqual(payload["tokens"], [10, 11])
