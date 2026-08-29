import unittest
from unittest import mock

from ais_bench.benchmark.utils.agent_params import (
    AgentParamAdapter,
    parse_env_strings,
    parse_kwarg_strings,
)


class TestParseKwargStrings(unittest.TestCase):
    def test_none(self):
        self.assertEqual(parse_kwarg_strings(None), {})
        self.assertEqual(parse_kwarg_strings([]), {})

    def test_json_literal(self):
        self.assertEqual(parse_kwarg_strings(["a=1"]), {"a": 1})
        self.assertEqual(
            parse_kwarg_strings(["a=[1,2]"]), {"a": [1, 2]}
        )
        self.assertEqual(
            parse_kwarg_strings(['a={"b": 1}']), {"a": {"b": 1}}
        )

    def test_bool_and_none(self):
        self.assertEqual(parse_kwarg_strings(["a=True"]), {"a": True})
        self.assertEqual(parse_kwarg_strings(["a=False"]), {"a": False})
        self.assertEqual(parse_kwarg_strings(["a=None"]), {"a": None})

    def test_string_fallback(self):
        self.assertEqual(parse_kwarg_strings(["a=hello world"]), {"a": "hello world"})

    def test_multiple_and_whitespace(self):
        self.assertEqual(
            parse_kwarg_strings([" a = 10 ", "b=x"]), {"a": 10, "b": "x"}
        )

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            parse_kwarg_strings(["noequals"])

    def test_empty_key_is_value(self):
        # "=value" has "=", so it parses to an empty key, no raise
        self.assertEqual(parse_kwarg_strings(["=value"]), {"": "value"})


class TestParseEnvStrings(unittest.TestCase):
    def test_none(self):
        self.assertEqual(parse_env_strings(None), {})
        self.assertEqual(parse_env_strings([]), {})

    def test_basic(self):
        self.assertEqual(
            parse_env_strings(["K=V", "K2 = V2"]), {"K": "V", "K2": "V2"}
        )

    def test_value_with_equals(self):
        self.assertEqual(
            parse_env_strings(["URL=http://x/v1=1"]), {"URL": "http://x/v1=1"}
        )

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            parse_env_strings(["invalid"])


class TestAgentParamAdapterTranslate(unittest.TestCase):

    def test_none_agent_uses_oracle_fallback(self):
        out = AgentParamAdapter.translate(
            None, {"api_base": "http://x/v1", "api_key": "k"},
        )
        # agent None -> oracle -> fallback env
        self.assertEqual(out["env"]["OPENAI_BASE_URL"], "http://x/v1")
        self.assertEqual(out["env"]["OPENAI_API_KEY"], "k")

    def test_terminus_2_kwarg_mapping(self):
        out = AgentParamAdapter.translate(
            "terminus-2",
            {"api_base": "http://x/v1", "api_key": "secret"},
        )
        self.assertEqual(out["kwargs"]["api_base"], "http://x/v1")
        self.assertEqual(out["kwargs"]["api_key"], "secret")
        self.assertEqual(out["env"], {})

    def test_claude_code_env_mapping(self):
        out = AgentParamAdapter.translate(
            "claude-code",
            {"api_base": "http://x/v1", "api_key": "sk"},
        )
        self.assertEqual(out["env"]["ANTHROPIC_BASE_URL"], "http://x/v1")
        self.assertEqual(out["env"]["ANTHROPIC_API_KEY"], "sk")
        self.assertEqual(out["kwargs"], {})

    def test_fallback_for_unknown_agent(self):
        out = AgentParamAdapter.translate(
            "not-a-real-agent",
            {"api_base": "http://x/v1", "api_key": "k"},
        )
        self.assertEqual(out["env"]["OPENAI_BASE_URL"], "http://x/v1")
        self.assertEqual(out["env"]["OPENAI_API_KEY"], "k")

    def test_model_info_and_llm_kwargs(self):
        out = AgentParamAdapter.translate(
            "oracle",
            {
                "api_base": "http://x/v1",
                "model_info": {"max_tokens": 100},
                "llm_kwargs": {"temperature": 0.2},
            },
        )
        self.assertEqual(out["kwargs"]["model_info"], {"max_tokens": 100})
        self.assertEqual(out["kwargs"]["temperature"], 0.2)

    def test_legacy_llm_call_kwargs(self):
        out = AgentParamAdapter.translate(
            "oracle", {"llm_call_kwargs": {"max_tokens": 50}}
        )
        self.assertEqual(out["kwargs"]["max_tokens"], 50)

    def test_missing_keys_skip(self):
        out = AgentParamAdapter.translate("oracle", {})
        self.assertEqual(out, {"kwargs": {}, "env": {}})

    def test_unsupported_value_not_in_model_cfg(self):
        # value None -> skipped
        out = AgentParamAdapter.translate(
            "terminus-2", {"api_base": "http://x/v1", "top_p": None}
        )
        self.assertNotIn("top_p", out["kwargs"])
        self.assertEqual(out["kwargs"]["api_base"], "http://x/v1")

    def test_mini_swe_agent_default_injected(self):
        out = AgentParamAdapter.translate(
            "mini-swe-agent",
            {"api_base": "http://x/v1", "api_key": "k"},
        )
        self.assertIn("config", out["kwargs"])
        self.assertEqual(
            out["kwargs"]["config"], {"model": {"model_class": "litellm"}}
        )
        self.assertEqual(out["env"]["OPENAI_BASE_URL"], "http://x/v1")

    def test_default_not_overridden_by_none(self):
        # explicit llm_kwargs with a config key should not be overwritten
        out = AgentParamAdapter.translate(
            "mini-swe-agent",
            {"api_base": "http://x/v1", "llm_kwargs": {"config": {"a": 1}}},
        )
        # setdefault only injects when absent; here llm_kwargs.config present
        self.assertEqual(out["kwargs"]["config"], {"a": 1})


class TestAgentParamAdapterDiscovery(unittest.TestCase):

    @mock.patch(
        "ais_bench.benchmark.utils.agent_params.AgentParamAdapter._discover_mapping",
        return_value={},
    )
    def test_import_path_discovery_empty(self, mock_discover):
        # translate should not call discovery for unknown colon import path
        result = AgentParamAdapter.translate("module:Class", {"api_base": "u"})
        self.assertIn("OPENAI_BASE_URL", result["env"])

    def test_match_semantic(self):
        # _match_semantic is case-sensitive; callers pass lowercased haystack
        self.assertEqual(AgentParamAdapter._match_semantic("api_base y"), "api_base")
        self.assertEqual(AgentParamAdapter._match_semantic("base_url"), "api_base")
        self.assertEqual(AgentParamAdapter._match_semantic("openai_api_key"), "api_key")
        self.assertEqual(AgentParamAdapter._match_semantic("something"), None)

    @mock.patch.dict("sys.modules", {"harbor": None}, clear=False)
    def test_discover_mapping_import_error(self):
        # when harbor import fails, discovery returns empty
        self.assertEqual(
            AgentParamAdapter._discover_mapping("qwen-coder"), {}
        )


if __name__ == "__main__":
    unittest.main()