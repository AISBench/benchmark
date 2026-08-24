"""Agent parameter adapter.

Harbor agents receive the same semantic parameters (e.g. the model service
base url) through different channels: some take a constructor kwarg
(terminus-2's ``api_base``), installed agents usually map them to environment
variables declared in ``BaseInstalledAgent.ENV_VARS`` / ``CLI_FLAGS``. This
adapter translates one unified set of user-facing parameters into per-agent
``AgentConfig.kwargs`` / ``AgentConfig.env``.

No harbor imports at module level: harbor is only needed at translate time,
so this module stays importable in non-agent AISBench environments.
"""

import json
from typing import Any

# Semantic unified keys understood by the adapter.
_SEMANTIC_KEYS = ("api_base", "api_key")

# Explicit per-agent mappings: agent name -> semantic key -> (kind, target).
# kind is "kwarg" (AgentConfig.kwargs[target]) or "env" (AgentConfig.env[target]).
EXPLICIT_MAP: dict[str, dict[str, tuple[str, str]]] = {
    "terminus-2": {
        "api_base": ("kwarg", "api_base"),
        "api_key": ("kwarg", "api_key"),
    },
}

# Fallback mapping used when no agent-specific mapping is discovered.
_FALLBACK_MAP: dict[str, tuple[str, str]] = {
    "api_base": ("env", "OPENAI_BASE_URL"),
    "api_key": ("env", "OPENAI_API_KEY"),
}

# Keyword fragments used to auto-discover descriptors for a semantic key.
_DISCOVERY_HINTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # (kwarg/env-name fragments that indicate the semantic key)
    "api_base": (("api_base", "base_url"), ("base_url", "base")),
    "api_key": (("api_key", "auth_token", "api_token"), ("api_key", "api_token")),
}


def parse_kwarg_strings(kwargs_list: list[str] | None) -> dict[str, Any]:
    """Parse ``key=value`` strings into a dict (values parsed as JSON literals).

    Mirrors harbor's ``harbor.cli.utils.parse_kwargs`` without importing harbor.
    """
    if not kwargs_list:
        return {}
    result: dict[str, Any] = {}
    for item in kwargs_list:
        if "=" not in item:
            raise ValueError(f"Invalid kwarg format: {item}. Expected key=value")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            if value == "True":
                result[key] = True
            elif value == "False":
                result[key] = False
            elif value == "None":
                result[key] = None
            else:
                result[key] = value
    return result


def parse_env_strings(env_list: list[str] | None) -> dict[str, str]:
    """Parse ``KEY=VALUE`` strings into a dict of strings."""
    if not env_list:
        return {}
    result: dict[str, str] = {}
    for item in env_list:
        if "=" not in item:
            raise ValueError(f"Invalid env var format: {item}. Expected KEY=VALUE")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


class AgentParamAdapter:
    """Translate unified user-facing parameters into per-agent kwargs / env."""

    @classmethod
    def translate(cls, agent_name: str | None, model_cfg: dict) -> dict[str, dict]:
        """Translate unified params in ``model_cfg`` for ``agent_name``.

        Unified fields read from ``model_cfg``:
          - api_base / api_key: translated per agent (kwarg or env)
          - llm_kwargs (or legacy llm_call_kwargs): merged into kwargs
          - model_info: merged into kwargs["model_info"]
          - temperature / max_tokens / top_p / top_k: translated per agent,
            falling back to kwargs

        Returns ``{"kwargs": {...}, "env": {...}}``. Raw ``agent_kwargs`` /
        ``agent_env`` in ``model_cfg`` are NOT touched here; the caller merges
        them afterwards so explicit user values take precedence.
        """
        name = agent_name or "oracle"
        mapping = cls._agent_mapping(name)

        kwargs: dict[str, Any] = {}
        env: dict[str, str] = {}

        for key in ("api_base", "api_key", "temperature", "max_tokens", "top_p", "top_k"):
            value = model_cfg.get(key)
            if value is None:
                continue
            target = mapping.get(key)
            if target is None:
                kwargs[key] = value
                continue
            kind, target_key = target
            if kind == "kwarg":
                kwargs[target_key] = value
            else:
                env[target_key] = str(value)

        llm_kwargs = model_cfg.get("llm_kwargs") or model_cfg.get("llm_call_kwargs") or {}
        if isinstance(llm_kwargs, dict):
            for key, value in llm_kwargs.items():
                target = mapping.get(key)
                if target is None:
                    kwargs[key] = value
                    continue
                kind, target_key = target
                if kind == "kwarg":
                    kwargs[target_key] = value
                else:
                    env[target_key] = str(value)

        if model_cfg.get("model_info") is not None:
            kwargs["model_info"] = model_cfg["model_info"]

        return {"kwargs": kwargs, "env": env}

    # ------------------------------------------------------------------
    # mapping resolution
    # ------------------------------------------------------------------

    @classmethod
    def _agent_mapping(cls, agent_name: str) -> dict[str, tuple[str, str]]:
        mapping: dict[str, tuple[str, str]] = {}
        for semantic, target in EXPLICIT_MAP.get(agent_name, {}).items():
            mapping.setdefault(semantic, target)
        for semantic, target in cls._discover_mapping(agent_name).items():
            mapping.setdefault(semantic, target)
        for semantic, target in _FALLBACK_MAP.items():
            mapping.setdefault(semantic, target)
        return mapping

    @classmethod
    def _discover_mapping(cls, agent_name: str) -> dict[str, tuple[str, str]]:
        """Auto-discover env/kwarg targets from harbor installed-agent descriptors.

        Reads ``BaseInstalledAgent.ENV_VARS`` / ``CLI_FLAGS`` declarative
        descriptors (0.21.0 ``harbor.agents.installed.base``) and maps semantic
        keys to the matching env var or kwarg.
        """
        if not agent_name or ":" in agent_name:
            return {}
        try:
            from harbor.agents.factory import AgentFactory
            from harbor.agents.installed.base import BaseInstalledAgent
            from harbor.models.agent.name import AgentName
        except Exception:
            return {}
        if agent_name not in AgentName.values():
            return {}
        try:
            agent_class = AgentFactory.get_agent_class(AgentName(agent_name))
        except Exception:
            return {}

        discovered: dict[str, tuple[str, str]] = {}
        if not issubclass(agent_class, BaseInstalledAgent):
            return discovered

        descriptors = [
            *getattr(agent_class, "CLI_FLAGS", []),
            *getattr(agent_class, "ENV_VARS", []),
        ]
        for descriptor in descriptors:
            kwarg = getattr(descriptor, "kwarg", "") or ""
            env_name = getattr(descriptor, "env", "") or ""
            env_fallback = getattr(descriptor, "env_fallback", "") or ""
            haystack = f"{kwarg} {env_name} {env_fallback}".lower()

            semantic = cls._match_semantic(haystack)
            if semantic is None or semantic in discovered:
                continue
            if env_name:
                discovered[semantic] = ("env", env_name)
            else:
                discovered[semantic] = ("kwarg", kwarg)
        return discovered

    @staticmethod
    def _match_semantic(haystack: str) -> str | None:
        for semantic, (kwarg_hints, env_hints) in _DISCOVERY_HINTS.items():
            if any(h in haystack for h in kwarg_hints) or any(
                h in haystack for h in env_hints
            ):
                return semantic
        return None
