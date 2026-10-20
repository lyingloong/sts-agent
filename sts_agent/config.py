"""Shared, validated runtime configuration for the StS agent tools."""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any

from sts_agent.core.modes import MODE_FRAMEWORKS, resolve_framework_mode
from sts_agent.paths import PROJECT_ROOT
from sts_agent.policy.formats import DEFAULT_PROMPT_FORMAT, require_prompt_format


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "framework": {
        "game": "sts1",
        "control_mode": "manual",
        "policy_backend": "none",
        "learning_method": "none",
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
    },
    "recording": {
        "listen": {
            "host": "127.0.0.1",
            "port": 8766,
        },
        "output_dir": "data/trajectories/human",
    },
    "codex": {
        "actor_id": "codex",
        "character": "IRONCLAD",
        "ascension": 0,
        "seed": "",
        "output_dir": "data/trajectories/codex",
        "max_actions_per_run": 2000,
        "transition_timeout_seconds": 90.0,
    },
    "policy": {
        "enabled": True,
        "runtime": "local",
        "model": {
            "base": "Qwen/Qwen3-0.6B",
            "revision": "main",
            "initial_adapter": "",
            "initial_policy_version": "sft-v0",
            "hub_endpoint": "https://huggingface.co",
        },
        "request": {
            "top_k": 3,
            "response_timeout_seconds": 8.0,
            "startup_timeout_seconds": 300.0,
            "screens": ["ALL"],
        },
        "local": {
            "python": "auto",
        },
        "remote": {
            "ssh_host": "gpu-server",
            "project_root": "/srv/sts-agent",
            "python": ".venv/bin/python",
            "online_rl_config": "config/online_rl.json",
            "connect_timeout_seconds": 30.0,
            "update_timeout_seconds": 1800.0,
            "inference_gpu": {
                "device": "auto",
                "minimum_free_memory_mib": 6000,
            },
        },
        "development": {
            "mock": False,
            "debug": False,
        },
        "confidence_auto": {
            "enabled": True,
            "min_probability": 0.85,
        },
        "api": {
            "local_port": 18080,
            "remote_port": 8000,
            "ssh_tunnel": True,
            "timeout_seconds": 120.0,
            "max_retries": 2,
            "retry_backoff_seconds": 5.0,
            "temperature": 0.0,
            "thinking": False,
            "max_output_tokens": 64,
            "prompt_format": DEFAULT_PROMPT_FORMAT,
            "prompt_template": "canonical",
            "output_dir": "data/experiments/baseline",
            "serving": {
                "backend": "vllm",
                "tensor_parallel_size": 4,
                "devices": "auto",
                "minimum_free_memory_mib": 21000,
                "maximum_gpu_utilization_percent": 10,
                "max_model_len": 16384,
                "gpu_memory_utilization": 0.85,
                "enforce_eager": True,
                "disable_nccl_p2p": True,
                "disable_nccl_ib": True,
                "deployment_file": "data/runtime/baseline-deployment.json",
            },
        },
    },
    "teacher": {
        "enabled": False,
        "base_url": "https://api.llm.ustc.edu.cn/v1",
        "model": "deepseek-v4-pro",
        "api_key_file": "deepseek_api.txt",
        "timeout_seconds": 90.0,
        "max_retries": 3,
        # RPM=20 is approximately one request every 3 seconds. Start a
        # retry at 5 seconds to avoid immediately hitting the same limit.
        "retry_backoff_seconds": 5.0,
        "retry_max_backoff_seconds": 30.0,
        "temperature": 0.0,
        "max_tokens": 512,
        # Optional total budget for thinking + final answer. Zero keeps the
        # max_tokens request for providers that do not support the combined budget.
        "max_completion_tokens": 0,
        "max_input_chars": 32000,
        "thinking": True,
        "json_mode": True,
        "streaming": True,
        "reasoning_effort": "",
        "stream_update_interval_seconds": 0.3,
        "screens": ["ALL"],
    },
    "posthoc_feedback": {
        "enabled": False,
        "mode": "human",
        "base_url": "https://api.llm.ustc.edu.cn/v1",
        "model": "deepseek-v4-pro",
        "api_key_file": "deepseek_api.txt",
        "timeout_seconds": 180.0,
        "max_retries": 3,
        "retry_backoff_seconds": 5.0,
        "retry_max_backoff_seconds": 30.0,
        "temperature": 0.0,
        "max_tokens": 2048,
        "max_input_chars": 96000,
        "max_reflection_chars": 4000,
        "max_candidates_per_direction": 10,
        "max_applied_per_direction": 8,
        "min_confidence": 0.75,
    },
    "prompt_evolution": {
        "enabled": False,
        "policy_dir": "prompts/characters",
        "output_dir": "data/prompt_evolution",
        "max_skill_chars": 2400,
        # DeepSeek v4 supports a very large context. Keep a headroom-aware
        # character budget here (the value is characters, not tokens) so a
        # long run is rarely truncated while leaving room for system/output.
        "max_analysis_input_chars": 900000,
        "max_analysis_tokens": 8192,
        "max_revisions_per_cycle": 3,
        "auto_continue": True,
    },
    "overlay": {
        "always_on_top": True,
        "frameless": False,
        "status": {
            "geometry": "540x680+24+84",
            "opacity": 0.96,
            "divider_columns": 36,
        },
        "control": {
            "geometry": "500x440+580+84",
            "opacity": 0.97,
        },
        "online_rl": {
            "geometry": "360x240+1090+84",
            "opacity": 0.97,
        },
        "prompt_evolution": {
            "geometry": "420x300+1090+340",
            "opacity": 0.97,
        },
    },
}

WINDOW_GEOMETRY_PATTERN = re.compile(r"^\d+x\d+[+-]\d+[+-]\d+$")


class ConfigError(ValueError):
    """Raised when config/agent.json is missing or contains unsafe settings."""


def project_root() -> Path:
    return PROJECT_ROOT


def config_path(root: Path | None = None, config_file: str | Path | None = None) -> Path:
    """Resolve the selected config relative to the project root by default."""

    base = root or project_root()
    if config_file is None:
        return base / "config" / "agent.json"
    path = Path(config_file)
    return path if path.is_absolute() else base / path


def _merge(default: dict[str, Any], provided: dict[str, Any], location: str = "") -> dict[str, Any]:
    result = copy.deepcopy(default)
    for key, value in provided.items():
        path = f"{location}.{key}" if location else key
        if key not in default:
            raise ConfigError(f"unknown config field: {path}")
        default_value = default[key]
        if isinstance(default_value, dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{path} must be an object")
            result[key] = _merge(default_value, value, path)
        else:
            result[key] = value
    return result


def _require_bool(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false")


def _require_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")


def _require_number(value: Any, name: str, minimum: float, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    if not math.isfinite(float(value)) or value < minimum or (maximum is not None and value > maximum):
        bounds = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ConfigError(f"{name} must be {bounds}")


def _validate_framework_combination(framework: dict[str, str]) -> None:
    control = framework["control_mode"]
    backend = framework["policy_backend"]
    learning = framework["learning_method"]
    if backend == "remote_api" and (control != "autonomous" or learning != "none"):
        raise ConfigError("remote_api requires autonomous control without learning")
    if control == "manual" and backend != "none":
        raise ConfigError("manual control requires framework.policy_backend=none")
    if control == "codex" and (backend != "none" or learning != "collect_codex"):
        raise ConfigError("codex control requires policy_backend=none and learning_method=collect_codex")
    if control == "candidate_assist" and backend not in {"local_qwen", "remote_qwen", "mock"}:
        raise ConfigError("candidate_assist requires a Qwen or mock policy backend")
    if learning in {"teacher_sft", "prompt_evolution"} and backend != "remote_teacher":
        raise ConfigError(f"{learning} requires framework.policy_backend=remote_teacher")
    if learning == "online_rl" and (control != "autonomous" or backend != "remote_qwen"):
        raise ConfigError("online_rl requires autonomous control and framework.policy_backend=remote_qwen")
    if learning == "prompt_evolution" and control != "autonomous":
        raise ConfigError("prompt_evolution requires autonomous control")
    if learning == "dagger" and control != "candidate_assist":
        raise ConfigError("dagger requires candidate_assist control")
    if learning == "collect_human" and (control != "manual" or backend != "none"):
        raise ConfigError("collect_human requires manual control and framework.policy_backend=none")
    if learning == "collect_codex" and (control != "codex" or backend != "none"):
        raise ConfigError("collect_codex requires codex control and framework.policy_backend=none")


def _apply_framework_runtime(config: dict[str, Any], framework: dict[str, str]) -> None:
    """Project the direct framework dimensions onto runtime component flags."""

    policy = config["policy"]
    teacher = config["teacher"]
    posthoc = config["posthoc_feedback"]
    evolution = config["prompt_evolution"]
    backend = framework["policy_backend"]
    learning = framework["learning_method"]

    if backend == "remote_api":
        policy["enabled"] = True
        policy["runtime"] = "remote_api"
        teacher["enabled"] = False
        posthoc["enabled"] = False
        evolution["enabled"] = False
    elif backend in {"local_qwen", "remote_qwen", "mock"}:
        policy["enabled"] = True
        policy["runtime"] = "local" if backend in {"local_qwen", "mock"} else (
            "remote_online_rl" if learning == "online_rl" else "remote_inference"
        )
        teacher["enabled"] = False
    elif backend == "remote_teacher":
        policy["enabled"] = False
        teacher["enabled"] = True
    else:
        policy["enabled"] = False
        teacher["enabled"] = False

    if learning == "teacher_sft":
        policy["enabled"] = False
        teacher["enabled"] = True
    elif learning == "prompt_evolution":
        policy["enabled"] = False
        teacher["enabled"] = True
        evolution["enabled"] = True
    elif learning == "online_rl":
        policy["enabled"] = True
        policy["runtime"] = "remote_online_rl"
    elif learning == "none" and framework["control_mode"] == "manual":
        posthoc["enabled"] = False


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a merged config and return it for convenient composition."""
    config = copy.deepcopy(config)
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    framework = config["framework"]
    try:
        resolved_framework = resolve_framework_mode(framework)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    _validate_framework_combination(resolved_framework)
    _apply_framework_runtime(config, resolved_framework)
    server = config["server"]
    recording = config["recording"]
    policy = config["policy"]
    teacher = config["teacher"]
    posthoc = config["posthoc_feedback"]
    evolution = config["prompt_evolution"]
    model = policy["model"]
    request = policy["request"]
    local = policy["local"]
    development = policy["development"]
    confidence = policy["confidence_auto"]
    overlay = config["overlay"]

    _require_string(server["host"], "server.host")
    port = server["port"]
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ConfigError("server.port must be an integer from 1 to 65535")

    recording_listen = recording["listen"]
    _require_string(recording_listen["host"], "recording.listen.host")
    recording_port = recording_listen["port"]
    if isinstance(recording_port, bool) or not isinstance(recording_port, int) or not 1 <= recording_port <= 65535:
        raise ConfigError("recording.listen.port must be an integer from 1 to 65535")
    _require_string(recording["output_dir"], "recording.output_dir")
    recording_path = Path(str(recording["output_dir"]))
    if recording_path.is_absolute() or ".." in recording_path.parts:
        raise ConfigError("recording.output_dir must be a project-relative path")

    codex = config["codex"]
    _require_string(codex["actor_id"], "codex.actor_id")
    if codex["character"] != "IRONCLAD":
        raise ConfigError("codex.character is currently fixed to IRONCLAD")
    if isinstance(codex["ascension"], bool) or codex["ascension"] != 0:
        raise ConfigError("codex.ascension is currently fixed to 0")
    seed = codex["seed"]
    if not isinstance(seed, str) or (seed and not re.fullmatch(r"[A-Za-z0-9]+", seed)):
        raise ConfigError("codex.seed must be empty for random seed or contain only letters and digits")
    _require_string(codex["output_dir"], "codex.output_dir")
    codex_output_path = Path(str(codex["output_dir"]))
    if codex_output_path.is_absolute() or ".." in codex_output_path.parts:
        raise ConfigError("codex.output_dir must be a project-relative path")
    max_actions = codex["max_actions_per_run"]
    if isinstance(max_actions, bool) or not isinstance(max_actions, int) or not 1 <= max_actions <= 10000:
        raise ConfigError("codex.max_actions_per_run must be an integer from 1 to 10000")
    _require_number(codex["transition_timeout_seconds"], "codex.transition_timeout_seconds", 1.0, 600.0)
    if framework["control_mode"] == "codex" and (policy["enabled"] or teacher["enabled"]):
        raise ConfigError("codex mode uses Codex as the external policy; disable policy and teacher backends")

    _require_bool(policy["enabled"], "policy.enabled")
    if policy["runtime"] not in {"local", "remote_inference", "remote_online_rl", "remote_api"}:
        raise ConfigError("policy.runtime must be local, remote_inference, or remote_online_rl")
    _require_string(model["base"], "policy.model.base")
    _require_string(model["revision"], "policy.model.revision")
    if not isinstance(model["initial_adapter"], str):
        raise ConfigError("policy.model.initial_adapter must be a string")
    _require_string(model["initial_policy_version"], "policy.model.initial_policy_version")
    _require_string(model["hub_endpoint"], "policy.model.hub_endpoint")
    if not str(model["hub_endpoint"]).startswith(("https://", "http://")):
        raise ConfigError("policy.model.hub_endpoint must start with http:// or https://")
    if policy["runtime"] == "remote_online_rl" and not model["initial_adapter"].strip() and not development["mock"]:
        raise ConfigError("policy.model.initial_adapter is required for policy.runtime=remote_online_rl")
    top_k = request["top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ConfigError("policy.request.top_k must be an integer >= 1")
    _require_number(request["response_timeout_seconds"], "policy.request.response_timeout_seconds", 0.1)
    _require_number(request["startup_timeout_seconds"], "policy.request.startup_timeout_seconds", 1.0)
    screens = request["screens"]
    if not isinstance(screens, list) or not screens or not all(isinstance(item, str) for item in screens):
        raise ConfigError("policy.request.screens must be a non-empty list of strings")
    unknown_screens = {item.strip().upper() for item in screens} - {"ALL", "COMBAT"}
    if unknown_screens:
        raise ConfigError(f"policy.request.screens contains unsupported values: {', '.join(sorted(unknown_screens))}")
    _require_string(local["python"], "policy.local.python")
    _require_bool(development["mock"], "policy.development.mock")
    _require_bool(development["debug"], "policy.development.debug")
    remote = policy["remote"]
    _require_string(remote["ssh_host"], "policy.remote.ssh_host")
    _require_string(remote["project_root"], "policy.remote.project_root")
    if not str(remote["project_root"]).startswith("/"):
        raise ConfigError("policy.remote.project_root must be an absolute remote POSIX path")
    _require_string(remote["python"], "policy.remote.python")
    _require_string(remote["online_rl_config"], "policy.remote.online_rl_config")
    online_config = Path(str(remote["online_rl_config"]))
    if online_config.is_absolute() or ".." in online_config.parts:
        raise ConfigError("policy.remote.online_rl_config must be a relative remote path")
    _require_number(remote["connect_timeout_seconds"], "policy.remote.connect_timeout_seconds", 1.0, 300.0)
    _require_number(remote["update_timeout_seconds"], "policy.remote.update_timeout_seconds", 30.0, 7200.0)
    inference_gpu = remote["inference_gpu"]
    device = inference_gpu["device"]
    if device != "auto" and (isinstance(device, bool) or not isinstance(device, int) or device < 0):
        raise ConfigError("policy.remote.inference_gpu.device must be 'auto' or a non-negative integer")
    minimum_memory = inference_gpu["minimum_free_memory_mib"]
    if isinstance(minimum_memory, bool) or not isinstance(minimum_memory, int) or minimum_memory < 1:
        raise ConfigError("policy.remote.inference_gpu.minimum_free_memory_mib must be a positive integer")
    _require_bool(confidence["enabled"], "policy.confidence_auto.enabled")
    _require_number(confidence["min_probability"], "policy.confidence_auto.min_probability", 0.0, 1.0)
    if policy["runtime"] == "remote_api":
        if model["initial_adapter"] or development["mock"]:
            raise ConfigError("baseline remote_api requires an unmodified model (no adapter/mock)")
    api = policy["api"]
    for key in ("local_port", "remote_port", "max_output_tokens"):
        value = api[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
            raise ConfigError(f"policy.api.{key} must be an integer from 1 to 65535")
    for key in ("ssh_tunnel", "thinking"):
        _require_bool(api[key], f"policy.api.{key}")
    _require_number(api["timeout_seconds"], "policy.api.timeout_seconds", 1, 600)
    _require_number(api["retry_backoff_seconds"], "policy.api.retry_backoff_seconds", 1, 60)
    _require_number(api["temperature"], "policy.api.temperature", 0, 2)
    if isinstance(api["max_retries"], bool) or not isinstance(api["max_retries"], int) or not 0 <= api["max_retries"] <= 5:
        raise ConfigError("policy.api.max_retries must be an integer from 0 to 5")
    if api["prompt_template"] not in {"canonical", "state_first", "decision_first"}:
        raise ConfigError("policy.api.prompt_template is invalid")
    try:
        require_prompt_format(api["prompt_format"])
    except ValueError as exc:
        raise ConfigError(f"policy.api.prompt_format: {exc}") from exc
    serving = api["serving"]
    if serving["backend"] not in {"vllm", "transformers"}:
        raise ConfigError("policy.api.serving.backend must be 'vllm' or 'transformers'")
    for key in ("tensor_parallel_size", "minimum_free_memory_mib", "max_model_len"):
        if isinstance(serving[key], bool) or not isinstance(serving[key], int) or serving[key] < 1:
            raise ConfigError(f"policy.api.serving.{key} must be a positive integer")
    devices = serving["devices"]
    if devices != "auto" and (not isinstance(devices, list) or len(devices) != serving["tensor_parallel_size"] or
            any(isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in devices) or len(set(devices)) != len(devices)):
        raise ConfigError("policy.api.serving.devices must be 'auto' or distinct GPU indices matching tensor_parallel_size")
    _require_number(serving["maximum_gpu_utilization_percent"], "policy.api.serving.maximum_gpu_utilization_percent", 0, 100)
    _require_number(serving["gpu_memory_utilization"], "policy.api.serving.gpu_memory_utilization", 0.1, 0.95)
    for key in ("enforce_eager", "disable_nccl_p2p", "disable_nccl_ib"):
        _require_bool(serving[key], f"policy.api.serving.{key}")
    for name, value in (("output_dir", api["output_dir"]), ("serving.deployment_file", serving["deployment_file"])):
        _require_string(value, f"policy.api.{name}")
        if Path(value).is_absolute() or ".." in Path(value).parts:
            raise ConfigError(f"policy.api.{name} must be project-relative")
    if api["max_output_tokens"] >= serving["max_model_len"]:
        raise ConfigError("policy.api.max_output_tokens must be smaller than max_model_len")

    _require_bool(teacher["enabled"], "teacher.enabled")
    _require_string(teacher["base_url"], "teacher.base_url")
    if not str(teacher["base_url"]).startswith(("https://", "http://")):
        raise ConfigError("teacher.base_url must start with http:// or https://")
    _require_string(teacher["model"], "teacher.model")
    _require_string(teacher["api_key_file"], "teacher.api_key_file")
    key_path = Path(str(teacher["api_key_file"]))
    if key_path.is_absolute() or ".." in key_path.parts:
        raise ConfigError("teacher.api_key_file must be a project-relative path")
    _require_number(teacher["timeout_seconds"], "teacher.timeout_seconds", 1.0)
    retries = teacher["max_retries"]
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 10:
        raise ConfigError("teacher.max_retries must be an integer from 0 to 10")
    _require_number(teacher["retry_backoff_seconds"], "teacher.retry_backoff_seconds", 0.0, 60.0)
    _require_number(teacher["retry_max_backoff_seconds"], "teacher.retry_max_backoff_seconds", 0.0, 300.0)
    if teacher["retry_max_backoff_seconds"] < teacher["retry_backoff_seconds"]:
        raise ConfigError("teacher.retry_max_backoff_seconds must be >= teacher.retry_backoff_seconds")
    _require_number(teacher["temperature"], "teacher.temperature", 0.0, 2.0)
    max_tokens = teacher["max_tokens"]
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 32 <= max_tokens <= 8192:
        raise ConfigError("teacher.max_tokens must be an integer from 32 to 8192")
    max_completion_tokens = teacher["max_completion_tokens"]
    if isinstance(max_completion_tokens, bool) or not isinstance(max_completion_tokens, int) or not 0 <= max_completion_tokens <= 131072:
        raise ConfigError("teacher.max_completion_tokens must be an integer from 0 to 131072")
    max_input_chars = teacher["max_input_chars"]
    if isinstance(max_input_chars, bool) or not isinstance(max_input_chars, int) or not 0 <= max_input_chars <= 262144:
        raise ConfigError("teacher.max_input_chars must be an integer from 0 to 262144")
    _require_bool(teacher["thinking"], "teacher.thinking")
    _require_bool(teacher["json_mode"], "teacher.json_mode")
    _require_bool(teacher["streaming"], "teacher.streaming")
    reasoning_effort = teacher["reasoning_effort"]
    if not isinstance(reasoning_effort, str) or reasoning_effort.strip().lower() not in {"", "low", "medium", "high", "max"}:
        raise ConfigError("teacher.reasoning_effort must be empty, low, medium, high, or max")
    _require_number(teacher["stream_update_interval_seconds"], "teacher.stream_update_interval_seconds", 0.05, 2.0)
    teacher_screens = teacher["screens"]
    if not isinstance(teacher_screens, list) or not teacher_screens or not all(isinstance(item, str) for item in teacher_screens):
        raise ConfigError("teacher.screens must be a non-empty list of strings")
    unknown_teacher_screens = {item.strip().upper() for item in teacher_screens} - {"ALL", "COMBAT"}
    if unknown_teacher_screens:
        raise ConfigError(f"teacher.screens contains unsupported values: {', '.join(sorted(unknown_teacher_screens))}")
    if teacher["enabled"] and policy["enabled"]:
        raise ConfigError("teacher.enabled and policy.enabled cannot both be true; disable local policy for teacher collection")

    _require_bool(posthoc["enabled"], "posthoc_feedback.enabled")
    if not isinstance(posthoc["mode"], str) or posthoc["mode"] not in {"human", "auto"}:
        raise ConfigError("posthoc_feedback.mode must be 'human' or 'auto'")
    _require_string(posthoc["base_url"], "posthoc_feedback.base_url")
    if not str(posthoc["base_url"]).startswith(("https://", "http://")):
        raise ConfigError("posthoc_feedback.base_url must start with http:// or https://")
    _require_string(posthoc["model"], "posthoc_feedback.model")
    _require_string(posthoc["api_key_file"], "posthoc_feedback.api_key_file")
    posthoc_key_path = Path(str(posthoc["api_key_file"]))
    if posthoc_key_path.is_absolute() or ".." in posthoc_key_path.parts:
        raise ConfigError("posthoc_feedback.api_key_file must be a project-relative path")
    _require_number(posthoc["timeout_seconds"], "posthoc_feedback.timeout_seconds", 1.0)
    posthoc_retries = posthoc["max_retries"]
    if isinstance(posthoc_retries, bool) or not isinstance(posthoc_retries, int) or not 0 <= posthoc_retries <= 10:
        raise ConfigError("posthoc_feedback.max_retries must be an integer from 0 to 10")
    _require_number(posthoc["retry_backoff_seconds"], "posthoc_feedback.retry_backoff_seconds", 0.0, 60.0)
    _require_number(posthoc["retry_max_backoff_seconds"], "posthoc_feedback.retry_max_backoff_seconds", 0.0, 300.0)
    if posthoc["retry_max_backoff_seconds"] < posthoc["retry_backoff_seconds"]:
        raise ConfigError("posthoc_feedback.retry_max_backoff_seconds must be >= posthoc_feedback.retry_backoff_seconds")
    _require_number(posthoc["temperature"], "posthoc_feedback.temperature", 0.0, 2.0)
    posthoc_tokens = posthoc["max_tokens"]
    if isinstance(posthoc_tokens, bool) or not isinstance(posthoc_tokens, int) or not 128 <= posthoc_tokens <= 8192:
        raise ConfigError("posthoc_feedback.max_tokens must be an integer from 128 to 8192")
    posthoc_input = posthoc["max_input_chars"]
    if isinstance(posthoc_input, bool) or not isinstance(posthoc_input, int) or not 1024 <= posthoc_input <= 262144:
        raise ConfigError("posthoc_feedback.max_input_chars must be an integer from 1024 to 262144")
    reflection_limit = posthoc["max_reflection_chars"]
    if isinstance(reflection_limit, bool) or not isinstance(reflection_limit, int) or not 128 <= reflection_limit <= 16384:
        raise ConfigError("posthoc_feedback.max_reflection_chars must be an integer from 128 to 16384")
    candidate_limit = posthoc["max_candidates_per_direction"]
    if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int) or not 1 <= candidate_limit <= 20:
        raise ConfigError("posthoc_feedback.max_candidates_per_direction must be an integer from 1 to 20")
    applied_limit = posthoc["max_applied_per_direction"]
    if isinstance(applied_limit, bool) or not isinstance(applied_limit, int) or not 1 <= applied_limit <= candidate_limit:
        raise ConfigError("posthoc_feedback.max_applied_per_direction must be an integer from 1 to max_candidates_per_direction")
    _require_number(posthoc["min_confidence"], "posthoc_feedback.min_confidence", 0.0, 1.0)
    if posthoc["enabled"] and (not policy["enabled"] or policy["runtime"] != "remote_online_rl"):
        raise ConfigError("posthoc_feedback.enabled requires policy.enabled with policy.runtime=remote_online_rl")

    _require_bool(evolution["enabled"], "prompt_evolution.enabled")
    for key in ("policy_dir", "output_dir"):
        _require_string(evolution[key], f"prompt_evolution.{key}")
        relative_path = Path(str(evolution[key]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ConfigError(f"prompt_evolution.{key} must be a project-relative path")
    for key in ("max_skill_chars", "max_analysis_input_chars", "max_analysis_tokens", "max_revisions_per_cycle"):
        value = evolution[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"prompt_evolution.{key} must be an integer")
    if not 128 <= evolution["max_skill_chars"] <= 12000:
        raise ConfigError("prompt_evolution.max_skill_chars must be from 128 to 12000")
    if not 4096 <= evolution["max_analysis_input_chars"] <= 900000:
        raise ConfigError("prompt_evolution.max_analysis_input_chars must be from 4096 to 900000")
    if not 256 <= evolution["max_analysis_tokens"] <= 8192:
        raise ConfigError("prompt_evolution.max_analysis_tokens must be from 256 to 8192")
    if not 0 <= evolution["max_revisions_per_cycle"] <= 12:
        raise ConfigError("prompt_evolution.max_revisions_per_cycle must be from 0 to 12")
    _require_bool(evolution["auto_continue"], "prompt_evolution.auto_continue")
    if evolution["enabled"]:
        if not teacher["enabled"]:
            raise ConfigError("prompt_evolution.enabled requires teacher.enabled")
        if policy["enabled"]:
            raise ConfigError("prompt_evolution.enabled requires policy.enabled=false")
        if posthoc["enabled"]:
            raise ConfigError("prompt_evolution.enabled requires posthoc_feedback.enabled=false")

    _require_bool(overlay["always_on_top"], "overlay.always_on_top")
    _require_bool(overlay["frameless"], "overlay.frameless")
    for name in ("status", "control", "online_rl", "prompt_evolution"):
        window = overlay[name]
        _require_string(window["geometry"], f"overlay.{name}.geometry")
        if not WINDOW_GEOMETRY_PATTERN.fullmatch(window["geometry"]):
            raise ConfigError(f"overlay.{name}.geometry must be WIDTHxHEIGHT+X+Y")
        _require_number(window["opacity"], f"overlay.{name}.opacity", 0.2, 1.0)
    divider_columns = overlay["status"]["divider_columns"]
    if isinstance(divider_columns, bool) or not isinstance(divider_columns, int) or not 12 <= divider_columns <= 78:
        raise ConfigError("overlay.status.divider_columns must be an integer from 12 to 78")
    config["framework"] = resolved_framework
    return config


def load_agent_config(root: Path | None = None, config_file: str | Path | None = None, *, mode: str | None = None) -> dict[str, Any]:
    """Load the active named mode plus shared settings into runtime config."""
    path = config_path(root, config_file)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"missing configuration file: {path}") from exc
    try:
        provided = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc.msg} (line {exc.lineno})") from exc
    if not isinstance(provided, dict):
        raise ConfigError(f"{path} must contain one JSON object")
    unknown_keys = set(provided) - {"schema_version", "active_mode", "shared"}
    if unknown_keys:
        unknown = ", ".join(sorted(str(key) for key in unknown_keys))
        raise ConfigError(f"unknown config field(s) in {path}: {unknown}; mode-specific settings belong in config/modes/<mode>.json")
    if provided.get("schema_version") != 2:
        raise ConfigError("config/agent.json schema_version must be 2")
    mode_name = mode if mode is not None else provided.get("active_mode")
    if not isinstance(mode_name, str) or mode_name not in MODE_FRAMEWORKS:
        choices = ", ".join(sorted(MODE_FRAMEWORKS))
        raise ConfigError(f"active_mode must be one of: {choices}")

    shared = provided.get("shared", {})
    if not isinstance(shared, dict):
        raise ConfigError("shared must be an object")
    if "framework" in shared:
        raise ConfigError("shared.framework is not allowed; choose one active_mode instead")
    merged = _merge(DEFAULT_CONFIG, shared, "shared")

    profile_path = path.parent / "modes" / f"{mode_name}.json"
    try:
        profile_raw = profile_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"missing mode profile for {mode_name}: {profile_path}") from exc
    try:
        profile = json.loads(profile_raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {profile_path}: {exc.msg} (line {exc.lineno})") from exc
    if not isinstance(profile, dict):
        raise ConfigError(f"{profile_path} must contain one JSON object")
    if "framework" in profile:
        raise ConfigError(f"{profile_path} must not define framework; mode combinations are fixed in sts_agent.core.modes")
    merged = _merge(merged, profile, f"mode.{mode_name}")
    merged["schema_version"] = DEFAULT_CONFIG["schema_version"]
    merged["framework"] = MODE_FRAMEWORKS[mode_name]
    return validate_config(merged)
