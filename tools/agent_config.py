"""Shared, validated runtime configuration for the StS agent tools."""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
    },
    "policy": {
        "enabled": True,
        "runtime": "local",
        "model": {
            "base": "Qwen/Qwen3-0.6B",
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
            "ssh_host": "linke84090",
            "project_root": "/data/wanghao/sts-agent",
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
    },
    "teacher": {
        "enabled": False,
        "base_url": "https://api.llm.ustc.edu.cn/v1",
        "model": "deepseek-v4-pro",
        "api_key_file": "deepseek_api.txt",
        "timeout_seconds": 90.0,
        "max_retries": 1,
        "temperature": 0.0,
        "max_tokens": 512,
        "max_input_chars": 32000,
        "thinking": True,
        "json_mode": True,
        "streaming": True,
        "screens": ["ALL"],
    },
    "posthoc_feedback": {
        "enabled": False,
        "mode": "human",
        "base_url": "https://api.llm.ustc.edu.cn/v1",
        "model": "deepseek-v4-pro",
        "api_key_file": "deepseek_api.txt",
        "timeout_seconds": 180.0,
        "max_retries": 1,
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
        "max_analysis_input_chars": 96000,
        "max_analysis_tokens": 1800,
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
    },
}

WINDOW_GEOMETRY_PATTERN = re.compile(r"^\d+x\d+[+-]\d+[+-]\d+$")


class ConfigError(ValueError):
    """Raised when config/agent.json is missing or contains unsafe settings."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a merged config and return it for convenient composition."""
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be 1")
    server = config["server"]
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

    _require_bool(policy["enabled"], "policy.enabled")
    if policy["runtime"] not in {"local", "remote_inference", "remote_online_rl"}:
        raise ConfigError("policy.runtime must be local, remote_inference, or remote_online_rl")
    _require_string(model["base"], "policy.model.base")
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
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 5:
        raise ConfigError("teacher.max_retries must be an integer from 0 to 5")
    _require_number(teacher["temperature"], "teacher.temperature", 0.0, 2.0)
    max_tokens = teacher["max_tokens"]
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 32 <= max_tokens <= 8192:
        raise ConfigError("teacher.max_tokens must be an integer from 32 to 8192")
    max_input_chars = teacher["max_input_chars"]
    if isinstance(max_input_chars, bool) or not isinstance(max_input_chars, int) or not 0 <= max_input_chars <= 262144:
        raise ConfigError("teacher.max_input_chars must be an integer from 0 to 262144")
    _require_bool(teacher["thinking"], "teacher.thinking")
    _require_bool(teacher["json_mode"], "teacher.json_mode")
    _require_bool(teacher["streaming"], "teacher.streaming")
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
    if isinstance(posthoc_retries, bool) or not isinstance(posthoc_retries, int) or not 0 <= posthoc_retries <= 5:
        raise ConfigError("posthoc_feedback.max_retries must be an integer from 0 to 5")
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
    if not 4096 <= evolution["max_analysis_input_chars"] <= 262144:
        raise ConfigError("prompt_evolution.max_analysis_input_chars must be from 4096 to 262144")
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
    for name in ("status", "control", "online_rl"):
        window = overlay[name]
        _require_string(window["geometry"], f"overlay.{name}.geometry")
        if not WINDOW_GEOMETRY_PATTERN.fullmatch(window["geometry"]):
            raise ConfigError(f"overlay.{name}.geometry must be WIDTHxHEIGHT+X+Y")
        _require_number(window["opacity"], f"overlay.{name}.opacity", 0.2, 1.0)
    divider_columns = overlay["status"]["divider_columns"]
    if isinstance(divider_columns, bool) or not isinstance(divider_columns, int) or not 12 <= divider_columns <= 78:
        raise ConfigError("overlay.status.divider_columns must be an integer from 12 to 78")
    return config


def load_agent_config(root: Path | None = None, config_file: str | Path | None = None) -> dict[str, Any]:
    """Load the selected config with strict keys so typos cannot silently alter runs."""
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
    return validate_config(_merge(DEFAULT_CONFIG, provided))
