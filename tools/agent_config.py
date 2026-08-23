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
        "model": "Qwen/Qwen3-0.6B",
        "endpoint": "https://huggingface.co",
        "top_k": 3,
        "timeout_seconds": 8.0,
        "start_timeout_seconds": 300.0,
        "screens": ["ALL"],
        "python": "auto",
        "mock": False,
        "debug": False,
        "confidence_auto": {
            "enabled": True,
            "min_probability": 0.85,
        },
    },
    "overlay": {
        "always_on_top": True,
        "frameless": True,
        "status": {
            "geometry": "540x680+24+84",
            "opacity": 0.96,
            "divider_columns": 36,
        },
        "control": {
            "geometry": "500x440+580+84",
            "opacity": 0.97,
        },
    },
}

WINDOW_GEOMETRY_PATTERN = re.compile(r"^\d+x\d+[+-]\d+[+-]\d+$")


class ConfigError(ValueError):
    """Raised when config/agent.json is missing or contains unsafe settings."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def config_path(root: Path | None = None) -> Path:
    return (root or project_root()) / "config" / "agent.json"


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
    confidence = policy["confidence_auto"]
    overlay = config["overlay"]

    _require_string(server["host"], "server.host")
    port = server["port"]
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ConfigError("server.port must be an integer from 1 to 65535")

    _require_bool(policy["enabled"], "policy.enabled")
    _require_string(policy["model"], "policy.model")
    _require_string(policy["endpoint"], "policy.endpoint")
    if not str(policy["endpoint"]).startswith(("https://", "http://")):
        raise ConfigError("policy.endpoint must start with http:// or https://")
    top_k = policy["top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ConfigError("policy.top_k must be an integer >= 1")
    _require_number(policy["timeout_seconds"], "policy.timeout_seconds", 0.1)
    _require_number(policy["start_timeout_seconds"], "policy.start_timeout_seconds", 1.0)
    _require_string(policy["python"], "policy.python")
    _require_bool(policy["mock"], "policy.mock")
    _require_bool(policy["debug"], "policy.debug")
    screens = policy["screens"]
    if not isinstance(screens, list) or not screens or not all(isinstance(item, str) for item in screens):
        raise ConfigError("policy.screens must be a non-empty list of strings")
    unknown_screens = {item.strip().upper() for item in screens} - {"ALL", "COMBAT"}
    if unknown_screens:
        raise ConfigError(f"policy.screens contains unsupported values: {', '.join(sorted(unknown_screens))}")
    _require_bool(confidence["enabled"], "policy.confidence_auto.enabled")
    _require_number(confidence["min_probability"], "policy.confidence_auto.min_probability", 0.0, 1.0)
    _require_bool(overlay["always_on_top"], "overlay.always_on_top")
    _require_bool(overlay["frameless"], "overlay.frameless")
    for name in ("status", "control"):
        window = overlay[name]
        _require_string(window["geometry"], f"overlay.{name}.geometry")
        if not WINDOW_GEOMETRY_PATTERN.fullmatch(window["geometry"]):
            raise ConfigError(f"overlay.{name}.geometry must be WIDTHxHEIGHT+X+Y")
        _require_number(window["opacity"], f"overlay.{name}.opacity", 0.2, 1.0)
    divider_columns = overlay["status"]["divider_columns"]
    if isinstance(divider_columns, bool) or not isinstance(divider_columns, int) or not 12 <= divider_columns <= 78:
        raise ConfigError("overlay.status.divider_columns must be an integer from 12 to 78")
    return config


def load_agent_config(root: Path | None = None) -> dict[str, Any]:
    """Load config/agent.json with strict keys so typos cannot silently alter runs."""
    path = config_path(root)
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
