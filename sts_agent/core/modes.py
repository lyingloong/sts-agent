"""Validation and normalization for the direct framework configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CONTROL_MODES = frozenset({"manual", "candidate_assist", "autonomous"})
POLICY_BACKENDS = frozenset({"none", "local_qwen", "remote_qwen", "remote_teacher", "mock"})
LEARNING_METHODS = frozenset({"none", "collect_human", "teacher_sft", "dagger", "prompt_evolution", "online_rl"})


@dataclass(frozen=True, slots=True)
class FrameworkMode:
    game: str
    control_mode: str
    policy_backend: str
    learning_method: str


def resolve_framework_mode(configured: dict[str, Any]) -> dict[str, str]:
    """Validate and return the direct framework dimensions."""

    if not isinstance(configured, dict):
        raise ValueError("framework must be an object")
    result = {
        "game": str(configured.get("game") or ""),
        "control_mode": str(configured.get("control_mode") or ""),
        "policy_backend": str(configured.get("policy_backend") or ""),
        "learning_method": str(configured.get("learning_method") or ""),
    }
    if result["game"] != "sts1":
        raise ValueError("framework.game must be 'sts1'")
    if result["control_mode"] not in CONTROL_MODES:
        raise ValueError("framework.control_mode is invalid")
    if result["policy_backend"] not in POLICY_BACKENDS:
        raise ValueError("framework.policy_backend is invalid")
    if result["learning_method"] not in LEARNING_METHODS:
        raise ValueError("framework.learning_method is invalid")
    return result
