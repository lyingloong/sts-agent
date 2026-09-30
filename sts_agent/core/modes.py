"""Validation and normalization for the direct framework configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CONTROL_MODES = frozenset({"manual", "candidate_assist", "autonomous", "codex"})
POLICY_BACKENDS = frozenset({"none", "local_qwen", "remote_qwen", "remote_teacher", "mock"})
LEARNING_METHODS = frozenset({"none", "collect_human", "collect_codex", "teacher_sft", "dagger", "prompt_evolution", "online_rl"})

# User-facing profiles collapse the three low-level framework dimensions into
# one named choice. Keep the mapping here so an invalid combination cannot be
# assembled accidentally by editing a JSON profile.
MODE_FRAMEWORKS: dict[str, dict[str, str]] = {
    "collect_human": {"game": "sts1", "control_mode": "manual", "policy_backend": "none", "learning_method": "collect_human"},
    "collect_codex": {"game": "sts1", "control_mode": "codex", "policy_backend": "none", "learning_method": "collect_codex"},
    "candidate_assist_local": {"game": "sts1", "control_mode": "candidate_assist", "policy_backend": "local_qwen", "learning_method": "none"},
    "candidate_assist_remote": {"game": "sts1", "control_mode": "candidate_assist", "policy_backend": "remote_qwen", "learning_method": "none"},
    "teacher_sft": {"game": "sts1", "control_mode": "autonomous", "policy_backend": "remote_teacher", "learning_method": "teacher_sft"},
    "dagger_local": {"game": "sts1", "control_mode": "candidate_assist", "policy_backend": "local_qwen", "learning_method": "dagger"},
    "dagger_remote": {"game": "sts1", "control_mode": "candidate_assist", "policy_backend": "remote_qwen", "learning_method": "dagger"},
    "prompt_evolution": {"game": "sts1", "control_mode": "autonomous", "policy_backend": "remote_teacher", "learning_method": "prompt_evolution"},
    "local_inference": {"game": "sts1", "control_mode": "autonomous", "policy_backend": "local_qwen", "learning_method": "none"},
    "remote_inference": {"game": "sts1", "control_mode": "autonomous", "policy_backend": "remote_qwen", "learning_method": "none"},
    "online_rl": {"game": "sts1", "control_mode": "autonomous", "policy_backend": "remote_qwen", "learning_method": "online_rl"},
}


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
