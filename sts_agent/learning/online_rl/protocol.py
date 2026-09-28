"""Validation and durable primitives for the local-to-remote JSONL protocol."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


class ProtocolError(ValueError):
    """Raised when a peer sends an incomplete or internally inconsistent message."""


def require_object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{field} must be an object")
    return value


def require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{field} must be a non-empty string")
    return value


def start_key(run_config: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable initial condition required for a relative group."""

    character = require_string(run_config.get("character"), "run_config.character").upper()
    seed = require_string(run_config.get("seed"), "run_config.seed").upper()
    ascension = run_config.get("ascension")
    if isinstance(ascension, bool) or not isinstance(ascension, int) or not 0 <= ascension <= 20:
        raise ProtocolError("run_config.ascension must be an integer from 0 to 20")
    return {"character": character, "ascension": ascension, "seed": seed}


def validate_episode_payload(value: Any) -> dict[str, Any]:
    payload = require_object(value, "episode")
    episode_id = require_string(payload.get("episode_id"), "episode.episode_id")
    group_id = require_string(payload.get("group_id"), "episode.group_id")
    policy_version = require_string(payload.get("policy_version"), "episode.policy_version")
    run_config = require_object(payload.get("run_config"), "episode.run_config")
    key = start_key(run_config)
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ProtocolError("episode.steps must be a non-empty list")
    normalized_steps: list[dict[str, Any]] = []
    seen_steps: set[int] = set()
    for index, step in enumerate(steps):
        item = require_object(step, f"episode.steps[{index}]")
        step_id = item.get("step_id")
        if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0 or step_id in seen_steps:
            raise ProtocolError(f"episode.steps[{index}].step_id must be a unique non-negative integer")
        seen_steps.add(step_id)
        require_object(item.get("observation"), f"episode.steps[{index}].observation")
        actions = item.get("legal_actions")
        if not isinstance(actions, list) or not actions:
            raise ProtocolError(f"episode.steps[{index}].legal_actions must be a non-empty list")
        selected_alias = require_string(item.get("selected_alias"), f"episode.steps[{index}].selected_alias")
        aliases = {str(action.get("id") or "") for action in actions if isinstance(action, dict)}
        if selected_alias not in aliases:
            raise ProtocolError(f"episode.steps[{index}].selected_alias is not legal")
        try:
            old_logprob = float(item.get("old_logprob"))
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"episode.steps[{index}].old_logprob must be numeric") from exc
        if old_logprob > 0:
            raise ProtocolError(f"episode.steps[{index}].old_logprob must be a log probability")
        normalized_steps.append(
            {
                "step_id": step_id,
                "observation": item["observation"],
                "legal_actions": actions,
                "selected_alias": selected_alias,
                "old_logprob": old_logprob,
            }
        )
    terminal_state = require_object(payload.get("terminal_state"), "episode.terminal_state")
    attributions = payload.get("attributions", [])
    if not isinstance(attributions, list):
        raise ProtocolError("episode.attributions must be a list")
    normalized_attributions: list[dict[str, Any]] = []
    attributed_steps: set[int] = set()
    for index, attribution in enumerate(attributions):
        item = require_object(attribution, f"episode.attributions[{index}]")
        step_id = item.get("step_id")
        if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id not in seen_steps:
            raise ProtocolError(f"episode.attributions[{index}].step_id must identify a sampled step")
        if step_id in attributed_steps:
            raise ProtocolError(f"episode.attributions[{index}].step_id is duplicated")
        attributed_steps.add(step_id)
        direction = item.get("direction")
        if direction not in {"reinforce", "weaken"}:
            raise ProtocolError(f"episode.attributions[{index}].direction must be reinforce or weaken")
        confidence = item.get("confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
            raise ProtocolError(f"episode.attributions[{index}].confidence must be finite")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ProtocolError(f"episode.attributions[{index}].confidence must be within [0, 1]")
        severity = item.get("severity")
        if isinstance(severity, bool) or not isinstance(severity, int) or severity not in {1, 2, 3}:
            raise ProtocolError(f"episode.attributions[{index}].severity must be 1, 2, or 3")
        normalized_attributions.append(
            {
                "step_id": step_id,
                "direction": direction,
                "confidence": confidence,
                "severity": severity,
            }
        )
    return {
        "episode_id": episode_id,
        "group_id": group_id,
        "policy_version": policy_version,
        "run_config": key,
        "steps": normalized_steps,
        "terminal_state": terminal_state,
        "attributions": normalized_attributions,
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
