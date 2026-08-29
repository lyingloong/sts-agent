"""Human reflection and teacher attribution for masked online GRPO updates.

The module deliberately stores only short human feedback and structured action
attributions.  Teacher chain-of-thought and raw completions are never written
to an episode artifact or sent to the remote training worker.
"""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from teacher_client import RemoteTeacherClient, TeacherError, compact_json


PROMPT_VERSION = "posthoc-attribution-v1"
SCHEMA_VERSION = "sts-posthoc-attribution-0.1"


class PosthocFeedbackError(RuntimeError):
    """A recoverable failure in the optional end-of-episode feedback path."""


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def compact_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _compact_object(value: Any, *, depth: int = 0) -> Any:
    """Keep an action ledger informative without sending full raw traces."""

    if depth >= 4:
        return "<nested>"
    if isinstance(value, str):
        return compact_text(value, 300)
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        result = [_compact_object(item, depth=depth + 1) for item in value[:16]]
        if len(value) > 16:
            result.append(f"...<{len(value) - 16} more>")
        return result
    if isinstance(value, dict):
        keys = (
            "screen_type",
            "room_phase",
            "action_phase",
            "act",
            "floor",
            "class",
            "ascension_level",
            "current_hp",
            "max_hp",
            "gold",
            "potions",
            "relics",
            "combat",
            "screen_state",
            "selection_context",
            "event_history",
            "recent_actions",
            "run_memory",
        )
        return {
            key: _compact_object(value[key], depth=depth + 1)
            for key in keys
            if key in value and value[key] not in (None, [], {})
        }
    return compact_text(value, 200)


def terminal_summary(terminal_state: dict[str, Any]) -> dict[str, Any]:
    """Return outcome facts used by the human and teacher, not a reward label."""

    fields = ("screen_type", "act", "floor", "current_hp", "max_hp", "gold", "class", "ascension_level")
    return {key: terminal_state.get(key) for key in fields if key in terminal_state}


def action_ledger(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Make teacher-visible, stable references to sampled policy actions."""

    result: list[dict[str, Any]] = []
    for step in episode.get("steps", []):
        if not isinstance(step, dict):
            continue
        selected_alias = str(step.get("selected_alias") or "")
        selected = next(
            (
                action
                for action in step.get("legal_actions", [])
                if isinstance(action, dict) and str(action.get("id") or "") == selected_alias
            ),
            None,
        )
        if selected is None:
            continue
        result.append(
            {
                "step_id": step.get("step_id"),
                "screen_type": step.get("observation", {}).get("screen_type")
                if isinstance(step.get("observation"), dict)
                else None,
                "selected_action": {
                    "alias": selected_alias,
                    "label": compact_text(selected.get("label") or selected_alias, 240),
                },
                "state": _compact_object(step.get("observation")),
            }
        )
    return result


def build_prompt(
    reflection: dict[str, Any],
    episode: dict[str, Any],
    *,
    max_candidates_per_direction: int,
    max_input_chars: int,
) -> list[dict[str, str]]:
    """Build a direction-only post-hoc attribution request in Chinese."""

    system = (
        "你是《杀戮尖塔》策略训练的事后归因标注器。人类的复盘是唯一的价值方向："
        "只定位日志中已经执行的模型动作，不要自行编造动作、奖励或隐藏状态。"
        "请分别找出值得强化和应当弱化的关键动作；同一步不得同时出现在两类。"
        "证据只写简短可审计事实，不输出推理过程、长篇攻略或 Markdown。"
        "严格返回一个 JSON 对象："
        '{"reinforce":[{"step_id":12,"confidence":0.86,"severity":2,"evidence":"..."}],'
        '"weaken":[{"step_id":18,"confidence":0.91,"severity":3,"evidence":"..."}]}'
        ". confidence 在 0 到 1；severity 为 1、2 或 3；每个数组最多 "
        f"{max_candidates_per_direction} 项。人类某一栏为空时，对应数组可以为空。"
    )
    user_value = {
        "human_reflection": {
            "reinforce": reflection.get("reinforce", ""),
            "weaken": reflection.get("weaken", ""),
        },
        "terminal_state": terminal_summary(episode.get("terminal_state", {})),
        "sampled_action_ledger": action_ledger(episode),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": compact_json(user_value, limit=max_input_chars)},
    ]


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PosthocFeedbackError("teacher attribution response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise PosthocFeedbackError("teacher attribution response must be a JSON object")
    return value


def _candidate(
    value: Any,
    *,
    direction: str,
    valid_steps: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(value, dict):
        return None, "candidate is not an object"
    step_id = value.get("step_id")
    if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id not in valid_steps:
        return None, "step_id does not identify a sampled policy action"
    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
        return None, "confidence is not finite"
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        return None, "confidence must be within [0, 1]"
    severity = value.get("severity")
    if isinstance(severity, bool) or not isinstance(severity, int) or severity not in {1, 2, 3}:
        return None, "severity must be 1, 2, or 3"
    action = valid_steps[step_id]
    return (
        {
            "step_id": step_id,
            "direction": direction,
            "confidence": confidence,
            "severity": severity,
            "evidence": compact_text(value.get("evidence"), 500),
            "selected_alias": action["selected_alias"],
            "action_label": action["action_label"],
        },
        None,
    )


def validate_attributions(
    value: dict[str, Any],
    episode: dict[str, Any],
    *,
    min_confidence: float,
    max_applied_per_direction: int,
    allowed_directions: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Validate, deduplicate and confidence-filter teacher action references."""

    valid_steps: dict[int, dict[str, Any]] = {}
    for step in episode.get("steps", []):
        if not isinstance(step, dict):
            continue
        step_id = step.get("step_id")
        if not isinstance(step_id, int):
            continue
        selected_alias = str(step.get("selected_alias") or "")
        action = next(
            (
                item
                for item in step.get("legal_actions", [])
                if isinstance(item, dict) and str(item.get("id") or "") == selected_alias
            ),
            {},
        )
        valid_steps[step_id] = {
            "selected_alias": selected_alias,
            "action_label": compact_text(action.get("label") or selected_alias, 240),
        }

    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    by_step: dict[int, dict[str, Any]] = {}
    conflicted_steps: set[int] = set()
    for direction in ("reinforce", "weaken"):
        candidates = value.get(direction, [])
        if not isinstance(candidates, list):
            rejected.append({"direction": direction, "reason": "direction payload is not a list"})
            continue
        if allowed_directions is not None and direction not in allowed_directions:
            rejected.extend(
                {"direction": direction, "reason": "human reflection did not authorize this direction"}
                for _ in candidates
            )
            continue
        for raw in candidates:
            candidate, reason = _candidate(raw, direction=direction, valid_steps=valid_steps)
            if candidate is None:
                rejected.append({"direction": direction, "reason": reason or "invalid candidate"})
                continue
            if candidate["confidence"] < min_confidence:
                rejected.append(
                    {
                        "step_id": candidate["step_id"],
                        "direction": direction,
                        "reason": f"confidence below configured threshold {min_confidence:.2f}",
                    }
                )
                continue
            if candidate["step_id"] in conflicted_steps:
                rejected.append({"step_id": candidate["step_id"], "direction": direction, "reason": "conflicting directions"})
                continue
            previous = by_step.get(candidate["step_id"])
            if previous is not None and previous["direction"] != direction:
                rejected.extend(
                    (
                        {"step_id": candidate["step_id"], "direction": previous["direction"], "reason": "conflicting directions"},
                        {"step_id": candidate["step_id"], "direction": direction, "reason": "conflicting directions"},
                    )
                )
                by_step.pop(candidate["step_id"], None)
                conflicted_steps.add(candidate["step_id"])
                continue
            if previous is None or (candidate["confidence"], candidate["severity"]) > (
                previous["confidence"],
                previous["severity"],
            ):
                by_step[candidate["step_id"]] = candidate

    for direction in ("reinforce", "weaken"):
        selected = sorted(
            (item for item in by_step.values() if item["direction"] == direction),
            key=lambda item: (item["confidence"], item["severity"]),
            reverse=True,
        )
        kept = selected[:max_applied_per_direction]
        applied.extend(kept)
        for item in selected[max_applied_per_direction:]:
            rejected.append({"step_id": item["step_id"], "direction": direction, "reason": "per-direction cap"})
    return {"applied": applied, "rejected": rejected}


class PosthocFeedbackClient:
    """Optional local teacher pass that produces masked-update annotations."""

    def __init__(self, project_root: Path, config: dict[str, Any]) -> None:
        self.enabled = bool(config["enabled"])
        self.max_reflection_chars = int(config["max_reflection_chars"])
        self.max_candidates_per_direction = int(config["max_candidates_per_direction"])
        self.max_applied_per_direction = int(config["max_applied_per_direction"])
        self.min_confidence = float(config["min_confidence"])
        self.remote = RemoteTeacherClient(
            project_root,
            dict(config) | {"thinking": False, "json_mode": True, "streaming": False, "screens": ["ALL"]},
        )
        self.last_error: str | None = None

    @property
    def model(self) -> str:
        return self.remote.model

    def preflight(self) -> str | None:
        if not self.enabled:
            return None
        error = self.remote.preflight()
        self.last_error = error
        return error

    def normalize_reflection(self, reinforce: Any, weaken: Any) -> dict[str, Any]:
        positive = compact_text(reinforce, self.max_reflection_chars)
        negative = compact_text(weaken, self.max_reflection_chars)
        return {
            "reinforce": positive,
            "weaken": negative,
            "skipped": not positive and not negative,
        }

    def annotate(self, reflection: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "applied": [], "rejected": []}
        if reflection.get("skipped"):
            return {"status": "human_skipped", "applied": [], "rejected": []}
        messages = build_prompt(
            reflection,
            episode,
            max_candidates_per_direction=self.max_candidates_per_direction,
            max_input_chars=self.remote.max_input_chars,
        )
        try:
            response = self.remote.complete(
                messages,
                prompt_version=PROMPT_VERSION,
                thinking=False,
                json_mode=True,
                streaming=False,
            )
            parsed = _parse_json_object(str(response["output"]))
            selected = validate_attributions(
                parsed,
                episode,
                min_confidence=self.min_confidence,
                max_applied_per_direction=self.max_applied_per_direction,
                allowed_directions={
                    direction
                    for direction in ("reinforce", "weaken")
                    if str(reflection.get(direction) or "").strip()
                },
            )
        except TeacherError as exc:
            self.last_error = str(exc)
            raise PosthocFeedbackError(str(exc)) from exc
        self.last_error = None
        return {
            "status": "annotated",
            "model_id": response["model_id"],
            "prompt_version": response["prompt_version"],
            "latency_ms": response["latency_ms"],
            "attempt": response["attempt"],
            "usage": response["usage"],
            **selected,
        }


def write_artifact(path: Path, value: dict[str, Any]) -> None:
    """Durably write a local feedback artifact without exposing temporary data."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
