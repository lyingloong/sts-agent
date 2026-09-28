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

from sts_agent.policy.teacher import RemoteTeacherClient, TeacherError, compact_json


PROMPT_VERSION = "posthoc-attribution-v3"
SCHEMA_VERSION = "sts-posthoc-attribution-0.1"

# Observations have already been normalized to public game data before this
# module receives them. These fields are nevertheless derived from the local
# static knowledge bases rather than from the changing run state, so they add
# a large amount of repeated prompt text without helping post-hoc attribution.
POSTHOC_STATIC_FIELDS = frozenset(
    {
        "description",
        "raw_description",
        "flavor",
        "flavor_text",
        "effect",
        "effects",
    }
)
POSTHOC_LIST_LIMITS = {
    "nodes": 128,
    "cards": 80,
    "hand": 16,
    "monsters": 12,
    "powers": 24,
    "potions": 12,
    "relics": 48,
    "rewards": 24,
    "choices": 48,
    "options": 48,
    "recent_actions": 8,
    "event_history": 12,
    "children": 8,
    "parents": 8,
}


class PosthocFeedbackError(RuntimeError):
    """A recoverable failure in the optional end-of-episode feedback path."""


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def compact_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


def _posthoc_value(value: Any, *, key: str | None = None, depth: int = 0) -> Any:
    """Retain public, dynamic state while bounding pathological payloads."""

    if depth >= 9:
        return "<nested>"
    if isinstance(value, str):
        limit = 1600 if key in {"body_text", "text"} else 600
        return compact_text(value, limit)
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        limit = POSTHOC_LIST_LIMITS.get(key or "", 64)
        result = [_posthoc_value(item, key=key, depth=depth + 1) for item in value[:limit]]
        if len(value) > limit:
            result.append(f"...<{len(value) - limit} more>")
        return result
    if isinstance(value, dict):
        return {
            item_key: _posthoc_value(item, key=item_key, depth=depth + 1)
            for item_key, item in value.items()
            if item_key not in POSTHOC_STATIC_FIELDS and item not in (None, [], {})
        }
    return compact_text(value, 200)


def _posthoc_state_snapshot(observation: Any, map_ids: dict[str, str]) -> dict[str, Any]:
    """Build a detailed dynamic snapshot and deduplicate repeated Act maps."""

    if not isinstance(observation, dict):
        return {}
    source = dict(observation)
    raw_map = source.pop("map", None)
    result = _posthoc_value(source)
    if not isinstance(result, dict) or not isinstance(raw_map, dict):
        return result if isinstance(result, dict) else {}

    raw_nodes = raw_map.get("nodes")
    map_summary = _posthoc_value(
        {key: item for key, item in raw_map.items() if key != "nodes"},
        key="map",
    )
    map_summary = map_summary if isinstance(map_summary, dict) else {}
    if isinstance(raw_nodes, list):
        signature = json.dumps(raw_nodes, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        map_id = map_ids.get(signature)
        if map_id is None:
            map_id = f"map-{len(map_ids) + 1}"
            map_ids[signature] = map_id
            map_summary["nodes"] = _posthoc_value(raw_nodes, key="nodes")
        map_summary["map_id"] = map_id
    if map_summary:
        result["map"] = map_summary
    return result


def terminal_summary(terminal_state: dict[str, Any]) -> dict[str, Any]:
    """Return outcome facts used by the human and teacher, not a reward label."""

    fields = ("screen_type", "act", "floor", "current_hp", "max_hp", "gold", "class", "ascension_level")
    return {key: terminal_state.get(key) for key in fields if key in terminal_state}


def action_review_from_trace(trace_path: Path) -> list[dict[str, Any]]:
    """Build a compact chronological review from every accepted trace action.

    Unlike the policy ledger, this includes controller-owned environment actions
    so the human can understand reward/shop transitions while reflecting on a
    full game.  It deliberately excludes observations and legal-action sets.
    """

    source_labels = {
        "online_policy_sample": "模型",
        "environment_auto": "自动",
        "teacher_auto": "教师",
        "model_auto": "模型",
        "human_override": "人工",
    }
    result: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("accepted") is not True:
                continue
            action = record.get("action")
            action = action if isinstance(action, dict) else {}
            decision = record.get("decision")
            decision = decision if isinstance(decision, dict) else {}
            state = record.get("pre_state")
            state = state if isinstance(state, dict) else {}
            raw_source = str(decision.get("source") or "")
            result.append(
                {
                    "step_id": record.get("step_id"),
                    "act": state.get("act"),
                    "floor": state.get("floor"),
                    "screen_type": record.get("screen_type") or state.get("screen_type"),
                    "source": source_labels.get(raw_source, "其他"),
                    "action": compact_text(
                        action.get("label") or action.get("command") or action.get("id") or "未知动作",
                        300,
                    ),
                }
            )
    return result


def action_ledger(episode: dict[str, Any]) -> list[dict[str, Any]]:
    """Make detailed, teacher-visible references to sampled policy actions."""

    result: list[dict[str, Any]] = []
    map_ids: dict[str, str] = {}
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
                "legal_actions": [
                    {
                        "alias": str(action.get("id") or ""),
                        "label": compact_text(action.get("label") or action.get("id") or "", 240),
                    }
                    for action in step.get("legal_actions", [])
                    if isinstance(action, dict) and str(action.get("id") or "")
                ],
                "state": _posthoc_state_snapshot(step.get("observation"), map_ids),
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

    feedback_mode = str(reflection.get("mode") or "human").strip().lower()
    if feedback_mode not in {"human", "auto"}:
        feedback_mode = "human"
    direction_source = (
        "当前为自动归因模式。终局死亡、低楼层、胜利或奖励数值本身不是任何单步的正负证据，"
        "不得把整局结果平均归因给前面的动作。先从账本找出存在明确局部反事实的少数步骤："
        "所选动作与当时其他合法动作相比，能从公开状态直接看出更差或更好的即时后果、不可逆后果，"
        "或对紧接着的战斗/路线产生明确影响。"
        "弱化动作必须在 evidence 中指出一个当时合法的替代动作或明确违反的约束；"
        "强化动作必须指出所选动作带来的已记录收益或避免的具体风险。"
        "不因动作名字、稀有度、一般攻略、事后随机结果，或单纯出现 END_TURN/SKIP/删牌/拿牌而归因。"
        "不要凑满名额：通常每个方向 0-3 条；只有相互独立、证据同样明确的重复错误才可更多。"
        "confidence>=0.90 仅用于状态直接矛盾或替代关系明确的步骤；0.75-0.89 用于证据强但仍有不确定性的步骤；"
        "低于 0.75 的候选不要返回。severity=3 仅用于濒死、明显可避免的大额掉血、不可逆关键选择或明确改变近期通关机会的步骤。"
        if feedback_mode == "auto"
        else "人类复盘是唯一的价值方向：只在其明确授权的方向中定位关键动作；"
        "证据仍必须锚定当时的公开状态和动作账本，不能用终局结果替代具体因果。"
    )
    system = (
        "你是《杀戮尖塔》策略训练的事后归因标注器。"
        f"{direction_source}"
        "只定位日志中已经执行的模型动作，不要自行编造动作、奖励或隐藏状态。"
        "每个步骤都给出了所选动作、当时全部合法动作和该步动态公开状态；"
        "state 中不包含静态知识库的卡牌、遗物、药水或能力描述。"
        "请分别找出值得强化和应当弱化的关键动作；同一步不得同时出现在两类。"
        "将一次战斗、地图选择、商店/奖励、事件或不可逆选牌分别评估；"
        "对战斗动作优先核对当回合能量、敌方可见意图、血量/格挡、手牌与结束回合后的风险；"
        "对非战斗动作优先核对可见奖励、金币、牌组、路线、首领和 decision_constraints。"
        "evidence 必须是简短可审计事实，最好包含“所选动作 + 当时状态/合法替代 + 可观察后果”；"
        "不输出推理过程、长篇攻略或 Markdown。"
        "严格返回一个 JSON 对象："
        '{"reinforce":[{"step_id":12,"confidence":0.86,"severity":2,"evidence":"..."}],'
        '"weaken":[{"step_id":18,"confidence":0.91,"severity":3,"evidence":"..."}]}'
        ". confidence 在 0 到 1；severity 为 1、2 或 3；每个数组最多 "
        f"{max_candidates_per_direction} 项。human 模式中人类某一栏为空时，对应数组应为空；auto 模式可独立判断两类。"
    )
    user_value = {
        "feedback_mode": feedback_mode,
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
        self.mode = str(config["mode"])
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
            "mode": "human",
            "reinforce": positive,
            "weaken": negative,
            "skipped": not positive and not negative,
        }

    def automatic_reflection(self) -> dict[str, Any]:
        """Describe a no-human-input attribution pass without fabricating text."""
        return {
            "mode": "auto",
            "reinforce": "",
            "weaken": "",
            "skipped": False,
            "reason": "teacher_auto",
        }

    def annotate(self, reflection: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "applied": [], "rejected": []}
        if reflection.get("skipped"):
            return {"status": "human_skipped", "applied": [], "rejected": []}
        feedback_mode = str(reflection.get("mode") or "human").strip().lower()
        if feedback_mode not in {"human", "auto"}:
            raise PosthocFeedbackError("feedback mode must be human or auto")
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
                allowed_directions=(
                    None
                    if feedback_mode == "auto"
                    else {
                        direction
                        for direction in ("reinforce", "weaken")
                        if str(reflection.get(direction) or "").strip()
                    }
                ),
            )
        except TeacherError as exc:
            self.last_error = str(exc)
            raise PosthocFeedbackError(str(exc)) from exc
        self.last_error = None
        return {
            "status": "annotated_auto" if feedback_mode == "auto" else "annotated",
            "feedback_mode": feedback_mode,
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
