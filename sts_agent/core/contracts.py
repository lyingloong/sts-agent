"""Stable data contracts for game observations and policy decisions.

The live controller still stores JSON dictionaries because that is the wire
format used by CommunicationMod. These helpers validate and normalize the
boundary without forcing the runtime to instantiate large dataclass graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


Observation = dict[str, Any]


class Action(TypedDict, total=False):
    id: str
    action_id: str
    label: str
    kind: str
    command: str
    target: Any
    payload: dict[str, Any]


class PolicyCandidate(TypedDict, total=False):
    action_id: str
    label: str
    rank: int
    probability: float
    score: float | None
    logprob: float


class PolicyResponse(TypedDict, total=False):
    type: str
    request_id: str | None
    model_id: str
    candidates: list[PolicyCandidate]
    selected_action_id: str
    error: str


def action_id(action: dict[str, Any]) -> str:
    """Return the stable short alias used by policy and trace records."""

    return str(action.get("id") or action.get("action_id") or "")


def validate_observation(value: Any) -> Observation:
    if not isinstance(value, dict):
        raise ValueError("observation must be an object")
    screen_type = value.get("screen_type")
    if screen_type is not None and not isinstance(screen_type, str):
        raise ValueError("observation.screen_type must be a string when present")
    return value


def validate_action_list(value: Any) -> list[Action]:
    if not isinstance(value, list):
        raise ValueError("legal_actions must be a list")
    result: list[Action] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"legal_actions[{index}] must be an object")
        alias = action_id(item)
        if not alias:
            raise ValueError(f"legal_actions[{index}] has no action id")
        if alias in seen:
            raise ValueError(f"duplicate legal action id: {alias}")
        seen.add(alias)
        result.append(item)
    return result


def validate_policy_response(value: Any, legal_actions: list[dict[str, Any]] | None = None) -> PolicyResponse:
    if not isinstance(value, dict):
        raise ValueError("policy response must be an object")
    candidates = value.get("candidates", [])
    if candidates is not None and not isinstance(candidates, list):
        raise ValueError("policy response candidates must be a list")
    legal_ids = {action_id(item) for item in (legal_actions or [])}
    for index, candidate in enumerate(candidates or []):
        if not isinstance(candidate, dict):
            raise ValueError(f"policy response candidates[{index}] must be an object")
        candidate_id = str(candidate.get("action_id") or "")
        if legal_ids and candidate_id not in legal_ids:
            raise ValueError(f"policy response contains non-legal action: {candidate_id}")
    selected = value.get("selected_action_id")
    if selected is not None and legal_ids and str(selected) not in legal_ids:
        raise ValueError(f"policy response selected action is not legal: {selected}")
    return value


@dataclass(slots=True)
class DecisionRecord:
    """Serializable one-step decision contract used by all learning methods."""

    step_id: int
    observation: Observation
    legal_actions: list[Action]
    selected_action_id: str | None = None
    source: str = "unknown"
    candidates: list[PolicyCandidate] = field(default_factory=list)
    outcome: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        validate_observation(self.observation)
        validate_action_list(self.legal_actions)
        legal_ids = {action_id(item) for item in self.legal_actions}
        if self.selected_action_id and self.selected_action_id not in legal_ids:
            raise ValueError(f"selected action is not legal: {self.selected_action_id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "observation": self.observation,
            "legal_actions": self.legal_actions,
            "selected_action_id": self.selected_action_id,
            "source": self.source,
            "candidates": self.candidates,
            "outcome": self.outcome,
        }
