"""Versioned, inspectable reward shaping for complete StS episodes."""

from __future__ import annotations

from typing import Any


DEFAULT_REWARD_WEIGHTS = {
    "victory": 1000.0,
    "act": 100.0,
    "floor": 1.0,
    "hp_fraction": 10.0,
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def episode_reward(terminal_state: dict[str, Any], weights: dict[str, Any] | None = None) -> dict[str, float | bool]:
    """Return reward components instead of hiding shaping inside the trainer."""

    effective = {**DEFAULT_REWARD_WEIGHTS, **(weights or {})}
    screen = str(terminal_state.get("screen_type") or "").upper()
    victory = screen == "VICTORY"
    act = max(0.0, _number(terminal_state.get("act")))
    floor = max(0.0, _number(terminal_state.get("floor")))
    current_hp = max(0.0, _number(terminal_state.get("current_hp")))
    max_hp = max(0.0, _number(terminal_state.get("max_hp")))
    hp_fraction = current_hp / max_hp if max_hp else 0.0
    components = {
        "victory": float(effective["victory"]) if victory else 0.0,
        "act": float(effective["act"]) * act,
        "floor": float(effective["floor"]) * floor,
        "hp_fraction": float(effective["hp_fraction"]) * hp_fraction,
    }
    return {"total": sum(components.values()), "won": victory, **components}
