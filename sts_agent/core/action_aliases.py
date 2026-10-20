"""Public action aliases that retain visible positions without control identities."""

from __future__ import annotations

import re
from typing import Any


_POSITION_FIELDS = (
    "card_index", "target", "slot", "index", "choice_index", "hand_index",
    "grid_index", "potion_slot", "enemy_index", "monster_index", "reward_index",
)
_REWARD_REFERENCE_FIELDS = ("reward_ref", "linked_reward_ref", "source_reward_ref")


def model_action_alias(action: dict[str, Any], alias: str) -> dict[str, Any]:
    """Keep decision references while excluding commands, UUIDs and internal IDs.

    The canonical action remains in the caller's private alias lookup. Position
    zero and explicit unknowns are meaningful and must survive every path.
    """
    result: dict[str, Any] = {
        "id": alias,
        "label": str(action.get("label") or action.get("command") or action.get("kind") or alias),
    }
    if isinstance(action.get("kind"), str):
        result["kind"] = action["kind"]
    for key in _POSITION_FIELDS:
        if key in action and (action[key] is None or type(action[key]) is int):
            result[key] = action[key]
    for key in _REWARD_REFERENCE_FIELDS:
        value = action.get(key)
        if isinstance(value, str) and re.fullmatch(r"R[1-9]\d*", value):
            result[key] = value
    return result
