"""Campfire facts and read-only presentation of indexed runtime options."""

from __future__ import annotations

from typing import Any


CAMPFIRE_ACTIVITY_RULE = (
    "本次营火只能完成一项营火活动，例如休息、锻造或回忆。"
    "完成其中一项后，其余营火活动将不再可选。"
    "若活动尚未完成，且界面允许取消返回，则可以重新选择。"
    "使用或丢弃药水不计入营火活动。"
)


def normalize_option_details(value: Any, choices: Any, current_choices: Any) -> list[dict[str, Any]] | None:
    """Keep only model fields, with no guessed index or effect association."""
    # The room's parallel list must match the actual CHOOSE space as a whole.
    # A partial index match cannot make a stale/reordered snapshot trustworthy.
    if (not isinstance(value, list) or not isinstance(choices, list)
            or not isinstance(current_choices, list) or choices != current_choices
            or len(value) != len(choices)):
        return None
    result = []
    for index, (item, choice) in enumerate(zip(value, choices)):
        if (not isinstance(item, dict) or type(item.get("choice_index")) is not int
                or item["choice_index"] != index or not isinstance(choice, str)
                or item.get("choice") != choice):
            return None
        result.append({
            "choice_index": index,
            "choice": choice,
            "name": item.get("name") if isinstance(item.get("name"), str) else None,
            "description": item.get("description") if isinstance(item.get("description"), str) else None,
            "usable": item.get("usable") if type(item.get("usable")) is bool else None,
        })
    return result


def choice_label(state: dict[str, Any], index: int, choice: Any) -> str:
    """Explain an existing CHOOSE action, never add/remove/remap an action."""
    fallback = str(choice)
    screen = state.get("screen_state")
    if not isinstance(screen, dict):
        return fallback
    details = normalize_option_details(
        screen.get("rest_option_details"), screen.get("rest_options"), state.get("choice_list"))
    if details is None or not 0 <= index < len(details):
        return fallback
    detail = details[index]
    if detail["choice"] != choice or detail["usable"] is False:
        return fallback
    name = " ".join((detail["name"] or fallback).split()) or fallback
    description = " / ".join(line.strip() for line in (detail["description"] or "").splitlines() if line.strip())
    return f"{name}——{description}" if description else name
