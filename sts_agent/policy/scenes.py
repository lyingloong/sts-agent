"""Shared scene classification for local policy, SFT and remote prompts.

Classification changes instructions only, never the recorded screen or actions.
"""
from __future__ import annotations

from typing import Any


SCENE_KEYS = frozenset({
    "MAP", "COMBAT", "CARD_REWARD", "COMBAT_CARD_SELECTION", "COMBAT_REWARD",
    "RELIC_PICKUP_CARD_SELECTION", "GRID_PURGE", "GRID_UPGRADE", "GRID_TRANSFORM",
    "GRID_SELECT", "GRID_CONFIRM", "HAND_SELECT", "SHOP_ROOM", "SHOP_SCREEN",
    "EVENT", "EVENT_MATCH", "REST", "CHEST", "BOSS_REWARD", "TRANSITION",
    "GAME_OVER", "VICTORY", "GENERIC",
})


def screen_key(observation: dict[str, Any]) -> str:
    screen = str(observation.get("screen_type") or "NONE").strip().upper()
    if screen == "SHOP":
        return "SHOP_SCREEN"
    if screen == "NONE" and str(observation.get("room_phase") or "").upper() == "COMBAT":
        return "COMBAT"
    return screen


def scene_key(observation: dict[str, Any]) -> str:
    screen = screen_key(observation)
    state = observation.get("screen_state")
    state = state if isinstance(state, dict) else {}
    context = observation.get("selection_context")
    context = context if isinstance(context, dict) else {}
    if screen == "CARD_REWARD":
        if context.get("kind") == "RELIC_PICKUP_CARD_SELECTION":
            return "RELIC_PICKUP_CARD_SELECTION"
        if (context.get("kind") == "COMBAT_GENERATED_CARD_CHOICE"
                or str(observation.get("room_phase") or "").upper() == "COMBAT"):
            return "COMBAT_CARD_SELECTION"
    if screen == "GRID":
        if state.get("confirm_screen_up") is True:
            return "GRID_CONFIRM"
        if context.get("kind") == "RELIC_PICKUP_CARD_SELECTION":
            return "RELIC_PICKUP_CARD_SELECTION"
        mode = str(state.get("selection_mode") or "").upper()
        return {
            "PURGE": "GRID_PURGE", "UPGRADE": "GRID_UPGRADE",
            "TRANSFORM": "GRID_TRANSFORM", "TRANSFORM_UPGRADE": "GRID_TRANSFORM",
        }.get(mode, "GRID_SELECT")
    if screen == "EVENT" and isinstance(state.get("match_game"), dict):
        return "EVENT_MATCH"
    return screen if screen in SCENE_KEYS else "GENERIC"
