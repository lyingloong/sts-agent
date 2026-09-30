"""Internal trajectory writer used by the CommunicationMod state bridge.

CommunicationMod supplies authoritative states and ActionRecorder sends
transaction markers. The bridge translates a matched transaction into the
small internal ``action_observed``/``step_resolved`` pair consumed here. This
module is intentionally not a standalone receiver or offline converter.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from sts_agent.runtime.controller import (
    ACTION_HISTORY_STORAGE_LIMIT,
    accepted_action_history_item,
    attach_runtime_context,
    game_state,
    normalize_state,
    remember_declined_combat_card_reward,
    route_policy_observation,
    semantic_actions,
    shop_visit_key,
)


def _message(state: Any, available: Any = None) -> dict[str, Any]:
    game = state if isinstance(state, dict) else {}
    commands = available.get("commands") if isinstance(available, dict) else []
    return {
        "in_game": bool(game),
        "ready_for_command": True,
        "available_commands": commands if isinstance(commands, list) else [],
        "game_state": game,
    }


def _token(value: Any) -> str:
    return "".join(character for character in str(value or "").casefold() if character.isalnum())


def _choice_keywords(action: dict[str, Any]) -> tuple[str, ...]:
    kind = str(action.get("kind") or "")
    fixed = {
        "card_reward_opened": ("card", "cards", "卡牌", "卡牌奖励"),
        "reward_card_claimed": ("card", "cards", "卡牌", "卡牌奖励"),
        "shop_opened": ("shop", "merchant", "商店", "商人"),
        "chest_opened": ("open", "chest", "打开", "开启", "宝箱"),
        "reward_gold_claimed": ("gold", "金币"),
        "reward_relic_claimed": ("relic", "遗物"),
        "reward_potion_claimed": ("potion", "药水"),
    }
    if kind in fixed:
        return fixed[kind]
    action_id = str(action.get("id") or "")
    if action_id.startswith("CAMPFIRE:"):
        option = action_id.partition(":")[2]
        return {
            "REST": ("rest", "sleep", "休息"),
            "SMITH": ("smith", "upgrade", "锻造", "升级"),
            "LIFT": ("lift", "举重"),
            "TOKE": ("toke", "purge", "移除"),
            "DIG": ("dig", "挖掘"),
            "RECALL": ("recall", "回忆"),
        }.get(option, ())
    return ()


def resolve_recorded_action(
    recorded: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool, str]:
    """Resolve a game-side semantic event to one current agent action."""
    action_id = str(recorded.get("id") or "")
    exact = next((item for item in legal_actions if str(item.get("id") or "") == action_id), None)
    if exact is not None:
        return copy.deepcopy(exact), True, "exact_id"

    state = state if isinstance(state, dict) else {}
    screen_state = state.get("screen_state") if isinstance(state.get("screen_state"), dict) else {}
    if str(recorded.get("kind") or "") == "map_node_selected":
        nodes = screen_state.get("next_nodes")
        if isinstance(nodes, list):
            for index, node in enumerate(nodes):
                if not isinstance(node, dict):
                    continue
                if node.get("x") == recorded.get("x") and node.get("y") == recorded.get("y"):
                    match_id = f"CHOOSE:index={index}"
                    match = next(
                        (item for item in legal_actions if str(item.get("id") or "") == match_id), None
                    )
                    if match is not None:
                        return copy.deepcopy(match), True, "map_coordinates"

    if str(recorded.get("kind") or "") == "card_selection_changed":
        cards_field = "hand" if str(recorded.get("screen") or "") == "hand" else "cards"
        selected_field = "selected" if cards_field == "hand" else "selected_cards"
        cards = screen_state.get(cards_field)
        selected_before = screen_state.get(selected_field)
        selected_after = recorded.get("selected_cards")

        def card_ids(values: Any) -> set[str]:
            if not isinstance(values, list):
                return set()
            return {
                str(item.get("uuid") or item.get("card_uuid"))
                for item in values
                if isinstance(item, dict) and (item.get("uuid") or item.get("card_uuid"))
            }

        changed = card_ids(selected_before) ^ card_ids(selected_after)
        if len(changed) == 1 and isinstance(cards, list):
            changed_id = next(iter(changed))
            index = next(
                (
                    index for index, card in enumerate(cards)
                    if isinstance(card, dict)
                    and str(card.get("uuid") or card.get("card_uuid")) == changed_id
                ),
                None,
            )
            if index is not None:
                match_id = f"CHOOSE:index={index}"
                match = next(
                    (item for item in legal_actions if str(item.get("id") or "") == match_id), None
                )
                if match is not None:
                    return copy.deepcopy(match), True, "selected_card_uuid"

    compatible_ids = {
        "PROCEED": ("PROCEED", "CONFIRM"),
        "RETURN": ("RETURN", "CANCEL"),
        "LEAVE": ("LEAVE", "RETURN", "CANCEL"),
    }.get(action_id, ())
    compatible = [item for item in legal_actions if str(item.get("id") or "") in compatible_ids]
    if len(compatible) == 1:
        return copy.deepcopy(compatible[0]), True, "compatible_control"

    fields = (
        "card_name", "card_id", "relic_name", "relic_id", "potion_name", "potion_id"
    )
    needles = {_token(recorded.get(name)) for name in fields if _token(recorded.get(name))}
    if needles:
        matches = [
            item for item in legal_actions
            if any(needle in _token(item.get("label")) for needle in needles)
        ]
        if len(matches) == 1:
            return copy.deepcopy(matches[0]), True, "entity_label"

    keywords = tuple(_token(value) for value in _choice_keywords(recorded))
    if keywords:
        matches = [
            item for item in legal_actions
            if any(keyword and keyword in _token(item.get("label")) for keyword in keywords)
        ]
        if len(matches) == 1:
            return copy.deepcopy(matches[0]), True, "choice_label"

    choices = [item for item in legal_actions if str(item.get("kind") or "") == "choose"]
    if len(choices) == 1 and (
        str(recorded.get("kind") or "").startswith(("reward_", "shop_"))
        or recorded.get("kind") in {"card_reward_opened", "chest_opened", "campfire_action"}
    ):
        return copy.deepcopy(choices[0]), True, "only_choice"

    unresolved = copy.deepcopy(recorded)
    unresolved.setdefault("label", recorded.get("kind") or action_id)
    return unresolved, False, "unresolved"


@dataclass
class RunContext:
    run_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    visited_shop_rooms: set[tuple[int, int]] = field(default_factory=set)
    declined_card_reward_keys: set[tuple[int, int]] = field(default_factory=set)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    output: TextIO | None = None
    output_path: Path | None = None


class ActionRecorderEnricher:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runs: dict[str, RunContext] = {}
        self.warned_missing_state: set[str] = set()

    def close(self) -> None:
        for context in self.runs.values():
            if context.output is not None:
                context.output.flush()
                context.output.close()
                context.output = None

    def consume(self, event: dict[str, Any]) -> dict[str, Any] | None:
        event_type = str(event.get("type") or "")
        run_id = str(event.get("run_id") or "")
        if not run_id:
            return None
        context = self.runs.setdefault(run_id, RunContext(run_id))
        if event_type == "run_started":
            context.metadata.update(
                {key: event.get(key) for key in ("character", "ascension", "seed", "act")}
            )
            self._open_output(context)
            return None
        if event_type == "action_observed":
            self._start_step(context, event)
            return None
        if event_type == "step_resolved":
            return self._finish_step(context, event)
        if event_type == "run_ended":
            self._flush_unresolved(context, str(event.get("reason") or "run_ended"))
        return None

    def _open_output(self, context: RunContext) -> None:
        if context.output is not None:
            return
        character = _token(context.metadata.get("character")) or "unknown"
        ascension = context.metadata.get("ascension", "unknown")
        seed = str(context.metadata.get("seed") or "unknown").replace("/", "_").replace("\\", "_")
        path = self.output_dir / f"human-{character}-A{ascension}-seed-{seed}-{context.run_id}.jsonl"
        context.output_path = path
        context.output = path.open("a", encoding="utf-8")

    def _start_step(self, context: RunContext, event: dict[str, Any]) -> None:
        self._open_output(context)
        raw_observation = event.get("observation_before")
        state_available = isinstance(raw_observation, dict) and bool(raw_observation)
        if not state_available and context.run_id not in self.warned_missing_state:
            self.warned_missing_state.add(context.run_id)
            print(
                f"warning: run {context.run_id} has no CommunicationMod state; "
                "steps will be marked unusable_for_sft",
                flush=True,
            )
        message = _message(raw_observation, event.get("available_actions"))
        state = game_state(message)
        if str(state.get("screen_type") or "").upper() == "SHOP_SCREEN":
            key = shop_visit_key(message)
            if key is not None:
                context.visited_shop_rooms.add(key)

        full_state = normalize_state(message)
        attach_runtime_context(
            full_state,
            context.visited_shop_rooms,
            context.action_history,
            context.declined_card_reward_keys,
        )
        legal_actions = semantic_actions(
            message, context.visited_shop_rooms, context.declined_card_reward_keys
        )
        recorded = event.get("chosen_action") if isinstance(event.get("chosen_action"), dict) else {}
        action, is_legal, resolution = resolve_recorded_action(recorded, legal_actions, state)
        step_id = str(event.get("step_id") or "")
        context.pending[step_id] = {
            "schema_version": "0.4",
            "source": "communicationmod_action_recorder_bridge",
            "run_id": context.run_id,
            "run_config": copy.deepcopy(context.metadata),
            "step_id": step_id,
            "screen_type": full_state.get("screen_type"),
            "pre_state": full_state,
            "policy_observation": route_policy_observation(copy.deepcopy(full_state)),
            "legal_actions": legal_actions,
            "action": action,
            "raw_action": copy.deepcopy(recorded),
            "selected_action_id": action.get("id"),
            "chosen_action_legal": is_legal,
            "usable_for_sft": state_available and is_legal and bool(legal_actions),
            "action_resolution": resolution,
            "command": action.get("command"),
            "pre_message": message,
            "recorded_at_ms": event.get("timestamp_ms"),
            "state_match": copy.deepcopy(event.get("state_match"))
            if isinstance(event.get("state_match"), dict) else None,
        }

    def _finish_step(self, context: RunContext, event: dict[str, Any]) -> dict[str, Any] | None:
        step_id = str(event.get("step_id") or "")
        pending = context.pending.pop(step_id, None)
        if pending is None:
            return None
        post_message = _message(event.get("observation_after"))
        post_state = None
        if post_message["in_game"]:
            if str(game_state(post_message).get("screen_type") or "").upper() == "SHOP_SCREEN":
                key = shop_visit_key(post_message)
                if key is not None:
                    context.visited_shop_rooms.add(key)
            post_state = normalize_state(post_message)

        record = {key: value for key, value in pending.items() if key not in {"pre_message"}}
        record.update(
            {
                "post_state": post_state,
                "accepted": True,
                "stable_after_action": event.get("resolution") in {"stable_state", "next_decision"},
                "resolution": event.get("resolution"),
                "resolved_at_ms": event.get("timestamp_ms"),
            }
        )
        if isinstance(record.get("state_match"), dict):
            record["state_match"] = dict(record["state_match"])
            record["state_match"]["after_state_seq"] = event.get("after_state_seq")
            record["state_match"]["resolution"] = event.get("resolution")

        if post_message["in_game"] and isinstance(record.get("post_state"), dict):
            history_pending = {
                "step_id": record.get("step_id"),
                "screen_type": record.get("screen_type"),
                "pre_state": record.get("pre_state"),
                "action": record.get("action"),
            }
            remember_declined_combat_card_reward(
                history_pending, post_message, context.declined_card_reward_keys
            )
            context.action_history.append(accepted_action_history_item(history_pending, post_message))
            del context.action_history[:-ACTION_HISTORY_STORAGE_LIMIT]
            attach_runtime_context(
                record["post_state"],
                context.visited_shop_rooms,
                context.action_history,
                context.declined_card_reward_keys,
            )
        self._write(context, record)
        return record


    def _flush_unresolved(self, context: RunContext, reason: str) -> None:
        for pending in list(context.pending.values()):
            record = {key: value for key, value in pending.items() if key != "pre_message"}
            record.update({"post_state": None, "accepted": True, "resolution": reason})
            self._write(context, record)
        context.pending.clear()

    @staticmethod
    def _write(context: RunContext, value: dict[str, Any]) -> None:
        if context.output is None:
            return
        context.output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        context.output.flush()
