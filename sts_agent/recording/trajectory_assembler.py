"""Internal trajectory writer used by the CommunicationMod state bridge.

CommunicationMod supplies authoritative states and ActionRecorder sends
transaction markers. The bridge translates a matched transaction into the
small internal ``action_observed``/``step_resolved`` pair consumed here. This
module is intentionally not a standalone receiver or offline converter.
"""

from __future__ import annotations

import copy
import json
import sys
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
        "reward_sapphire_key_claimed": ("sapphire", "蓝宝石", "蓝钥匙"),
        "reward_emerald_key_claimed": ("emerald", "翡翠", "绿钥匙"),
        "shop_purge_opened": ("purge", "删牌", "移除"),
        "singing_bowl_chosen": ("bowl", "颂钵", "碗"),
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
    state = state if isinstance(state, dict) else {}
    screen_state = state.get("screen_state") if isinstance(state.get("screen_state"), dict) else {}
    source_id = recorded.get("source_reward_id")
    if source_id is not None and source_id != screen_state.get("source_reward_id"):
        return copy.deepcopy(recorded), False, "source_reward_identity_mismatch"
    reward_id = recorded.get("reward_id")
    if reward_id is not None:
        entries = screen_state.get("relics" if state.get("screen_type") == "BOSS_REWARD" else "rewards")
        matches = [(i, item) for i, item in enumerate(entries if isinstance(entries, list) else [])
                   if isinstance(item, dict) and item.get("reward_id") == reward_id]
        if len(matches) != 1:
            return copy.deepcopy(recorded), False, "reward_identity_missing_or_ambiguous"
        index, reward = matches[0]
        if (type(recorded.get("reward_index")) is not int or recorded["reward_index"] != index
                or type(reward.get("reward_index")) is not int or reward["reward_index"] != index):
            return copy.deepcopy(recorded), False, "reward_index_mismatch"
        if recorded.get("reward_type") and reward.get("reward_type") != recorded["reward_type"]:
            return copy.deepcopy(recorded), False, "reward_type_mismatch"
        match = next((item for item in legal_actions if item.get("id") == f"CHOOSE:index={index}"), None)
        return (copy.deepcopy(match), True, "reward_identity") if match else (
            copy.deepcopy(recorded), False, "reward_choice_not_legal")
    exact = next((item for item in legal_actions if str(item.get("id") or "") == action_id), None)
    if exact is not None:
        # A positional PLAY/CHOOSE ID is insufficient if the hand/order moved.
        card_uuid = recorded.get("card_uuid")
        if card_uuid and action_id.startswith("PLAY:"):
            combat = state.get("combat_state", {}) if isinstance(state, dict) else {}
            hand = combat.get("hand", [])
            index = int(exact.get("card_index", 0)) - 1
            if not (0 <= index < len(hand) and str(hand[index].get("uuid")) == str(card_uuid)):
                return copy.deepcopy(recorded), False, "card_identity_mismatch"
        return copy.deepcopy(exact), True, "exact_id"

    state = state if isinstance(state, dict) else {}
    screen_state = state.get("screen_state") if isinstance(state.get("screen_state"), dict) else {}
    # Parent reward choices have generic labels (e.g. 'potion'). Match the
    # actual structured reward, never a held potion's use/discard label.
    reward_type = recorded.get("reward_type")
    rewards = screen_state.get("rewards")
    if reward_type and state.get("screen_type") == "COMBAT_REWARD" and isinstance(rewards, list):
        matches = []
        for index, reward in enumerate(rewards):
            if not isinstance(reward, dict) or reward.get("reward_type") != reward_type:
                continue
            entity_type = {"POTION": "potion", "RELIC": "relic"}.get(reward_type)
            if entity_type and recorded.get(f"{entity_type}_id"):
                entity = reward.get(entity_type) or {}
                if entity.get("id") != recorded[f"{entity_type}_id"]:
                    continue
            matches.extend(item for item in legal_actions if item.get("id") == f"CHOOSE:index={index}")
        if len(matches) == 1:
            return copy.deepcopy(matches[0]), True, "reward_entry"
        return copy.deepcopy(recorded), False, "ambiguous_or_missing_reward"
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
        if recorded.get("card_uuid"):
            changed = {str(recorded["card_uuid"])}
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
        "SELECT_CARDS:CONFIRM": ("CONFIRM", "PROCEED"),
        "SELECT_CARDS:HAND_CONFIRM": ("CONFIRM", "PROCEED"),
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
            item for item in legal_actions if item.get("kind") in {"choose", "shop_buy_card", "shop_buy_potion", "shop_buy_relic"}
            if any(needle in _token(item.get("label")) for needle in needles)
        ]
        if len(matches) == 1:
            return copy.deepcopy(matches[0]), True, "entity_label"

    keywords = tuple(_token(value) for value in _choice_keywords(recorded))
    if keywords:
        matches = [
            item for item in legal_actions
            if item.get("kind") == "choose"
            if any(keyword and keyword in _token(item.get("label")) for keyword in keywords)
        ]
        if len(matches) == 1:
            return copy.deepcopy(matches[0]), True, "choice_label"

    unresolved = copy.deepcopy(recorded)
    unresolved.setdefault("label", recorded.get("kind") or action_id)
    return unresolved, False, "unresolved"


@dataclass
class RunContext:
    run_id: str
    segment_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    visited_shop_rooms: set[tuple[int, int]] = field(default_factory=set)
    declined_card_reward_keys: set[tuple[int, int]] = field(default_factory=set)
    action_history: list[dict[str, Any]] = field(default_factory=list)
    pending: dict[str, dict[str, Any]] = field(default_factory=dict)
    selection_steps: list[dict[str, Any]] = field(default_factory=list)
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
            self._flush_selection(context, cancelled=True)
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
        if event_type in {"run_started", "run_resumed"}:
            if event_type == "run_resumed":
                self._flush_unresolved(context, "save_reload")
                self._flush_selection(context, cancelled=True)
            context.metadata.update(
                {key: event.get(key) for key in ("character", "ascension", "seed", "act")}
            )
            self._open_output(context)
            context.segment_id = event.get("segment_id")
            if event_type == "run_resumed":
                context.action_history.clear()
                context.visited_shop_rooms.clear()
                context.declined_card_reward_keys.clear()
                self._write(context, {**copy.deepcopy(event), "run_config": copy.deepcopy(context.metadata)})
            return None
        if event_type == "action_observed":
            self._start_step(context, event)
            return None
        if event_type == "step_resolved":
            return self._finish_step(context, event)
        if event_type in {"run_finished", "run_ended"}:
            if event_type == "run_ended":
                self._flush_unresolved(context, str(event.get("reason") or "run_ended"))
            self._flush_selection(context, cancelled=True)
            self._open_output(context)
            self._write(context, {"type": event_type, "schema_version": "0.5", "run_id": run_id,
                                  "run_config": copy.deepcopy(context.metadata),
                                  "reason": event.get("reason"), "terminal": event.get("terminal"),
                                  "context": copy.deepcopy(event.get("context")),
                                  "timestamp_ms": event.get("timestamp_ms")})
        return None

    def _open_output(self, context: RunContext) -> None:
        if context.output is not None:
            return
        character = _token(context.metadata.get("character")) or "unknown"
        ascension = context.metadata.get("ascension", "unknown")
        seed = str(context.metadata.get("seed") or "unknown").replace("/", "_").replace("\\", "_")
        path = self.output_dir / f"human-{character}-A{ascension}-seed-{seed}-{context.run_id}.jsonl"
        context.output_path = path
        if path.exists():
            # Save-and-resume keeps the same run file and its recent history,
            # even when CommunicationMod restarts the Python process.
            with path.open(encoding="utf-8") as existing:
                for line in existing:
                    try:
                        previous = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if previous.get("type") == "run_resumed":
                        context.action_history.clear()
                        continue
                    if not previous.get("accepted") or not isinstance(previous.get("action"), dict):
                        continue
                    post = previous.get("raw_message_after")
                    if not isinstance(post, dict):
                        post = _message(previous.get("observation_after"))
                    context.action_history.append(accepted_action_history_item(previous, post))
                    del context.action_history[:-ACTION_HISTORY_STORAGE_LIMIT]
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
                file=sys.stderr,
            )
        message = (copy.deepcopy(event["raw_message"]) if isinstance(event.get("raw_message"), dict)
                   else _message(raw_observation, event.get("available_actions")))
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
        legal_actions = semantic_actions(message, fine_grained=True, include_recorded_choices=True)
        recorded = event.get("chosen_action") if isinstance(event.get("chosen_action"), dict) else {}
        action, is_legal, resolution = resolve_recorded_action(recorded, legal_actions, state)
        execution_message = event.get("execution_raw_message")
        execution_match = event.get("execution_state_match") or {}
        tracked = bool(event.get("execution_tracking"))
        training_message = execution_message if tracked and isinstance(execution_message, dict) else message
        training_state = normalize_state(training_message)
        attach_runtime_context(training_state, context.visited_shop_rooms, context.action_history,
                               context.declined_card_reward_keys)
        training_legal = semantic_actions(training_message, fine_grained=True, include_recorded_choices=True)
        execution_action = event.get("execution_action") or {}
        training_recorded = execution_action if tracked else recorded
        training_action, training_is_legal, training_resolution = resolve_recorded_action(
            training_recorded, training_legal, game_state(training_message))
        exact_pre = (isinstance(event.get("state_match"), dict)
                     and event["state_match"].get("match_status") == "exact_state_id")
        usable = state_available and is_legal and bool(legal_actions) and exact_pre
        exclusion = None if usable else (resolution if not is_legal else "missing_or_invalid_submission_state")
        if tracked:
            execution_state = game_state(execution_message) if isinstance(execution_message, dict) else {}
            same_identity = (execution_action.get("kind") == recorded.get("kind")
                             and (recorded.get("kind") != "play_card"
                                  or execution_action.get("card_uuid") == recorded.get("card_uuid")))
            same_context = all(state.get(key) == execution_state.get(key) for key in ("act", "floor"))
            usable = bool(exact_pre and event.get("execution_status") == "executed"
                          and execution_match.get("match_status") == "exact_state_id"
                          and execution_match.get("logical_boundary") is True
                          and same_identity and same_context and training_is_legal)
            if not usable:
                exclusion = ("execution_not_performed" if event.get("execution_status") in {"cancelled", "skipped"}
                             else "missing_or_invalid_execution_state")
        elif state.get("screen_type") == "NONE" and state.get("action_phase") == "EXECUTING_ACTIONS":
            usable = False
            exclusion = "combat_submission_during_execution_without_execution_state"
        step_id = str(event.get("step_id") or "")
        context.pending[step_id] = {
            "schema_version": "0.5",
            "source": "communicationmod_action_recorder_bridge",
            "run_id": context.run_id,
            "segment_id": event.get("segment_id") or context.segment_id,
            "run_config": copy.deepcopy(context.metadata),
            "step_id": step_id,
            "screen_type": full_state.get("screen_type"),
            "pre_state": full_state,
            "observation_before": copy.deepcopy(raw_observation),
            "available_actions": copy.deepcopy(event.get("available_actions")),
            "raw_message_before": copy.deepcopy(message),
            "policy_observation": route_policy_observation(copy.deepcopy(full_state)),
            "legal_actions": legal_actions,
            "action": action,
            "raw_action": copy.deepcopy(recorded),
            "selected_action_id": action.get("id"),
            "chosen_action_legal": is_legal,
            "usable_for_sft": usable,
            "training_exclusion_reason": exclusion,
            "execution_tracking": tracked,
            "execution_status": event.get("execution_status") if tracked else "not_tracked",
            "execution_observation_before": copy.deepcopy(game_state(execution_message))
            if isinstance(execution_message, dict) else None,
            "execution_raw_message_before": copy.deepcopy(execution_message),
            "execution_state_match": copy.deepcopy(event.get("execution_state_match")),
            "execution_action": copy.deepcopy(event.get("execution_action")),
            "training_observation_source": "execution" if tracked else "submission",
            "training_pre_state": training_state if (not tracked or isinstance(execution_message, dict)) else None,
            "training_policy_observation": route_policy_observation(copy.deepcopy(training_state))
            if (not tracked or isinstance(execution_message, dict)) else None,
            "training_legal_actions": training_legal if (not tracked or isinstance(execution_message, dict)) else [],
            "training_action": training_action if (not tracked or isinstance(execution_message, dict)) else None,
            "training_action_resolution": training_resolution,
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
        post_message = (copy.deepcopy(event["raw_message"]) if isinstance(event.get("raw_message"), dict)
                        else _message(event.get("observation_after")))
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
                "observation_after": copy.deepcopy(event.get("observation_after")),
                "raw_message_after": copy.deepcopy(event.get("raw_message")),
                "accepted": True,
                "stable_after_action": (event.get("observation_after") is not None
                                        and event.get("effect_complete", True)
                                        and event.get("decision_ready_after",
                                            post_message.get("ready_for_command", False)
                                            and not (game_state(post_message).get("screen_type") == "NONE"
                                                     and game_state(post_message).get("action_phase") == "EXECUTING_ACTIONS"))),
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
            history_item = accepted_action_history_item(history_pending, post_message)
            if record.get("execution_tracking"):
                history_item["execution_status"] = record.get("execution_status")
            context.action_history.append(history_item)
            del context.action_history[:-ACTION_HISTORY_STORAGE_LIMIT]
            attach_runtime_context(
                record["post_state"],
                context.visited_shop_rooms,
                context.action_history,
                context.declined_card_reward_keys,
            )
        self._write_step(context, record)
        return record

    def _write_step(self, context: RunContext, record: dict[str, Any]) -> None:
        raw = record.get("raw_action") or {}
        before = record.get("observation_before") or {}
        after = record.get("observation_after") or {}
        selection_screen = before.get("screen_type") in {"GRID", "HAND_SELECT"}
        cancelled = selection_screen and raw.get("id") in {"RETURN", "CANCEL", "LEAVE"}
        if cancelled:
            record["usable_for_sft"] = False
            record["training_exclusion_reason"] = "selection_cancelled_navigation"
            self._flush_selection(context, cancelled=True)
        if raw.get("kind") == "card_selection_changed" and not raw.get("selected_cards"):
            record["usable_for_sft"] = False
            record["training_exclusion_reason"] = "empty_or_deselected_selection"
        if context.selection_steps or (raw.get("kind") == "card_selection_changed"
                and after.get("screen_type") in {"GRID", "HAND_SELECT"}):
            context.selection_steps.append(record)
            if (raw.get("kind") == "card_selection_confirmed"
                    or after.get("screen_type") not in {"GRID", "HAND_SELECT"}):
                self._flush_selection(context, cancelled=False)
        else:
            self._write(context, record)

    def _flush_selection(self, context: RunContext, *, cancelled: bool) -> None:
        for record in context.selection_steps:
            if cancelled and (record.get("raw_action") or {}).get("kind") == "card_selection_changed":
                record["usable_for_sft"] = False
                record["training_exclusion_reason"] = "selection_not_confirmed"
            self._write(context, record)
        context.selection_steps.clear()


    def _flush_unresolved(self, context: RunContext, reason: str) -> None:
        for pending in list(context.pending.values()):
            record = {key: value for key, value in pending.items() if key != "pre_message"}
            record.update({"post_state": None, "accepted": True, "resolution": reason})
            self._write_step(context, record)
        context.pending.clear()

    @staticmethod
    def _write(context: RunContext, value: dict[str, Any]) -> None:
        if context.output is None:
            return
        context.output.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        context.output.flush()
