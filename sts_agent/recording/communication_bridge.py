"""CommunicationMod state bridge for transaction-safe ActionRecorder traces.

CommunicationMod owns the authoritative, stable game-state stream.  The Java
ActionRecorder sends semantic ``action_begin``/``action_accepted`` markers and
the same published states over its side TCP channel. This module deduplicates
the two state streams and joins actions using explicit state IDs and context
checks. It deliberately refuses to
guess when the streams cannot be matched uniquely.

The process is suitable for CommunicationMod's ``command=`` setting::

    python -m sts_agent.recording.communication_bridge --config config/agent.json

Its stdout is reserved for the CommunicationMod protocol.  Diagnostics and
match decisions are written to stderr and to a small bridge event log.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import queue
import socket
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from sts_agent.config import ConfigError, load_agent_config
from sts_agent.paths import PROJECT_ROOT
from sts_agent.recording.trajectory_assembler import ActionRecorderEnricher
from sts_agent.runtime.controller import game_state


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "agent.json"


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _screen(message: dict[str, Any]) -> str:
    state = game_state(message)
    return str(state.get("screen_type") or state.get("screen_name") or "").upper()


def _state_message(message: dict[str, Any]) -> dict[str, Any]:
    """Normalize one CommunicationMod line without changing its game state."""
    if not isinstance(message, dict):
        return {"in_game": False, "ready_for_command": False, "game_state": {}}
    state = message.get("game_state")
    if not isinstance(state, dict):
        state = message if "screen_type" in message else {}
    state = copy.deepcopy(state)
    state.pop("seed", None)
    if state.get("screen_type") != "MAP":
        state["map"] = None
    result = {
        "in_game": bool(message.get("in_game", bool(state))),
        "ready_for_command": bool(message.get("ready_for_command", True)),
        "available_commands": list(message.get("available_commands") or []),
        "game_state": state,
        "recorder_state_id": message.get("recorder_state_id"),
        "recorder_state_seq": message.get("recorder_state_seq"),
        "recorder_session": message.get("recorder_session"),
    }
    if "recorder_segment_id" in message:
        result["recorder_segment_id"] = message["recorder_segment_id"]
    return result


def _available(message: dict[str, Any]) -> dict[str, Any]:
    state = game_state(message)
    choices = state.get("choice_list")
    if not isinstance(choices, list):
        choices = state.get("choices") if isinstance(state.get("choices"), list) else []
    return {
        "commands": list(message.get("available_commands") or []),
        "choices": copy.deepcopy(choices),
    }


def _effect_complete(action: dict[str, Any], before: dict[str, Any], after: dict[str, Any]) -> bool:
    """Conservative checks for effects that can lag a UI transition."""
    kind = action.get("kind")
    if kind == "end_turn" and before.get("combat_state"):
        if after.get("screen_type") == "GAME_OVER" or after.get("room_phase") != "COMBAT":
            return True
        return ((after.get("combat_state") or {}).get("turn", -1)
                > (before.get("combat_state") or {}).get("turn", -1))
    key = {"reward_sapphire_key_claimed": "sapphire", "reward_emerald_key_claimed": "emerald",
           "reward_ruby_key_claimed": "ruby"}.get(kind)
    if kind == "campfire_action" and action.get("option") == "RECALL":
        key = "ruby"
    if key:
        return (after.get("keys") or {}).get(key) is True
    if kind == "play_card" and action.get("card_uuid"):
        if after.get("screen_type") not in {"NONE", "HAND_SELECT", "GRID", "CARD_REWARD"}:
            return True
        hand = (after.get("combat_state") or {}).get("hand", [])
        return not any(card.get("uuid") == action["card_uuid"] for card in hand)
    if kind == "reward_potion_claimed" and action.get("potion_id"):
        count = lambda state: sum(item.get("id") == action["potion_id"] for item in state.get("potions", []))
        return count(after) > count(before)
    if kind in {"potion_use_requested", "potion_discarded"}:
        slot = action.get("slot")
        old_slots, new_slots = before.get("potions") or [], after.get("potions") or []
        if isinstance(slot, int) and 0 <= slot < len(old_slots) and slot < len(new_slots):
            return old_slots[slot].get("id") != new_slots[slot].get("id")
        return False
    if kind == "card_reward_selected" and action.get("card_uuid"):
        return any(card.get("uuid") == action["card_uuid"] for card in after.get("deck", []))
    if kind == "card_selection_confirmed":
        screen = before.get("screen_state") or {}
        card = screen.get("confirmation_card") or {}
        identifier = card.get("uuid")
        if identifier and screen.get("for_purge"):
            return not any(item.get("uuid") == identifier for item in after.get("deck", []))
        if identifier and screen.get("for_upgrade"):
            old = next((item for item in before.get("deck", []) if item.get("uuid") == identifier), None)
            new = next((item for item in after.get("deck", []) if item.get("uuid") == identifier), None)
            return bool(old and new and new.get("upgrades", 0) > old.get("upgrades", 0))
    return True


def _decision_ready(message: dict[str, Any]) -> bool:
    state = game_state(message)
    if state.get("screen_type") == "GAME_OVER":
        return True
    if state.get("screen_type") == "NONE" and state.get("action_phase") == "EXECUTING_ACTIONS":
        return False
    return bool(message.get("ready_for_command"))


@dataclass(frozen=True)
class StateSnapshot:
    seq: int
    received_at_ms: int
    message: dict[str, Any]
    state_hash: str
    screen_type: str


@dataclass
class PendingTransaction:
    transaction_id: str
    begin: dict[str, Any]
    begin_received_at_ms: int
    expected_screen: str
    before: StateSnapshot | None = None
    before_status: str = "unmatched"
    accepted: dict[str, Any] | None = None
    rejected: dict[str, Any] | None = None
    execution_begin: dict[str, Any] | None = None
    execution_before: StateSnapshot | None = None
    execution_result: dict[str, Any] | None = None
    settled: dict[str, Any] | None = None
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class CommunicationBridgeRecorder:
    """Join CommunicationMod states and ActionRecorder transactions."""

    def __init__(
        self,
        output_dir: Path,
        *,
        max_states: int = 256,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_states = max(32, int(max_states))
        self.states: deque[StateSnapshot] = deque(maxlen=self.max_states)
        self.transactions: dict[str, PendingTransaction] = {}
        self.accepted_order: deque[str] = deque()
        self.completed: dict[str, tuple[PendingTransaction, StateSnapshot | None, str]] = {}
        self.closed_ids: deque[str] = deque(maxlen=4096)
        self.runs: set[str] = set()
        self.event_log: TextIO | None = None
        self.enricher = ActionRecorderEnricher(output_dir)
        self.session_id = uuid.uuid4().hex
        self.state_seq = 0
        self.marker_count = 0
        self._lock = threading.RLock()

    def start(self) -> None:
        self.event_log = (self.output_dir / f"communication-bridge-{self.session_id}.jsonl").open(
            "a", encoding="utf-8"
        )

    def close(self) -> None:
        with self._lock:
            for transaction in list(self.transactions.values()):
                if transaction.accepted is not None:
                    after = self._latest_after(transaction)
                    self._finalize(transaction, after, "latest_observation" if after else "bridge_closed")
            self.transactions.clear()
            self.enricher.close()
            if self.event_log is not None:
                self.event_log.flush()
                self.event_log.close()
                self.event_log = None

    def _log(self, value: dict[str, Any]) -> None:
        if self.event_log is None:
            return
        self.event_log.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.event_log.flush()

    def ingest_state(self, raw: dict[str, Any]) -> StateSnapshot | None:
        with self._lock:
            if raw.get("error") is not None:
                self._log({"type": "protocol_error", "error": raw["error"]})
                return None
            message = _state_message(raw)
            state_id = message.get("recorder_state_id")
            for existing in self.states:
                if state_id and existing.message.get("recorder_state_id") == state_id:
                    return existing
            self.state_seq += 1
            snapshot = StateSnapshot(
                seq=self.state_seq,
                received_at_ms=int(time.time() * 1000),
                message=message,
                state_hash=_json_hash({key: value for key, value in message.items()
                                       if not key.startswith("recorder_")}),
                screen_type=_screen(message),
            )
            self.states.append(snapshot)
            self._settle_transactions(snapshot)
            self._log({
                "type": "state_received",
                "state_seq": snapshot.seq,
                "received_at_ms": snapshot.received_at_ms,
                "screen_type": snapshot.screen_type,
                "state_hash": snapshot.state_hash,
            })
            return snapshot

    def ingest_marker(self, raw: dict[str, Any]) -> None:
        if not isinstance(raw, dict):
            return
        with self._lock:
            self.marker_count += 1
            event_type = str(raw.get("type") or "")
            if event_type == "state_published":
                if isinstance(raw.get("message"), dict):
                    self.ingest_state(raw["message"])
                return
            if event_type == "run_finished":
                self.enricher.consume(raw)
                self._log({"type": "run_finished_marker", "run_id": raw.get("run_id"), "reason": raw.get("reason")})
                return
            if event_type == "hello":
                self._log({"type": "marker_hello", "recorder_session": raw.get("recorder_session")})
                return
            if event_type == "run_checkpoint":
                # A save notification is lifecycle metadata, not an action
                # transaction. It must not close pending actions or history.
                self._log({
                    "type": "run_checkpoint_marker",
                    **{key: raw.get(key) for key in (
                        "run_id", "segment_id", "checkpoint_id",
                        "checkpoint_state_id", "timestamp_ms",
                    )},
                })
                return
            if event_type in {"action_observed", "step_resolved", "room_changed", "combat_turn_changed", "raw_input"}:
                # Legacy snapshots and auxiliary events must never compete
                # with the explicit state-ID transaction protocol.
                return
            if event_type in {"run_started", "run_resumed"}:
                run_id = str(raw.get("run_id") or "")
                if event_type == "run_resumed":
                    for transaction in list(self.transactions.values()):
                        if transaction.begin.get("run_id") != run_id:
                            continue
                        if transaction.accepted is not None:
                            self._finalize(transaction, None, "save_reload")
                        else:
                            self.transactions.pop(transaction.transaction_id, None)
                            self.closed_ids.append(transaction.transaction_id)
                if run_id:
                    self.runs.add(run_id)
                self.enricher.consume(raw)
                self._log({"type": event_type + "_marker", "run_id": run_id, "segment_id": raw.get("segment_id")})
                return
            if event_type == "run_ended":
                for transaction in list(self.transactions.values()):
                    if transaction.begin.get("run_id") == raw.get("run_id"):
                        after = self._latest_after(transaction)
                        self._finalize(transaction, after, "latest_observation" if after else "run_ended_unmatched")
                self.enricher.consume(raw)
                self._log({"type": "run_ended_marker", "run_id": raw.get("run_id")})
                return
            transaction_id = str(raw.get("transaction_id") or "")
            if event_type == "action_begin":
                if not transaction_id:
                    self._log({"type": "marker_rejected", "reason": "missing_transaction_id", "marker": raw})
                    return
                # A previous accepted marker may have arrived before its
                # pre-state line crossed the CommunicationMod pipe. Give the
                # state cache one chance to resolve it before opening the next
                # decision; if it still cannot be matched it is finalized as
                # unmatched rather than being silently attached to this one.
                next_before = next((item for item in self.states
                                    if item.message.get("recorder_state_id") == raw.get("before_state_id")
                                    and raw.get("before_state_id")), None)
                for previous in list(self.transactions.values()):
                    if previous.accepted is None:
                        continue
                    if previous.accepted.get("execution_tracking"):
                        continue  # A click boundary is not a queued execution boundary.
                    if previous.before is None or next_before is None:
                        continue
                    if (next_before.message.get("recorder_session") == previous.before.message.get("recorder_session")
                            and (next_before.message.get("recorder_state_seq") or 0)
                            > (previous.before.message.get("recorder_state_seq") or 0)):
                        self._finalize(previous, next_before, "next_decision")
                transaction = PendingTransaction(
                    transaction_id=transaction_id,
                    begin=copy.deepcopy(raw),
                    begin_received_at_ms=int(time.time() * 1000),
                    expected_screen=str(raw.get("expected_screen") or "").upper(),
                )
                transaction.before, transaction.before_status = self._find_before(transaction)
                self.transactions[transaction_id] = transaction
                self._log({
                    "type": "action_begin_matched",
                    "transaction_id": transaction_id,
                    "before_state_seq": transaction.before.seq if transaction.before else None,
                    "match_status": transaction.before_status,
                    "expected_screen": transaction.expected_screen,
                })
                return
            transaction = self.transactions.get(transaction_id)
            if transaction is None:
                if event_type in {"action_execution_begin", "action_execution_result", "action_effects_settled"} and transaction_id in self.closed_ids:
                    return  # Ready state can arrive before the optional settled marker.
                self._log({"type": "marker_rejected", "reason": "unknown_transaction", "marker": raw})
                return
            if event_type == "action_accepted":
                if transaction.accepted is not None:
                    return
                transaction.accepted = copy.deepcopy(raw)
                self.accepted_order.append(transaction_id)
                action = raw.get("action") if isinstance(raw.get("action"), dict) else raw.get("chosen_action")
                # The begin marker intentionally contains only the game
                # context. Resolve the exact action at acceptance time, when
                # the semantic action ID and target fields are available.
                if isinstance(action, dict):
                    transaction.begin["action"] = copy.deepcopy(action)
                    transaction.before, transaction.before_status = self._find_before(transaction)
                self._settle_transaction_if_possible(transaction)
            elif event_type in {"action_execution_begin", "action_execution_result", "action_effects_settled"}:
                if event_type == "action_execution_begin":
                    transaction.execution_begin = copy.deepcopy(raw)
                elif event_type == "action_execution_result":
                    transaction.execution_result = copy.deepcopy(raw)
                else:
                    transaction.settled = copy.deepcopy(raw)
                self._settle_transaction_if_possible(transaction)
            elif event_type == "action_rejected":
                transaction.rejected = copy.deepcopy(raw)
                self.transactions.pop(transaction_id, None)
                self._log({
                    "type": "action_rejected",
                    "transaction_id": transaction_id,
                    "reason": raw.get("reason"),
                })

    def _find_before(self, transaction: PendingTransaction) -> tuple[StateSnapshot | None, str]:
        # IDs are assigned in Java on the actual publication boundary. Either
        # channel can arrive first; receipt time is never a causal identifier.
        state_id = transaction.begin.get("before_state_id")
        if not state_id:
            return None, "unmatched_missing_state_id"
        candidate = next((item for item in self.states
                          if item.message.get("recorder_state_id") == state_id), None)
        if candidate is None:
            return None, "unmatched_state_id"
        if (transaction.begin.get("segment_id") is not None
                and candidate.message.get("recorder_segment_id") != transaction.begin["segment_id"]):
            return None, "unmatched_segment"
        state = game_state(candidate.message)
        context = transaction.begin.get("context") or {}
        for key in ("act", "floor"):
            if context.get(key) is not None and context.get(key) != state.get(key):
                return None, "unmatched_context"
        expected = transaction.expected_screen
        if expected not in {"", "NULL"} and expected not in {
            candidate.screen_type, str(state.get("screen_name") or "").upper(),
        }:
            return None, "unmatched_screen"
        return candidate, "exact_state_id"

    def _settle_transactions(self, newest: StateSnapshot) -> None:
        for transaction in list(self.transactions.values()):
            if transaction.accepted is not None and transaction.before is None:
                action = transaction.accepted.get("action") if isinstance(transaction.accepted.get("action"), dict) else transaction.accepted.get("chosen_action")
                if isinstance(action, dict):
                    transaction.begin["action"] = copy.deepcopy(action)
                    transaction.before, transaction.before_status = self._find_before(transaction)
            self._settle_transaction_if_possible(transaction, newest)

    def _settle_transaction_if_possible(
        self,
        transaction: PendingTransaction,
        newest: StateSnapshot | None = None,
    ) -> None:
        if transaction.accepted is None:
            return
        if transaction.before is None:
            # The next state may still be the delayed pre-state. Keep the
            # transaction pending until another marker, run end, or bridge
            # shutdown gives it a definitive unmatched outcome.
            return
        if transaction.before_status != "exact_state_id":
            self._finalize(transaction, None, transaction.before_status)
            return
        if transaction.accepted.get("execution_tracking"):
            marker = transaction.execution_begin or {}
            if transaction.execution_before is None:
                candidate = self._referenced_state(marker.get("execution_before_state_id"))
                if (candidate is not None
                        and candidate.message.get("recorder_session") == transaction.before.message.get("recorder_session")
                        and candidate.message.get("recorder_segment_id") == transaction.before.message.get("recorder_segment_id")
                        and (candidate.message.get("recorder_state_seq") or 0)
                        > (transaction.before.message.get("recorder_state_seq") or 0)
                        and all(game_state(candidate.message).get(key) == value
                                for key, value in (marker.get("context") or {}).items()
                                if key in {"act", "floor"})):
                    transaction.execution_before = candidate
            result = transaction.execution_result or {}
            if result.get("status") in {"skipped", "cancelled"}:
                self._finalize(transaction, None, "execution_not_performed")
                return
            if result.get("status") != "executed" or transaction.execution_before is None:
                return
            after = self._referenced_state((transaction.settled or {}).get("after_state_id"))
            if after is not None and (transaction.settled or {}).get("logical_boundary") is True:
                if (after.message.get("recorder_session") == transaction.execution_before.message.get("recorder_session")
                        and after.message.get("recorder_segment_id") == transaction.execution_before.message.get("recorder_segment_id")
                        and (after.message.get("recorder_state_seq") or 0)
                        > (transaction.execution_before.message.get("recorder_state_seq") or 0)):
                    self._finalize(transaction, after, "execution_settled")
                    return
        after_candidates = self._after_candidates(transaction)
        if not after_candidates:
            # No completed effect yet. A removed reward row alone is not proof
            # that the key/potion/card has already reached the player.
            return
        self._finalize(transaction, after_candidates[0], "stable_state")

    def _referenced_state(self, state_id: Any) -> StateSnapshot | None:
        if not state_id:
            return None
        return next((item for item in self.states if item.message.get("recorder_state_id") == state_id), None)

    def _after_candidates(self, transaction: PendingTransaction) -> list[StateSnapshot]:
        if transaction.before is None or transaction.accepted is None:
            return []
        action = transaction.accepted.get("action") or {}
        before = transaction.execution_before or transaction.before
        return sorted([
            item for item in self.states
            if item.message.get("recorder_session") == before.message.get("recorder_session")
            and item.message.get("recorder_segment_id") == before.message.get("recorder_segment_id")
            and (item.message.get("recorder_state_seq") or 0) > (before.message.get("recorder_state_seq") or 0)
            and item.state_hash != before.state_hash
            and item.message.get("in_game")
            and _decision_ready(item.message)
            and _effect_complete(action, game_state(before.message), game_state(item.message))
        ], key=lambda item: item.message.get("recorder_state_seq") or 0)

    def _latest_after(self, transaction: PendingTransaction) -> StateSnapshot | None:
        candidates = self._after_candidates(transaction)
        return candidates[-1] if candidates else None

    def _finalize(
        self,
        transaction: PendingTransaction,
        after: StateSnapshot | None,
        resolution: str,
    ) -> None:
        self.transactions.pop(transaction.transaction_id, None)
        self.closed_ids.append(transaction.transaction_id)
        self.completed[transaction.transaction_id] = (transaction, after, resolution)
        while self.accepted_order and self.accepted_order[0] in self.completed:
            self._emit_transaction(*self.completed.pop(self.accepted_order.popleft()))

    def _emit_transaction(
        self, transaction: PendingTransaction, after: StateSnapshot | None, resolution: str,
    ) -> None:
        accepted = transaction.accepted
        if accepted is None:
            return
        run_id = str(accepted.get("run_id") or transaction.begin.get("run_id") or "")
        action = accepted.get("action")
        if not isinstance(action, dict):
            action = accepted.get("chosen_action") if isinstance(accepted.get("chosen_action"), dict) else {}
        if not run_id or transaction.before is None:
            self._log({
                "type": "transaction_unmatched",
                "transaction_id": transaction.transaction_id,
                "match_status": transaction.before_status,
                "resolution": resolution,
            })
            print(f"warning: ActionRecorder transaction {transaction.transaction_id} has no matched pre-state: "
                  f"{transaction.before_status}; inspect the local journal", file=sys.stderr, flush=True)
            return
        if run_id not in self.runs:
            metadata = {
                "type": "run_started",
                "run_id": run_id,
                "character": game_state(transaction.before.message).get("class", "UNKNOWN"),
                "ascension": game_state(transaction.before.message).get("ascension_level", 0),
                "seed": game_state(transaction.before.message).get("seed"),
                "act": game_state(transaction.before.message).get("act"),
            }
            self.runs.add(run_id)
            self.enricher.consume(metadata)
        observation_before = transaction.before.message.get("game_state", {})
        available_before = _available(transaction.before.message)
        observation_after = after.message.get("game_state", {}) if after is not None else None
        self.enricher.consume({
            "type": "action_observed",
            "run_id": run_id,
            "segment_id": accepted.get("segment_id") or transaction.begin.get("segment_id"),
            "step_id": transaction.transaction_id,
            "timestamp_ms": accepted.get("timestamp_ms"),
            "observation_before": observation_before,
            "available_actions": available_before,
            "raw_message": transaction.before.message,
            "action": action,
            "chosen_action": action,
            "execution_tracking": bool(accepted.get("execution_tracking")),
            "execution_status": (transaction.execution_result or {}).get("status", "unknown"),
            "execution_raw_message": transaction.execution_before.message if transaction.execution_before else None,
            "execution_action": (transaction.execution_begin or {}).get("action"),
            "execution_state_match": {
                "match_status": "exact_state_id",
                "before_state_id": transaction.execution_before.message.get("recorder_state_id"),
                "before_state_seq": transaction.execution_before.message.get("recorder_state_seq"),
                "logical_boundary": (transaction.execution_begin or {}).get("logical_boundary") is True,
            } if transaction.execution_before else None,
            "state_match": {
                "match_status": transaction.before_status,
                "before_state_seq": transaction.before.seq,
                "before_state_id": transaction.before.message.get("recorder_state_id"),
                "before_state_hash": transaction.before.state_hash,
                "bridge_session": self.session_id,
            },
        })
        self.enricher.consume({
            "type": "step_resolved",
            "run_id": run_id,
            "step_id": transaction.transaction_id,
            "timestamp_ms": int(time.time() * 1000),
            "observation_after": observation_after,
            "raw_message": after.message if after else None,
            "after_state_seq": after.seq if after is not None else None,
            "resolution": resolution,
            "effect_complete": bool(after and (resolution == "execution_settled"
                                                 or _effect_complete(action, observation_before, observation_after))),
            "decision_ready_after": bool(after and (_decision_ready(after.message)
                                                       or resolution == "execution_settled")),
        })
        self._log({
            "type": "transaction_finalized",
            "transaction_id": transaction.transaction_id,
            "run_id": run_id,
            "before_state_seq": transaction.before.seq,
            "after_state_seq": after.seq if after is not None else None,
            "match_status": transaction.before_status,
            "resolution": resolution,
            "action_id": action.get("id"),
        })


class MarkerServer(threading.Thread):
    def __init__(self, host: str, port: int, marker_queue: queue.Queue[dict[str, Any]]) -> None:
        super().__init__(name="action-recorder-marker-server", daemon=True)
        self.host = host
        self.port = port
        self.marker_queue = marker_queue
        self.stop_event = threading.Event()
        self.server: socket.socket | None = None
        self.connection: socket.socket | None = None
        self.ready_event = threading.Event()
        self.startup_error: str | None = None

    def run(self) -> None:
        try:
            self.server = socket.create_server((self.host, self.port), backlog=4)
            self.server.settimeout(0.5)
            self.ready_event.set()
            print(f"ActionRecorder marker bridge listening on {self.host}:{self.port}", file=sys.stderr, flush=True)
            while not self.stop_event.is_set():
                try:
                    connection, address = self.server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                print(f"ActionRecorder marker connected: {address[0]}:{address[1]}", file=sys.stderr, flush=True)
                with connection:
                    self.connection = connection
                    # A buffered text reader becomes unusable after a socket
                    # timeout. Human thinking time must not kill this server.
                    connection.settimeout(None)
                    with connection.makefile("r", encoding="utf-8", newline="\n") as stream:
                        try:
                            for line in stream:
                                if self.stop_event.is_set():
                                    break
                                if line.strip():
                                    try:
                                        value = json.loads(line)
                                    except json.JSONDecodeError:
                                        continue
                                    if isinstance(value, dict):
                                        self.marker_queue.put(value)
                        except (OSError, UnicodeError) as exc:
                            print(f"ActionRecorder marker disconnected: {exc}", file=sys.stderr, flush=True)
                    self.connection = None
        except OSError as exc:
            self.startup_error = str(exc)
            self.ready_event.set()
        finally:
            if self.server is not None:
                try:
                    self.server.close()
                except OSError:
                    pass

    def close(self) -> None:
        self.stop_event.set()
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self.server is not None:
            try:
                self.server.close()
            except OSError:
                pass


def drain_markers(marker_queue: queue.Queue[dict[str, Any]], recorder: CommunicationBridgeRecorder) -> None:
    while True:
        try:
            marker = marker_queue.get_nowait()
        except queue.Empty:
            return
        recorder.ingest_marker(marker)


def main() -> int:
    parser = argparse.ArgumentParser(description="CommunicationMod state/action transaction bridge")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--host", default=None, help="override ActionRecorder marker host")
    parser.add_argument("--port", type=int, default=None, help="override ActionRecorder marker port")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-states", type=int, default=256)
    args = parser.parse_args()
    try:
        config = load_agent_config(config_file=args.config)
    except ConfigError as exc:
        print(f"invalid agent configuration: {exc}", file=sys.stderr, flush=True)
        return 2
    recording = config["recording"]
    listen = recording["listen"]
    host = str(args.host or listen["host"])
    port = int(args.port if args.port is not None else listen["port"])
    output_dir = args.output_dir or (PROJECT_ROOT / str(recording["output_dir"]))
    recorder = CommunicationBridgeRecorder(
        output_dir,
        max_states=args.max_states,
    )
    marker_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    marker_server = MarkerServer(host, port, marker_queue)
    recorder.start()
    marker_server.start()
    if not marker_server.ready_event.wait(timeout=5) or marker_server.startup_error:
        print(f"cannot start ActionRecorder listener: {marker_server.startup_error or 'startup timeout'}",
              file=sys.stderr, flush=True)
        marker_server.close()
        recorder.close()
        return 2
    def consume_markers() -> None:
        while not marker_server.stop_event.is_set():
            try:
                recorder.ingest_marker(marker_queue.get(timeout=0.2))
            except queue.Empty:
                continue
    marker_worker = threading.Thread(target=consume_markers, name="recorder-transactions", daemon=True)
    marker_worker.start()
    # Required CommunicationMod handshake. Nothing else may be printed to
    # stdout except protocol commands.
    print("ready", flush=True)
    try:
        for line in sys.stdin:
            raw = line.rstrip("\r\n")
            if not raw:
                continue
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                print("invalid CommunicationMod JSON; ignoring", file=sys.stderr, flush=True)
                continue
            if not isinstance(message, dict):
                continue
            recorder.ingest_state(message)
            # Purely passive: do not reply with commands. WAIT requires a
            # frame argument and creates a polling loop even when valid.
            # States arrive on source notifications and action boundaries.
    except KeyboardInterrupt:
        return 0
    finally:
        marker_server.close()
        marker_server.join(timeout=2)
        marker_worker.join(timeout=2)
        drain_markers(marker_queue, recorder)
        recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
