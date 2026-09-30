"""CommunicationMod state bridge for transaction-safe ActionRecorder traces.

CommunicationMod owns the authoritative, stable game-state stream.  The Java
ActionRecorder only sends semantic ``action_begin``/``action_accepted`` markers
over its side TCP channel.  This module joins the two streams using an
immutable state sequence and a precondition check.  It deliberately refuses to
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
from sts_agent.runtime.controller import game_state, semantic_actions


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
        # A few older CommunicationMod builds sent the state object directly.
        state = message
    return {
        "in_game": bool(message.get("in_game", bool(state))),
        "ready_for_command": bool(message.get("ready_for_command", True)),
        "available_commands": list(message.get("available_commands") or []),
        "game_state": copy.deepcopy(state),
    }


def _available(message: dict[str, Any]) -> dict[str, Any]:
    state = game_state(message)
    choices = state.get("choice_list")
    if not isinstance(choices, list):
        choices = state.get("choices") if isinstance(state.get("choices"), list) else []
    return {
        "commands": list(message.get("available_commands") or []),
        "choices": copy.deepcopy(choices),
    }


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
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))


class CommunicationBridgeRecorder:
    """Join CommunicationMod states and ActionRecorder transactions."""

    def __init__(
        self,
        output_dir: Path,
        *,
        max_states: int = 256,
        match_window_ms: int = 750,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_states = max(32, int(max_states))
        self.match_window_ms = max(50, int(match_window_ms))
        self.states: deque[StateSnapshot] = deque(maxlen=self.max_states)
        self.transactions: dict[str, PendingTransaction] = {}
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
                    self._finalize(transaction, None, "bridge_closed")
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

    def ingest_state(self, raw: dict[str, Any]) -> StateSnapshot:
        with self._lock:
            message = _state_message(raw)
            self.state_seq += 1
            snapshot = StateSnapshot(
                seq=self.state_seq,
                received_at_ms=int(time.time() * 1000),
                message=message,
                state_hash=_json_hash(message),
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
            if event_type == "hello":
                self._log({"type": "marker_hello", "recorder_session": raw.get("recorder_session")})
                return
            if event_type in {"action_observed", "step_resolved", "room_changed", "combat_turn_changed", "raw_input"}:
                # The Java Mod keeps these schema-0.3 events for local/offline
                # compatibility. They contain a potentially stale in-process
                # state and must never compete with the transaction protocol.
                return
            if event_type == "run_started":
                run_id = str(raw.get("run_id") or "")
                if run_id:
                    self.runs.add(run_id)
                self.enricher.consume(raw)
                self._log({"type": "run_started_marker", "run_id": run_id})
                return
            if event_type == "run_ended":
                for transaction in list(self.transactions.values()):
                    self._finalize(transaction, None, "run_ended_unmatched")
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
                for previous in list(self.transactions.values()):
                    if previous.accepted is None:
                        continue
                    if previous.before is None:
                        self._finalize(previous, None, "unmatched_next_decision")
                        continue
                    after_candidates = [
                        item for item in self.states
                        if item.seq > previous.before.seq
                        and item.state_hash != previous.before.state_hash
                        and item.received_at_ms >= int(previous.accepted.get("timestamp_ms") or 0)
                    ]
                    self._finalize(
                        previous,
                        after_candidates[0] if after_candidates else None,
                        "next_decision",
                    )
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
                self._log({"type": "marker_rejected", "reason": "unknown_transaction", "marker": raw})
                return
            if event_type == "action_accepted":
                transaction.accepted = copy.deepcopy(raw)
                action = raw.get("action") if isinstance(raw.get("action"), dict) else raw.get("chosen_action")
                # The begin marker intentionally contains only the game
                # context. Resolve the exact action at acceptance time, when
                # the semantic action ID and target fields are available.
                if transaction.before_status != "exact_legal_action" and isinstance(action, dict):
                    transaction.begin["action"] = copy.deepcopy(action)
                    transaction.before, transaction.before_status = self._find_before(transaction)
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
        candidates = list(self.states)
        if not candidates:
            return None, "unmatched_no_state"
        expected = transaction.expected_screen
        marker_timestamp = transaction.begin.get("timestamp_ms")
        begin_ms = int(marker_timestamp) if isinstance(marker_timestamp, (int, float)) else transaction.begin_received_at_ms
        # Only states received no later than the Java marker are eligible. We
        # intentionally do not use a future state as a fallback: dropping a
        # step is safer than assigning a post-transition screen as its before
        # observation. ``match_window_ms`` is retained for compatibility with
        # existing command lines but never relaxes this ordering guarantee.
        expected_is_generic = expected in {"", "NONE", "NULL"}
        temporal = [
            item for item in candidates
            if item.received_at_ms <= begin_ms
            and (expected_is_generic or item.screen_type == expected)
        ]
        if not temporal:
            return None, "unmatched_no_prior_state"
        if not temporal:
            return None, "unmatched_screen"
        temporal.sort(key=lambda item: item.seq)
        candidate = temporal[-1]
        action = transaction.begin.get("action")
        if isinstance(action, dict) and self._action_legal(candidate.message, action):
            return candidate, "exact_legal_action"
        if expected and candidate.screen_type == expected:
            return candidate, "screen_context_only"
        return None, "unmatched_precondition"

    @staticmethod
    def _action_legal(message: dict[str, Any], action: dict[str, Any]) -> bool:
        action_id = str(action.get("id") or "")
        if not action_id:
            return False
        try:
            actions = semantic_actions(message)
        except Exception:
            actions = []
        if any(str(item.get("id") or "") == action_id for item in actions):
            return True
        state = game_state(message)
        screen = str(state.get("screen_type") or "").upper()
        room_type = str(state.get("room_type") or "").upper()
        if action_id.startswith("MAP:") and screen == "MAP":
            try:
                parts = dict(part.split("=", 1) for part in action_id.split(":")[1:] if "=" in part)
                x, y = int(parts["x"]), int(parts["y"])
            except (KeyError, TypeError, ValueError):
                return False
            screen_state = state.get("screen_state")
            next_nodes = screen_state.get("next_nodes") if isinstance(screen_state, dict) else None
            if isinstance(next_nodes, list):
                return any(
                    isinstance(node, dict) and node.get("x") == x and node.get("y") == y
                    for node in next_nodes
                )
            return False
        if action_id == "OPEN:SHOP" and screen in {"SHOP_ROOM", "ROOM_EVENT"}:
            return True
        if action_id.startswith("SHOP:") and screen in {"SHOP_SCREEN", "GRID", "SHOP_ROOM"}:
            return True
        if action_id.startswith("SELECT_CARDS:") and screen in {"GRID", "HAND_SELECT"}:
            return True
        if action_id.startswith("CAMPFIRE:") and room_type in {"REST", "REST_ROOM", "CAMPFIRE"}:
            return True
        if action_id.startswith("REWARD:") and screen in {"COMBAT_REWARD", "CARD_REWARD", "BOSS_REWARD"}:
            return True
        if action_id.startswith("BOSS_RELIC:") and screen == "BOSS_REWARD":
            return True
        if action_id.startswith("POTION_") and str(state.get("room_phase") or "").upper() == "COMBAT":
            return True
        if action_id in {"END_TURN"} and str(state.get("room_phase") or "").upper() == "COMBAT":
            return True
        if action_id.startswith("OPEN:CHEST") and room_type in {"TREASURE", "TREASURE_ROOM"}:
            return True
        return False

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
        if transaction.before_status != "exact_legal_action":
            self._finalize(transaction, None, transaction.before_status)
            return
        after_candidates = [
            item for item in self.states
            if item.seq > transaction.before.seq and item.state_hash != transaction.before.state_hash
            and item.received_at_ms >= int(transaction.accepted.get("timestamp_ms") or 0)
        ]
        if not after_candidates:
            # The next accepted action will trigger a final attempt. Do not
            # invent a duplicated after-state while the game is animating.
            return
        self._finalize(transaction, after_candidates[0], "stable_state")

    def _finalize(
        self,
        transaction: PendingTransaction,
        after: StateSnapshot | None,
        resolution: str,
    ) -> None:
        self.transactions.pop(transaction.transaction_id, None)
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
            "step_id": transaction.transaction_id,
            "timestamp_ms": accepted.get("timestamp_ms"),
            "observation_before": observation_before,
            "available_actions": available_before,
            "action": action,
            "chosen_action": action,
            "state_match": {
                "match_status": transaction.before_status,
                "before_state_seq": transaction.before.seq,
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
            "after_state_seq": after.seq if after is not None else None,
            "resolution": resolution,
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

    def run(self) -> None:
        try:
            self.server = socket.create_server((self.host, self.port), backlog=4)
            self.server.settimeout(0.5)
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
                    connection.settimeout(1.0)
                    with connection.makefile("r", encoding="utf-8", newline="\n") as stream:
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
        finally:
            if self.server is not None:
                try:
                    self.server.close()
                except OSError:
                    pass

    def close(self) -> None:
        self.stop_event.set()
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
    parser.add_argument("--match-window-ms", type=int, default=750)
    args = parser.parse_args()
    try:
        config = load_agent_config(config_file=args.config)
    except ConfigError as exc:
        print(f"invalid agent configuration: {exc}", file=sys.stderr, flush=True)
        return 2
    recording = config["recording"]
    listen = recording["listen"]
    host = str(args.host or listen["host"])
    port = int(args.port or listen["port"])
    output_dir = args.output_dir or (PROJECT_ROOT / str(recording["output_dir"]))
    recorder = CommunicationBridgeRecorder(
        output_dir,
        max_states=args.max_states,
        match_window_ms=args.match_window_ms,
    )
    marker_queue: queue.Queue[dict[str, Any]] = queue.Queue()
    marker_server = MarkerServer(host, port, marker_queue)
    recorder.start()
    marker_server.start()
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
                print("WAIT", flush=True)
                continue
            if not isinstance(message, dict):
                continue
            drain_markers(marker_queue, recorder)
            recorder.ingest_state(message)
            drain_markers(marker_queue, recorder)
            if message.get("ready_for_command"):
                # Recording is passive; human input happens in the game UI.
                print("WAIT", flush=True)
    except KeyboardInterrupt:
        return 0
    finally:
        marker_server.close()
        recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
