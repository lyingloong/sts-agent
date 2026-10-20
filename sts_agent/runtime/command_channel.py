"""Single-writer CommunicationMod transport; acceptance is not effect completion."""

from __future__ import annotations

import base64
import time
import uuid
from typing import Any

from .lifecycle import GameDisconnected


class CommandProtocolFailure(GameDisconnected):
    """A state query failed; submitting further commands is unsafe."""


class CommandChannel:
    def __init__(self) -> None:
        self.correlated = False
        self.inflight: str | None = None
        self.inflight_command: str | None = None
        self.deadline: float | None = None
        self.last_receipt: dict[str, Any] | None = None
        self.last_sent_command: str | None = None
        self.awaiting_reply = False
        self.failure_reason: str | None = None
        self.reply_matched = False
        self.accepted_command: str | None = None

    def encode(self, command: str, observation: dict[str, Any] | None) -> str:
        if self.failure_reason is not None:
            raise CommandProtocolFailure(self.failure_reason)
        if self.inflight is not None:
            raise GameDisconnected("Cannot submit a second command before its correlated reply")
        self.last_receipt = None
        self.reply_matched = False
        self.accepted_command = None
        if not self.correlated:
            self.last_sent_command = command
            self.awaiting_reply = True
            return command  # Stock CM: explicitly unverified legacy transport.
        expected = (observation or {}).get("recorder_state_id")
        if command.upper() != "STATE" and not expected:
            raise GameDisconnected("Missing recorder_state_id; cannot safely submit an action")
        self.inflight = uuid.uuid4().hex
        self.inflight_command = command
        self.last_sent_command = command
        self.awaiting_reply = True
        self.deadline = time.monotonic() + 60.0
        encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
        return f"AR1 {self.inflight} {expected or '-'} {encoded}"

    def receive(self, message: dict[str, Any]) -> bool:
        self.reply_matched = False
        self.accepted_command = None
        if message.get("recorder_snapshot_only") is True:
            return False
        if message.get("recorder_command_protocol") == 1:
            self.correlated = True
        if self.inflight is None:
            # Legacy transport has no request ID, but still knows the local
            # outstanding command. Do not mistake later unsolicited errors
            # for a query that has already completed successfully.
            if message.get("ready_for_command") or "error" in message:
                command = self.last_sent_command if self.awaiting_reply else None
                self.awaiting_reply = False
                self._check_state_failure(command, message)
                if command is not None and command.upper() != "STATE" and "error" not in message:
                    self.accepted_command = command
            return True
        receipt = message.get("recorder_command")
        if not isinstance(receipt, dict) or receipt.get("id") != self.inflight:
            return False  # Old buffered states/errors never acknowledge another command.
        status = receipt.get("status")
        execution = receipt.get("execution_status")
        if ((status in {"accepted", "query"} or receipt.get("command") is not None)
                and receipt.get("command") != self.inflight_command):
            raise GameDisconnected("Correlated receipt has a mismatched command; execution unknown")
        if status == "rejected" or execution in {"skipped", "cancelled"}:
            message.setdefault("error", receipt.get("error") or (
                f"Command execution {execution}" if execution else "Command rejected"))
        elif status not in {"accepted", "query"} or not message.get("ready_for_command") or execution == "pending":
            return False
        self.last_receipt = dict(receipt)
        self.reply_matched = True
        command = self.inflight_command
        self.inflight = None
        self.inflight_command = None
        self.deadline = None
        self.awaiting_reply = False
        self._check_state_failure(command, message)
        if status == "accepted" and command.upper() != "STATE" and "error" not in message:
            self.accepted_command = command
        return True

    def _check_state_failure(self, command: str | None, message: dict[str, Any]) -> None:
        # Decoder failures may preserve the request ID but omit the command;
        # trust the command we submitted, not a possibly incomplete receipt.
        if command is not None and command.upper() == "STATE" and "error" in message:
            identifier = (self.last_receipt or {}).get("id") or "legacy_unverified"
            self.failure_reason = (
                f"CommunicationMod protocol failure: STATE query failed (command_id={identifier}): "
                f"{message.get('error') or 'Command rejected'}; stopped, not retried"
            )
            raise CommandProtocolFailure(self.failure_reason)

    def check_deadline(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise GameDisconnected(f"Command {self.inflight} timed out; execution unknown, not retried")

    def confirmation(self) -> dict[str, Any]:
        return {"transport": "correlated" if self.correlated else "legacy_unverified",
                "reply_matched": self.reply_matched,
                "command_id": (self.last_receipt or {}).get("id") or self.inflight,
                "command": self.last_sent_command,
                "accepted_command": self.accepted_command,
                "receipt": self.last_receipt}
