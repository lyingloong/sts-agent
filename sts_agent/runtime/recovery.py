"""Bounded command recovery based on proof of non-execution, never error text."""

from __future__ import annotations

from enum import Enum
from typing import Any


class RecoveryPhase(str, Enum):
    READY = "ready"
    REFRESH_AUTO = "refresh_auto"
    REFRESH_HUMAN = "refresh_human"
    REFRESH_EXHAUSTED = "refresh_exhausted"
    HUMAN_REQUIRED = "human_required"
    EXHAUSTED = "exhausted"


class CommandRecovery:
    """Refresh after failures; re-decide only for verified pre-execution rejects.

    A refresh never replays a command or resets the consecutive recovery
    budget. Only an accepted gameplay action opens a new recovery budget.
    """

    RETRYABLE_CODES = frozenset({"stale_observation", "state_changed"})
    MAX_CONSECUTIVE_RECOVERIES = 3

    def __init__(self) -> None:
        self.phase = RecoveryPhase.READY
        self.consecutive_recoveries = 0
        self.reason: str | None = None

    @property
    def decision_error(self) -> str | None:
        if self.phase == RecoveryPhase.EXHAUSTED:
            return f"{self.reason or '命令被拒绝'}；连续自动恢复已达 3 次，请人工操作。"
        if self.phase == RecoveryPhase.HUMAN_REQUIRED:
            return self.reason
        return None

    @classmethod
    def proves_not_executed(cls, confirmation: dict[str, Any]) -> bool:
        receipt = confirmation.get("receipt")
        command = confirmation.get("command")
        return (
            confirmation.get("transport") == "correlated"
            and confirmation.get("reply_matched") is True
            and isinstance(receipt, dict)
            and isinstance(receipt.get("id"), str)
            and bool(receipt["id"])
            and receipt["id"] == confirmation.get("command_id")
            and isinstance(command, str)
            and bool(command)
            and command.upper() != "STATE"
            and receipt.get("command") == command
            and receipt.get("status") == "rejected"
            and receipt.get("execution_status") == "not_executed"
            and isinstance(receipt.get("error_code"), str)
            and receipt.get("error_code") in cls.RETRYABLE_CODES
        )

    def command_failed(self, confirmation: dict[str, Any], reason: str) -> None:
        self.reason = reason
        if self.consecutive_recoveries >= self.MAX_CONSECUTIVE_RECOVERIES:
            self.phase = RecoveryPhase.REFRESH_EXHAUSTED
        elif self.proves_not_executed(confirmation):
            self.consecutive_recoveries += 1
            self.phase = RecoveryPhase.REFRESH_AUTO
        else:
            self.phase = RecoveryPhase.REFRESH_HUMAN

    def request_read_only_refresh(self) -> None:
        """Human STATE resumes decisions unless an accepted action is required."""
        if self.phase in {RecoveryPhase.EXHAUSTED, RecoveryPhase.REFRESH_EXHAUSTED}:
            self.phase = RecoveryPhase.REFRESH_EXHAUSTED
        else:
            self.phase = RecoveryPhase.REFRESH_AUTO

    def observation_refreshed(self) -> None:
        if self.phase == RecoveryPhase.REFRESH_AUTO:
            self.phase = RecoveryPhase.READY
            self.reason = None
        elif self.phase == RecoveryPhase.REFRESH_HUMAN:
            self.phase = RecoveryPhase.HUMAN_REQUIRED
        elif self.phase == RecoveryPhase.REFRESH_EXHAUSTED:
            self.phase = RecoveryPhase.EXHAUSTED

    def gameplay_accepted(self) -> None:
        self.consecutive_recoveries = 0
        self.phase = RecoveryPhase.READY
        self.reason = None

    def snapshot(self) -> dict[str, Any]:
        return {"phase": self.phase.value, "consecutive_recoveries": self.consecutive_recoveries,
                "max_consecutive_recoveries": self.MAX_CONSECUTIVE_RECOVERIES, "reason": self.reason}
