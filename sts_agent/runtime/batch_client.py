"""Read-only batch status and opt-in automation lifetime tied to a TCP client."""

from __future__ import annotations

import copy
import select
import socket
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1


class BatchClientGuard:
    """Ordinary UI disconnects are harmless; batch supervision must fail closed."""

    def __init__(self) -> None:
        self.attached = False
        self.failure: str | None = None

    def attach(self) -> None:
        self.attached = True
        self.failure = None

    def detach(self) -> None:
        self.attached = False
        self.failure = None

    def disconnected(self) -> None:
        if self.attached:
            self.failure = "批量测评客户端已断开，自动决策已暂停；请检查批量报告后再手动恢复。"
            self.attached = False

    def error(self, connection: socket.socket | None) -> str | None:
        if self.failure or not self.attached:
            return self.failure
        if connection is None:
            self.disconnected()
            return self.failure
        try:
            readable, _, _ = select.select([connection], [], [], 0)
            if readable and connection.recv(1, socket.MSG_PEEK) == b"":
                self.disconnected()
        except (OSError, ValueError):
            self.disconnected()
        return self.failure


def client_status(api: Any, message: dict, commands: set[str], run_id: str | None,
                  guard: BatchClientGuard | None) -> dict:
    """Stable structured status; do not parse UI text or expose terminal state blobs."""
    summaries = getattr(api, "run_summaries", None)
    summary = summaries.get(run_id) if isinstance(summaries, dict) else None
    if isinstance(summary, dict):
        summary = copy.deepcopy({key: value for key, value in summary.items() if key != "terminal_state"})
    else:
        summary = None
    directory = getattr(api, "output_dir", None)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "mode": "baseline_inference",
        "experiment_dir": str(Path(directory).resolve()) if isinstance(directory, (str, Path)) else None,
        "model": getattr(api, "model", None),
        "run_id": run_id,
        "ready_for_command": message.get("ready_for_command") is True,
        "can_start_run": (message.get("in_game") is False
                          and message.get("ready_for_command") is True and "start" in commands),
        "batch_attached": bool(guard and guard.attached),
        "run_summary": summary,
    }
