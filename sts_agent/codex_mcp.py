"""Local MCP bridge from Codex to a running StS Agent controller."""

from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import threading
import time
from typing import Any

from sts_agent.config import ConfigError, load_agent_config


def can_start_run(packet: dict[str, Any]) -> bool:
    """Require the game's explicit START capability, not a missing scene."""
    commands = packet.get("available_commands")
    return (
        packet.get("in_game") is False
        and packet.get("ready_for_command") is True
        and not packet.get("error")
        and isinstance(commands, list)
        and any(isinstance(command, str) and command.lower() == "start" for command in commands)
    )


class ControllerClient:
    """Single-client JSON-lines connection to the existing agent controller."""

    def __init__(self, host: str, port: int, connect_timeout: float) -> None:
        self.socket = socket.create_connection((host, port), timeout=connect_timeout)
        self.socket.settimeout(None)
        self.reader = self.socket.makefile("r", encoding="utf-8", newline="\n")
        self.writer = self.socket.makefile("w", encoding="utf-8", newline="\n")
        self.condition = threading.Condition()
        self.write_lock = threading.Lock()
        self.current_view: dict[str, Any] | None = None
        self.closed = False
        self.error: str | None = None
        self._thread = threading.Thread(target=self._read_loop, name="sts-controller-reader", daemon=True)
        self._thread.start()

    def _read_loop(self) -> None:
        try:
            for line in self.reader:
                try:
                    packet = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(packet, dict):
                    continue
                if packet.get("type") == "view":
                    with self.condition:
                        self.current_view = packet
                        self.condition.notify_all()
                elif packet.get("type") == "shutdown":
                    with self.condition:
                        self.closed = True
                        self.condition.notify_all()
        except (OSError, ValueError) as exc:
            with self.condition:
                self.error = str(exc)
        finally:
            with self.condition:
                self.closed = True
                self.current_view = None
                self.condition.notify_all()

    def wait_for_view(self, timeout_seconds: float, after_state_id: str | None = None) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        with self.condition:
            while True:
                if self.closed:
                    raise ConnectionError(self.error or "StS Agent controller disconnected")
                packet = self.current_view
                state_id = str(packet.get("state_id") or "") if packet else ""
                if packet is not None and (after_state_id is None or state_id != after_state_id):
                    return packet
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("timed out waiting for the next StS Agent observation")
                self.condition.wait(remaining)

    def _send_text(self, text: str) -> None:
        with self.write_lock:
            self.writer.write(json.dumps({"type": "input", "text": text, "actor": "codex"}, ensure_ascii=False) + "\n")
            self.writer.flush()

    def start_run(self, character: str, ascension: int, seed: str, timeout_seconds: float) -> dict[str, Any]:
        packet = self.wait_for_view(timeout_seconds)
        if packet.get("in_game"):
            raise ValueError("A run is already active; do not send a second START command")
        if not can_start_run(packet):
            raise ValueError("Game has not advertised a ready START command; inspect the latest observation")
        command = f"s {character} {ascension}" + (f" {seed}" if seed else "")
        with self.condition:
            if self.current_view is not packet:
                raise ValueError("State changed while validating START; fetch the latest observation")
            self.current_view = None
        self._send_text(command)
        return {
            "submitted": True, "command": command, "actor": "codex",
            "state_id": packet.get("state_id"),
            "next_step": "Call wait_for_observation with this state_id before choosing a game action.",
        }

    def execute_action(self, action_id: str, state_id: str, timeout_seconds: float) -> dict[str, Any]:
        packet = self.wait_for_view(timeout_seconds)
        current_state_id = str(packet.get("state_id") or "")
        if not current_state_id or current_state_id != state_id:
            raise ValueError("Stale state_id; fetch the latest observation before choosing an action")
        actions = packet.get("legal_actions")
        if not isinstance(actions, list):
            raise RuntimeError("controller did not provide structured legal actions; enable Codex mode")
        selected = next(
            (item for item in actions if isinstance(item, dict) and str(item.get("id") or "") == action_id),
            None,
        )
        if selected is None:
            raise ValueError(f"action {action_id!r} is not legal in state {state_id}")
        with self.condition:
            if self.current_view is not packet:
                raise ValueError("State changed while validating the action; fetch the latest observation")
            self.current_view = None
        self._send_text(action_id)
        return {"submitted": True, "state_id": state_id, "action_id": action_id, "actor": "codex"}

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.current_view = None
            self.condition.notify_all()
        # Wake the reader before closing its file wrapper (which holds a lock
        # while readline blocks). Otherwise MCP shutdown can hang indefinitely.
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        for handle in (self.reader, self.writer, self.socket):
            try:
                handle.close()
            except OSError:
                pass
        self._thread.join(timeout=2)


def observation_payload(packet: dict[str, Any]) -> dict[str, Any]:
    observation = packet.get("observation")
    actions = packet.get("legal_actions")
    if not isinstance(observation, dict) or not isinstance(actions, list):
        raise RuntimeError("structured Codex observation is missing; set framework.control_mode to codex")
    start_allowed = can_start_run(packet)
    game_status = (
        "error" if packet.get("error") else
        "ready_to_start" if start_allowed else
        "in_game" if packet.get("in_game") else "not_ready"
    )
    return {
        "state_id": packet.get("state_id"),
        "in_game": bool(packet.get("in_game")),
        "game_status": game_status,
        "can_start_run": start_allowed,
        "ready_for_command": packet.get("ready_for_command"),
        "available_commands": packet.get("available_commands", []),
        "error": packet.get("error"),
        "observation": observation,
        "legal_actions": actions,
        "start_note": (
            "can_start_run=true confirms the game accepts START. Before a run, null screen fields "
            "and empty legal_actions are normal; use start_a0_ironclad to begin an authorized run. "
            "After finishing the requested run, stop instead of starting another."
        ),
        "automatic_actions_note": (
            "Controller-owned automatic actions have already been executed before this observation. "
            "Use room_history for context and do not repeat already executed actions."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Expose a running StS Agent controller to Codex over local MCP")
    parser.add_argument("--config", default="config/agent.json")
    args = parser.parse_args()
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ImportError:
        print(
            "Missing optional dependency. Install it with: uv pip install -r requirements/codex.txt",
            file=sys.stderr,
            flush=True,
        )
        return 2

    try:
        config = load_agent_config(config_file=args.config)
        if config["framework"]["control_mode"] != "codex":
            raise ConfigError("select framework.control_mode=codex before starting the MCP bridge")
        client = ControllerClient(
            config["server"]["host"],
            config["server"]["port"],
            connect_timeout=10.0,
        )
    except (ConfigError, OSError) as exc:
        print(f"StS Agent MCP startup failed: {exc}", file=sys.stderr, flush=True)
        return 2

    server = FastMCP("sts-agent")
    wait_default = float(config["codex"]["transition_timeout_seconds"])
    action_limit = int(config["codex"]["max_actions_per_run"])
    run_actions = 0

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
    def get_observation(timeout_seconds: float = wait_default) -> dict[str, Any]:
        """Get game state and legal actions. can_start_run=true permits starting despite empty menu actions."""
        packet = client.wait_for_view(max(1.0, min(timeout_seconds, 600.0)))
        return observation_payload(packet)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False))
    def wait_for_observation(after_state_id: str, timeout_seconds: float = wait_default) -> dict[str, Any]:
        """Wait until the game has advanced beyond a previously observed state."""
        packet = client.wait_for_view(max(1.0, min(timeout_seconds, 600.0)), after_state_id=after_state_id)
        return observation_payload(packet)

    @server.tool()
    def start_a0_ironclad(seed: str = "") -> dict[str, Any]:
        """Start Ironclad A0 when can_start_run=true, even with null screen fields and no legal_actions."""
        nonlocal run_actions
        selected_seed = seed.strip() or str(config["codex"]["seed"] or "")
        if selected_seed and not re.fullmatch(r"[A-Za-z0-9]+", selected_seed):
            raise ValueError("Seed must contain only letters and digits")
        result = client.start_run(
            str(config["codex"]["character"]),
            int(config["codex"]["ascension"]),
            selected_seed,
            wait_default,
        )
        run_actions = 0
        return result

    @server.tool()
    def execute_action(action_id: str, state_id: str) -> dict[str, Any]:
        """Execute exactly one action from the supplied current legal-action list."""
        nonlocal run_actions
        if run_actions >= action_limit:
            raise RuntimeError(f"Per-run Codex action limit reached ({action_limit}); stop and inspect the run")
        result = client.execute_action(action_id, state_id, wait_default)
        run_actions += 1
        result["actions_submitted_this_run"] = run_actions
        result["action_limit"] = action_limit
        result["next_step"] = "Call wait_for_observation before choosing another action."
        return result

    try:
        server.run(transport="stdio")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
