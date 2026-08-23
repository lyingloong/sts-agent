"""Autoplay a fixed Slay the Spire seed through the controller TCP bridge.

This is intentionally a terminal client, not a second game controller. It
only accepts the policy server's top candidate with the existing ``a`` command,
so every action is still checked and compiled by communication_controller.py.
One loaded policy stays frozen for a complete episode. The script writes a
small manifest; the controller remains the source of full state/action traces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


PLAYER_CLASSES = {"IRONCLAD", "SILENT", "THE_SILENT", "DEFECT", "WATCHER"}
TERMINAL_SCREENS = {"GAME_OVER", "VICTORY"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_start_command(character: str, ascension: int, seed: str) -> str:
    """Build the documented controller start syntax after CLI validation."""
    return f"s {character} {ascension} {seed}"


def view_status(text: str) -> dict[str, Any]:
    """Extract stable summary fields from the controller's human view."""
    result: dict[str, Any] = {"run_id": None, "screen": None, "act": None, "floor": None}
    for line in text.splitlines():
        if not line.startswith("RUN "):
            continue
        parts = [part.strip() for part in line.split("|")]
        if parts:
            result["run_id"] = parts[0][4:].strip() or None
        for part in parts:
            if part.startswith("Act "):
                fields = part.split()
                if len(fields) >= 4 and fields[2] == "Floor":
                    result["act"] = fields[1]
                    result["floor"] = fields[3]
            elif part in TERMINAL_SCREENS:
                result["screen"] = part
            elif part.isupper() and "_" in part:
                result["screen"] = part
        break
    return result


def top_candidate(packet: dict[str, Any]) -> dict[str, Any] | None:
    decision = packet.get("decision")
    if not isinstance(decision, dict):
        return None
    candidates = decision.get("model_candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    candidate = candidates[0]
    return candidate if isinstance(candidate, dict) else None


class ManifestWriter:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.handle: TextIO | None = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("w", encoding="utf-8", buffering=1)

    def write(self, value: dict[str, Any]) -> None:
        if self.handle is not None:
            self.handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


def send_input(writer: TextIO, text: str, autoplay: bool = False) -> None:
    packet: dict[str, Any] = {"type": "input", "text": text}
    if autoplay:
        packet["autoplay"] = True
    writer.write(json.dumps(packet, ensure_ascii=False) + "\n")
    writer.flush()


def default_report_path(project_root: Path, seed: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return project_root / "data" / "rollouts" / f"single-seed-{seed}-{timestamp}.jsonl"


def stop_with_handoff(
    manifest: ManifestWriter,
    reason: str,
    episodes_completed: int,
    actions_in_episode: int,
) -> int:
    manifest.write(
        {
            "event": "solver_stopped",
            "reason": reason,
            "episodes_completed": episodes_completed,
            "actions_in_current_episode": actions_in_episode,
            "at": now_iso(),
        }
    )
    print(f"solver stopped: {reason}. The controller remains available for human_terminal.py.", file=sys.stderr)
    return 2


def run_solver(args: argparse.Namespace) -> int:
    project_root = Path(__file__).resolve().parents[1]
    report_path = None if args.no_report else Path(args.report) if args.report else default_report_path(project_root, args.seed)
    manifest = ManifestWriter(report_path)
    manifest.write(
        {
            "event": "solver_start",
            "seed": args.seed,
            "character": args.character,
            "ascension": args.ascension,
            "episodes_requested": args.episodes,
            "max_actions": args.max_actions,
            "at": now_iso(),
        }
    )

    try:
        connection = socket.create_connection((args.host, args.port), timeout=args.connect_timeout)
    except OSError as exc:
        manifest.write({"event": "solver_stopped", "reason": f"connect_failed: {exc}", "at": now_iso()})
        manifest.close()
        print(f"cannot connect to controller {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1

    started_episode = False
    awaiting_main_menu = False
    actions_in_episode = 0
    episodes_completed = 0
    terminal_seen = False
    terminal_return_attempts = 0
    last_view_digest: str | None = None
    repeated_views = 0
    current_run_id: str | None = None

    try:
        with connection:
            connection.settimeout(None)
            reader = connection.makefile("r", encoding="utf-8", newline="\n")
            writer = connection.makefile("w", encoding="utf-8", newline="\n")
            with reader, writer:
                for line in reader:
                    try:
                        packet = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(packet, dict):
                        continue
                    packet_type = packet.get("type")
                    if packet_type == "hello":
                        continue
                    if packet_type == "shutdown":
                        manifest.write({"event": "solver_stopped", "reason": "controller_shutdown", "at": now_iso()})
                        return 1
                    if packet_type != "view":
                        continue

                    text = str(packet.get("text", ""))
                    in_game = bool(packet.get("in_game"))
                    status = view_status(text)
                    if status.get("run_id"):
                        current_run_id = str(status["run_id"])

                    if not in_game:
                        if started_episode and not awaiting_main_menu:
                            return stop_with_handoff(
                                manifest,
                                "game left the run before a terminal outcome",
                                episodes_completed,
                                actions_in_episode,
                            )
                        if episodes_completed >= args.episodes:
                            manifest.write(
                                {
                                    "event": "solver_complete",
                                    "episodes_completed": episodes_completed,
                                    "at": now_iso(),
                                }
                            )
                            print(f"solver complete: {episodes_completed} episode(s)")
                            return 0
                        send_input(writer, build_start_command(args.character, args.ascension, args.seed), autoplay=True)
                        started_episode = True
                        awaiting_main_menu = False
                        terminal_seen = False
                        terminal_return_attempts = 0
                        actions_in_episode = 0
                        repeated_views = 0
                        last_view_digest = None
                        manifest.write(
                            {
                                "event": "episode_start_requested",
                                "episode": episodes_completed + 1,
                                "seed": args.seed,
                                "at": now_iso(),
                            }
                        )
                        if args.verbose:
                            print(f"episode {episodes_completed + 1}: START {args.character} A{args.ascension} {args.seed}")
                        continue

                    screen = status.get("screen")
                    if screen in TERMINAL_SCREENS:
                        if not terminal_seen:
                            episodes_completed += 1
                            terminal_seen = True
                            manifest.write(
                                {
                                    "event": "episode_terminal_seen",
                                    "episode": episodes_completed,
                                    "run_id": current_run_id,
                                    "outcome": "victory" if screen == "VICTORY" else "death",
                                    "act": status.get("act"),
                                    "floor": status.get("floor"),
                                    "actions": actions_in_episode,
                                    "at": now_iso(),
                                }
                            )
                            print(
                                f"episode {episodes_completed}: {screen} "
                                f"Act {status.get('act', '?')} Floor {status.get('floor', '?')} "
                                f"after {actions_in_episode} actions"
                            )
                        terminal_return_attempts += 1
                        if terminal_return_attempts > args.max_repeated_views:
                            return stop_with_handoff(
                                manifest,
                                f"terminal screen did not close after {terminal_return_attempts - 1} return attempts",
                                episodes_completed,
                                actions_in_episode,
                            )
                        # CommunicationMod exposes a return/cancel action on
                        # terminal screens. The controller resolves x safely.
                        send_input(writer, "x", autoplay=True)
                        awaiting_main_menu = True
                        continue

                    candidate = top_candidate(packet)
                    if candidate is None:
                        return stop_with_handoff(
                            manifest,
                            "model did not provide a legal candidate",
                            episodes_completed,
                            actions_in_episode,
                        )
                    if actions_in_episode >= args.max_actions:
                        return stop_with_handoff(
                            manifest,
                            f"max_actions={args.max_actions} reached",
                            episodes_completed,
                            actions_in_episode,
                        )

                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    repeated_views = repeated_views + 1 if digest == last_view_digest else 0
                    last_view_digest = digest
                    if repeated_views >= args.max_repeated_views:
                        return stop_with_handoff(
                            manifest,
                            f"same controller view repeated {repeated_views + 1} times",
                            episodes_completed,
                            actions_in_episode,
                        )

                    if args.action_delay > 0:
                        time.sleep(args.action_delay)
                    send_input(writer, "a", autoplay=True)
                    actions_in_episode += 1
                    if args.verbose:
                        print(
                            f"episode {episodes_completed + 1} action {actions_in_episode}: "
                            f"{candidate.get('action_id', '?')}"
                        )
    finally:
        manifest.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoplay one fixed StS seed with the controller's top model candidate")
    parser.add_argument("seed", help="StS display seed; letters and digits only")
    parser.add_argument("--character", default="IRONCLAD", type=str.upper, choices=sorted(PLAYER_CLASSES))
    parser.add_argument("--ascension", type=int, default=0, choices=range(0, 21))
    parser.add_argument("--episodes", type=int, default=1, help="repeat this exact seed in the same game process")
    parser.add_argument("--max-actions", type=int, default=1200, help="watchdog cap per episode")
    parser.add_argument("--max-repeated-views", type=int, default=3, help="stop when an unchanged view repeats this many times")
    parser.add_argument("--action-delay", type=float, default=0.1, help="seconds before each automatic action")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--report", help="optional JSONL manifest path")
    parser.add_argument("--no-report", action="store_true", help="do not write a solver manifest")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.seed.isalnum():
        parser.error("seed must contain only letters and digits")
    if args.episodes < 1:
        parser.error("episodes must be at least 1")
    if args.max_actions < 1:
        parser.error("max-actions must be at least 1")
    if args.max_repeated_views < 1:
        parser.error("max-repeated-views must be at least 1")
    if args.action_delay < 0:
        parser.error("action-delay cannot be negative")
    return args


if __name__ == "__main__":
    raise SystemExit(run_solver(parse_args()))
