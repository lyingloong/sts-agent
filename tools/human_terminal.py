"""Terminal client for communication_controller.py."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys


def clear_screen() -> None:
    """Clear the terminal before every refreshed game view."""
    os.system("cls" if os.name == "nt" else "clear")


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminal UI for the StS CommunicationMod controller")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    try:
        connection = socket.create_connection((args.host, args.port), timeout=30)
    except OSError as exc:
        print(f"无法连接 controller {args.host}:{args.port}: {exc}", file=sys.stderr)
        print("请先启动带 communication_controller.py 的游戏。", file=sys.stderr)
        return 1

    with connection:
        reader = connection.makefile("r", encoding="utf-8", newline="\n")
        writer = connection.makefile("w", encoding="utf-8", newline="\n")
        print(f"已连接到 StS controller {args.host}:{args.port}")
        for line in reader:
            try:
                packet = json.loads(line)
            except json.JSONDecodeError:
                print(line.rstrip())
                continue
            packet_type = packet.get("type")
            if packet_type == "hello":
                print(packet.get("message", "connected"))
            elif packet_type == "view":
                clear_screen()
                print("=" * 72)
                print(packet.get("text", ""))
                try:
                    text = input("action> ")
                except EOFError:
                    text = "q"
                writer.write(json.dumps({"type": "input", "text": text}, ensure_ascii=False) + "\n")
                writer.flush()
            elif packet_type == "shutdown":
                print(packet.get("message", "controller stopped"))
                break
            else:
                print(packet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
