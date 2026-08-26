"""Terminal client for communication_controller.py."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys

from agent_config import ConfigError, load_agent_config


RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
BLUE = "\x1b[34m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"

LOCAL_HELP = (
    "终端快捷操作\n"
    "  p <手牌> [敌人]          出牌；手牌从 1 开始，单敌人可省略目标\n"
    "  e                        结束回合\n"
    "  u <药水槽> [敌人]        使用药水\n"
    "  d <药水槽>               丢弃药水\n"
    "  <序号> / m <序号>        选择事件、奖励或商店选项\n"
    "  c                        确认/继续\n"
    "  x                        取消/返回/跳过\n"
    "  s / r                    刷新状态（主菜单中的 s 用于开始游戏）\n"
    "  a / accept              接受模型第一候选（Candidate Assist）\n"
    "  pick <序号>              选择模型显示的候选动作\n"
    "  1 / 2 / 3               保留给事件、奖励或商店中的游戏选项\n"
    "  h / help                 显示本帮助，不发送游戏操作\n"
    "  q                        断开终端，controller 继续运行"
)


def clear_screen(enabled: bool = True) -> None:
    """Clear without invoking a shell, which is unreliable in some terminals."""
    if not enabled:
        return
    if sys.stdout.isatty():
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()
    else:
        # Useful when stdout is redirected but the client is still interactive
        # through a host terminal wrapper.
        os.system("cls" if os.name == "nt" else "clear")


def style_line(line: str, use_color: bool) -> str:
    if not use_color:
        return line
    stripped = line.strip()
    if line == "STS AGENT" or line.startswith("RUN ") or line.startswith("状态："):
        return f"{BOLD}{CYAN}{line}{RESET}"
    if line.startswith("[NOTICE]") or line.startswith("错误："):
        return f"{BOLD}{RED}{line}{RESET}"
    if stripped in {"敌人", "手牌", "药水", "合法动作"} or stripped.startswith("地图概览"):
        return f"{BOLD}{BLUE}{line}{RESET}"
    if line.startswith("输入："):
        return f"{DIM}{line}{RESET}"
    if line.lstrip().startswith("["):
        return f"{GREEN}{line}{RESET}"
    if line.lstrip().startswith("CHOOSE") or line.lstrip().startswith("PLAY"):
        return f"{YELLOW}{line}{RESET}"
    return line


def render_view(text: str, clear: bool, use_color: bool, extra: str | None = None) -> None:
    clear_screen(clear)
    for line in text.rstrip().splitlines():
        print(style_line(line, use_color))
    if extra:
        print()
        for line in extra.splitlines():
            print(style_line(line, use_color))


def read_action(view_text: str, clear: bool, use_color: bool, in_game: bool) -> str:
    """Keep help local so it cannot be lost in a controller refresh roundtrip."""
    while True:
        try:
            prompt = f"{CYAN}action> {RESET}" if use_color else "action> "
            text = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            return "q"
        command = text.strip().lower()
        if command in {"h", "help", "?"}:
            render_view(view_text, clear, use_color, LOCAL_HELP)
            continue
        if in_game and command in {"r", "refresh"}:
            text = "s"
        return text


def main() -> int:
    try:
        config = load_agent_config()
    except ConfigError as exc:
        print(f"配置文件错误：{exc}", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description="Terminal UI for the StS CommunicationMod controller")
    parser.add_argument("--host", default=config["server"]["host"])
    parser.add_argument("--port", type=int, default=config["server"]["port"])
    parser.add_argument("--no-clear", action="store_true", help="不要在刷新时清空终端")
    parser.add_argument("--no-color", action="store_true", help="禁用终端颜色")
    args = parser.parse_args()
    clear = not args.no_clear
    use_color = not args.no_color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    try:
        connection = socket.create_connection((args.host, args.port), timeout=30)
    except OSError as exc:
        print(f"无法连接 controller {args.host}:{args.port}: {exc}", file=sys.stderr)
        print("请先启动带 communication_controller.py 的游戏。", file=sys.stderr)
        return 1

    # The connect timeout should not become a read timeout while the player is
    # thinking between actions.
    connection.settimeout(None)
    with connection:
        reader = connection.makefile("r", encoding="utf-8", newline="\n")
        writer = connection.makefile("w", encoding="utf-8", newline="\n")
        print(f"已连接到 StS controller {args.host}:{args.port}", flush=True)
        for line in reader:
            try:
                packet = json.loads(line)
            except json.JSONDecodeError:
                print(line.rstrip())
                continue
            packet_type = packet.get("type")
            if packet_type == "hello":
                print(f"{DIM}{packet.get('message', 'connected')}{RESET}" if use_color else packet.get("message", "connected"), flush=True)
            elif packet_type in {"view", "teacher_update"}:
                view_text = str(packet.get("text", ""))
                render_view(view_text, clear, use_color)
                if packet_type == "teacher_update" or packet.get("auto_execute"):
                    continue
                text = read_action(view_text, clear, use_color, bool(packet.get("in_game", False)))
                writer.write(json.dumps({"type": "input", "text": text}, ensure_ascii=False) + "\n")
                writer.flush()
            elif packet_type == "shutdown":
                print(packet.get("message", "controller stopped"), flush=True)
                break
            else:
                print(json.dumps(packet, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
