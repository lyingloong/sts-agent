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


def read_posthoc_reflection(packet: dict, use_color: bool) -> dict:
    """Collect optional two-sided feedback without issuing a game command."""

    terminal = packet.get("terminal_state")
    terminal = terminal if isinstance(terminal, dict) else {}
    outcome = terminal.get("screen_type", "终局")
    act = terminal.get("act", "?")
    floor = terminal.get("floor", "?")
    print(f"\n本局结束：{outcome} | Act {act} Floor {floor}")
    action_review = packet.get("action_review")
    if isinstance(action_review, list) and action_review:
        print("\n本局动作复盘")
        for index, item in enumerate(action_review, start=1):
            if not isinstance(item, dict):
                continue
            step = item.get("step_id")
            step_text = f"#{step}" if isinstance(step, int) else f"#{index}"
            action = str(item.get("action") or "未知动作")
            source = str(item.get("source") or "其他")
            screen = str(item.get("screen_type") or "?")
            print(f"{step_text}  {screen}  [{source}] {action}")
    print("填写本局可强化与应弱化的关键操作；两项均留空即跳过。")
    try:
        reinforce = input("可强化> ").strip()
        weaken = input("应弱化> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return {"type": "posthoc_reflection", "skip": True}
    if not reinforce and not weaken:
        return {"type": "posthoc_reflection", "skip": True}
    return {"type": "posthoc_reflection", "skip": False, "reinforce": reinforce, "weaken": weaken}


def print_posthoc_attribution(packet: dict) -> None:
    """Print the same filtered attribution result shown in the Overlay."""

    annotation = packet.get("annotation")
    annotation = annotation if isinstance(annotation, dict) else {}
    applied = annotation.get("applied")
    applied = [item for item in applied if isinstance(item, dict)] if isinstance(applied, list) else []
    print("\n教师归因结果")
    print(f"状态：{annotation.get('status', 'unknown')} | 通过动作：{len(applied)}")
    for direction, title in (("reinforce", "可强化"), ("weaken", "应弱化")):
        print(title)
        items = [item for item in applied if item.get("direction") == direction]
        if not items:
            print("  无通过校验的动作")
            continue
        for item in items:
            step = item.get("step_id", "?")
            label = item.get("action_label") or item.get("selected_alias") or "未知动作"
            confidence = item.get("confidence", "?")
            evidence = item.get("evidence") or ""
            print(f"  #{step} q={confidence} {label}")
            if evidence:
                print(f"    {evidence}")


def read_prompt_evolution_review(packet: dict) -> dict:
    review = packet.get("review") if isinstance(packet.get("review"), dict) else {}
    phase = str(review.get("phase") or packet.get("phase") or "validation")
    comparison = review.get("comparison") if isinstance(review.get("comparison"), dict) else {}
    print("\n" + json.dumps(comparison, ensure_ascii=False, indent=2))
    change = comparison.get("change_summary") if isinstance(comparison.get("change_summary"), dict) else {}
    print(f"目标 skill：{change.get('target_skill', '?')}")
    print(f"原 skill：{change.get('previous_skill') or change.get('previous_instruction') or '（无）'}")
    print(f"候选 skill：{change.get('skill') or change.get('instruction') or '（无）'}")
    print(f"诊断：{change.get('diagnosis') or '未提供'}")
    print(f"证据步骤：{change.get('evidence_step_ids') or []}")
    candidate = review.get("candidate") if isinstance(review.get("candidate"), dict) else {}
    if candidate.get("analysis_input_truncated"):
        print(
            "输入告警：候选分析账本已截断，"
            f"约 {candidate.get('analysis_input_chars', '?')} / {candidate.get('analysis_input_limit_chars', '?')} 字符。"
        )
    prompt = "请审核候选 skill：accept 开始验证，reject 拒绝并重写，abandon 放弃本次。" if phase == "candidate" else "请审核验证结果：accept 固化候选，reject 拒绝并重写，abandon 放弃本次。"
    print(prompt)
    try:
        verdict = input("审核> ").strip().lower()
        reason = input("理由（拒绝时必填）> ").strip() if verdict == "reject" else ""
    except (EOFError, KeyboardInterrupt):
        print()
        return {"type": "prompt_evolution_review", "verdict": "abandon", "reason": "terminal interrupted"}
    if verdict in {"a", "accept", "y", "yes"}:
        verdict = "accept"
    elif verdict in {"r", "reject", "n", "no"}:
        verdict = "reject"
    else:
        verdict = "abandon"
    return {"type": "prompt_evolution_review", "verdict": verdict, "reason": reason}


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminal UI for the StS CommunicationMod controller")
    parser.add_argument("--config", default="config/agent.json", help="项目根目录下的配置文件路径")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-clear", action="store_true", help="不要在刷新时清空终端")
    parser.add_argument("--no-color", action="store_true", help="禁用终端颜色")
    try:
        args = parser.parse_args()
        config = load_agent_config(config_file=args.config)
    except ConfigError as exc:
        print(f"配置文件错误：{exc}", file=sys.stderr)
        return 2
    host = args.host or config["server"]["host"]
    port = args.port or config["server"]["port"]
    clear = not args.no_clear
    use_color = not args.no_color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    try:
        connection = socket.create_connection((host, port), timeout=30)
    except OSError as exc:
        print(f"无法连接 controller {host}:{port}: {exc}", file=sys.stderr)
        print("请先启动带 communication_controller.py 的游戏。", file=sys.stderr)
        return 1

    # The connect timeout should not become a read timeout while the player is
    # thinking between actions.
    connection.settimeout(None)
    with connection:
        reader = connection.makefile("r", encoding="utf-8", newline="\n")
        writer = connection.makefile("w", encoding="utf-8", newline="\n")
        print(f"已连接到 StS controller {host}:{port}", flush=True)
        for line in reader:
            try:
                packet = json.loads(line)
            except json.JSONDecodeError:
                print(line.rstrip())
                continue
            packet_type = packet.get("type")
            if packet_type == "hello":
                print(f"{DIM}{packet.get('message', 'connected')}{RESET}" if use_color else packet.get("message", "connected"), flush=True)
            elif packet_type in {"view", "teacher_update", "model_update"}:
                view_text = str(packet.get("text", ""))
                render_view(view_text, clear, use_color)
                if packet_type == "teacher_update" or packet.get("auto_execute"):
                    continue
                text = read_action(view_text, clear, use_color, bool(packet.get("in_game", False)))
                writer.write(json.dumps({"type": "input", "text": text}, ensure_ascii=False) + "\n")
                writer.flush()
            elif packet_type == "posthoc_reflection_request":
                response = read_posthoc_reflection(packet, use_color)
                writer.write(json.dumps(response, ensure_ascii=False) + "\n")
                writer.flush()
            elif packet_type == "posthoc_attribution_result":
                print_posthoc_attribution(packet)
            elif packet_type == "prompt_evolution_review_request":
                response = read_prompt_evolution_review(packet)
                writer.write(json.dumps(response, ensure_ascii=False) + "\n")
                writer.flush()
            elif packet_type == "shutdown":
                print(packet.get("message", "controller stopped"), flush=True)
                break
            else:
                print(json.dumps(packet, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
