"""Two-window translucent desktop overlay for the StS controller.

The controller owns game actions and only accepts one TCP client at a time.
This is an alternative to the text terminal: it renders the same protocol in
separate movable, resizable Windows windows without changing trace semantics.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import queue
import re
import socket
import sys
import threading
from pathlib import Path
from typing import Any, Callable

try:
    import tkinter as tk
    from tkinter import font as tkfont
    from tkinter import messagebox
except ImportError as exc:  # pragma: no cover - Windows Python includes Tk by default.
    raise SystemExit(f"Tkinter is required for the overlay client: {exc}") from exc

from sts_agent.config import ConfigError, load_agent_config
from sts_agent.learning.prompt_evolution import skill_text_segments
from sts_agent.paths import PROJECT_ROOT
from sts_agent.policy.history import history_step_text


ACTION_HELP = """输入格式

  p <手牌> [敌人]          出牌；单敌人可省略目标
  e                        结束回合
  u <药水槽> [敌人]        使用药水
  d <药水槽>               丢弃药水
  <序号> / m <序号>        选择事件、奖励或商店选项
  c                        确认/继续
  x                        取消/返回/跳过
  s [角色] [进阶] [种子]   主菜单中开始游戏
  a / accept               接受模型第一候选
  pick <序号>              选择其他模型候选
  q                        断开 Overlay，controller 继续运行
"""

GEOMETRY_PATTERN = re.compile(r"^\d+x\d+[+-]\d+[+-]\d+$")
DIVIDER_PATTERN = re.compile(r"^[=-]{3,}$")

BACKGROUND = "#242526"
PANEL = "#303235"
FIELD = "#191a1b"
TEXT = "#edf4f7"
MUTED = "#a9b8c1"
ACCENT = "#42c8ad"
WARNING = "#f4c35c"
ERROR = "#ef7c75"
BUTTON = "#3e4246"
BUTTON_ACTIVE = "#555b60"
DISABLED = "#6d7d86"


def format_probability(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "-"


def display_model_name(value: Any) -> str:
    """Render a concise model identifier while preserving an adapter suffix."""
    text = str(value or "").strip().replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1] if text else "-"


def action_panel_mode_info(decision: Any) -> tuple[str, str]:
    """Return a mode-specific title and short purpose for MODEL / ACTION."""
    mode = decision.get("mode") if isinstance(decision, dict) else None
    return {
        "candidate_assist": ("动作面板 · 候选辅助", "模型给出候选，等待人工确认"),
        "confidence_auto": ("动作面板 · 置信度自动", "高置信度动作自动执行，其余等待确认"),
        "remote_online_rl": ("动作面板 · 在线强化学习", "策略采样结果自动执行"),
        "baseline_inference": ("动作面板 · 预训练基线", "单动作生成；复核状态后自动执行"),
        "environment_automation": ("动作面板 · 环境推进", "Controller 正在执行无决策推进动作"),
        "teacher_collection": ("动作面板 · 远程教师", "教师选择动作并记录 SFT 数据"),
        "prompt_evolution": ("动作面板 · Prompt 演化", "教师选择动作并评估策略候选"),
    }.get(str(mode), ("动作面板 · 人工操作", "等待当前合法动作"))


def enable_windows_dpi_awareness() -> None:
    """Avoid Windows bitmap scaling, which makes Tk text look blurry."""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def state_text_only(text: str, divider_columns: int = 36) -> str:
    """Fallback for controllers that do not yet send structured Overlay data."""
    result: list[str] = []
    skipping_candidates = False
    for line in text.splitlines():
        if line.startswith(("模型置信度：", "模型状态：")):
            continue
        if line.startswith("模型候选"):
            skipping_candidates = True
            continue
        if skipping_candidates and line.startswith("  ["):
            continue
        skipping_candidates = False
        if DIVIDER_PATTERN.fullmatch(line):
            result.append(line[0] * min(len(line), divider_columns))
        else:
            result.append(line)
    return "\n".join(result).strip()


def table_cell(value: Any) -> str:
    """Keep one legal action inside one unbordered table row."""
    return " ".join(str(value or "").split())


def history_line(item: dict[str, Any]) -> str:
    """Exactly the model's history line; the UI limits count, not contents."""
    return history_step_text(item)


def state_panel_text(overlay: Any, divider_columns: int = 36) -> str:
    """Render the small state summary and the three-column legal-action table."""
    if not isinstance(overlay, dict):
        return state_text_only(str(overlay or ""), divider_columns)
    summary = str(overlay.get("summary") or "等待当前状态。").strip()
    deck = overlay.get("deck")
    history = overlay.get("recent_actions")
    actions = overlay.get("legal_actions")
    lines = [summary]
    if isinstance(deck, list):
        lines.extend(("", "当前牌组"))
        deck_labels: list[str] = []
        for card in deck:
            if not isinstance(card, dict):
                continue
            name = table_cell(card.get("name") or "未知卡牌")
            try:
                upgrades = int(card.get("upgrades", 0) or 0)
            except (TypeError, ValueError):
                upgrades = 0
            try:
                count = int(card.get("count", 0) or 0)
            except (TypeError, ValueError):
                count = 0
            if upgrades > 0:
                name += f"+{upgrades}"
            deck_labels.append(f"{name}*{count}")
        lines.append("  ".join(deck_labels) if deck_labels else "暂无牌组信息")
    if isinstance(history, list):
        lines.extend(("", "当前房间历史（最近6步）"))
        history_lines = [history_line(item) for item in history if isinstance(item, dict)]
        lines.append("\n".join(history_lines) if history_lines else "暂无历史动作")
    lines.extend(("", "合法动作"))
    if not isinstance(actions, list) or not actions:
        lines.append("当前没有可执行动作。")
        return "\n".join(lines)
    lines.append("动作类型\t选择\t目标\t动作指令")
    for action in actions:
        if not isinstance(action, dict):
            continue
        lines.append(
            "\t".join(
                (
                    table_cell(action.get("type")),
                    table_cell(action.get("selection")),
                    table_cell(action.get("target")),
                    table_cell(action.get("command")),
                )
            )
        )
    return "\n".join(lines)


def candidate_button_text(candidate: dict[str, Any], index: int) -> str:
    """Keep each model suggestion in its button, without a duplicate text view."""
    label = table_cell(candidate.get("label") or candidate.get("action_id") or f"候选 {index}")
    score = candidate.get("score")
    score_text = f"  p={format_probability(score)}" if isinstance(score, (int, float)) else ""
    prefix = "接受一" if index == 1 else f"选择{index}"
    return f"{prefix}: {label}{score_text}"


def online_rl_action_text(decision: dict[str, Any]) -> str:
    """Render the sampled action, which need not be the top-probability one."""
    label = table_cell(decision.get("model_action_label") or decision.get("model_action") or "未知动作")
    probability = decision.get("model_action_probability", decision.get("model_confidence"))
    return f"本步已执行：{label}  p={format_probability(probability)}"


def online_rl_distribution_text(candidate: dict[str, Any]) -> str:
    rank = candidate.get("rank")
    prefix = f"策略 Top {rank}" if isinstance(rank, int) else "策略候选"
    label = table_cell(candidate.get("label") or candidate.get("action_id") or "未知动作")
    return f"{prefix}: {label}  p={format_probability(candidate.get('score'))}"


def online_rl_panel_text(status: Any, model_id: Any = None) -> str:
    """Render a compact, persistent status summary for autonomous RL runs."""
    value = status if isinstance(status, dict) else {}
    model = display_model_name(model_id)
    version = table_cell(value.get("policy_version") or "等待远端 worker")
    iteration = value.get("iteration")
    iteration_text = str(iteration) if isinstance(iteration, int) else "-"
    group = table_cell(value.get("group_id") or "等待首个 rollout")
    group_size = value.get("group_size")
    rollout_index = value.get("rollout_index")
    collected = value.get("episodes_collected")
    progress = "-"
    if isinstance(group_size, int) and group_size > 0:
        current = rollout_index if isinstance(rollout_index, int) else "-"
        completed = collected if isinstance(collected, int) else 0
        progress = f"当前 {current}/{group_size}  |  已完成 {completed}/{group_size}"
    lines = [
        "远程在线强化学习",
        f"模型：{model}",
        f"策略版本：{version}  |  参数更新：第 {iteration_text} 次",
        f"分组：{group}",
        f"Rollout：{progress}",
        "",
        "当前状态",
        table_cell(value.get("stage") or "等待 controller 状态"),
    ]
    gpu = value.get("gpu")
    if isinstance(gpu, dict) and isinstance(gpu.get("index"), int):
        details = [f"GPU {gpu['index']}"]
        if isinstance(gpu.get("free_memory_mib"), int):
            details.append(f"剩余 {gpu['free_memory_mib']} MiB")
        if isinstance(gpu.get("utilization_percent"), int):
            details.append(f"利用率 {gpu['utilization_percent']}%")
        lines.extend(("", "远端设备", " · ".join(details)))
    error = table_cell(value.get("error"))
    if error:
        lines.extend(("", "异常", error))
    return "\n".join(lines)


def prompt_evolution_panel_text(status: Any, model_id: Any = None) -> str:
    """Render the prompt-evolution lifecycle independently from action UI."""

    value = status if isinstance(status, dict) else {}
    model = display_model_name(model_id)
    character = table_cell(value.get("character") or "等待开始游戏")
    role = table_cell(value.get("role") or "-")
    active = table_cell(value.get("active_version") or "-")
    policy = table_cell(value.get("policy_id") or "-")
    candidate = table_cell(value.get("candidate_id") or "-")
    target = table_cell(value.get("target_skill") or "-")
    revision = value.get("revision")
    revision_text = str(revision) if isinstance(revision, int) else "-"
    lines = [
        "Prompt Evolution",
        f"教师：{model}",
        f"角色：{character}  |  当前角色：{role}",
        f"生效版本：{active}  |  运行策略：{policy}",
        f"候选：{candidate}  |  目标 skill：{target}",
        f"修订次数：{revision_text}",
        "",
        "当前阶段",
        table_cell(value.get("stage") or "等待 controller 状态"),
    ]
    input_chars = value.get("analysis_input_chars")
    input_limit = value.get("analysis_input_limit_chars")
    if isinstance(input_chars, int) and isinstance(input_limit, int) and input_chars > 0:
        if value.get("analysis_input_truncated"):
            lines.extend(
                (
                    "",
                    "输入告警",
                    f"候选分析输入实际 {input_chars:,} 字符；原始 {value.get('analysis_input_original_chars', '?')} 字符，"
                    f"上限 {input_limit:,}，尾部步骤已裁剪。",
                )
            )
        else:
            lines.extend(("", f"分析输入：{input_chars:,}/{input_limit:,} 字符，未截断"))
    error = table_cell(value.get("error"))
    if error:
        lines.extend(("", "异常", error))
    return "\n".join(lines)


def posthoc_reflection_text(packet: dict[str, Any]) -> str:
    """Summarize a completed rollout without exposing raw terminal state."""

    terminal = packet.get("terminal_state")
    terminal = terminal if isinstance(terminal, dict) else {}
    outcome = table_cell(terminal.get("screen_type") or "终局")
    act = terminal.get("act")
    floor = terminal.get("floor")
    hp = terminal.get("current_hp")
    max_hp = terminal.get("max_hp")
    details = [outcome]
    if isinstance(act, int) and isinstance(floor, int):
        details.append(f"Act {act} Floor {floor}")
    if isinstance(hp, int) and isinstance(max_hp, int):
        details.append(f"HP {hp}/{max_hp}")
    return "  |  ".join(details)


def action_review_text(value: Any) -> str:
    """Render the full, compact chronological action list for reflection."""

    if not isinstance(value, list) or not value:
        return "本局没有可复盘的已确认动作。"
    lines: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            continue
        step = item.get("step_id")
        step_text = f"#{step}" if isinstance(step, int) else f"#{index}"
        act = item.get("act")
        floor = item.get("floor")
        location = f"A{act} F{floor}" if isinstance(act, int) and isinstance(floor, int) else "A? F?"
        screen = table_cell(item.get("screen_type") or "?")
        source = table_cell(item.get("source") or "其他")
        action = table_cell(item.get("action") or "未知动作")
        lines.append(f"{step_text}  {location}  {screen}  [{source}] {action}")
    return "\n".join(lines) if lines else "本局没有可复盘的已确认动作。"


def attribution_result_text(packet: dict[str, Any]) -> str:
    """Render only validated teacher attributions, never raw model output."""

    annotation = packet.get("annotation")
    annotation = annotation if isinstance(annotation, dict) else {}
    status = table_cell(annotation.get("status") or "unknown")
    model = display_model_name(annotation.get("model_id"))
    latency = annotation.get("latency_ms")
    latency_text = f"{latency} ms" if isinstance(latency, (int, float)) else "-"
    applied = annotation.get("applied")
    applied = [item for item in applied if isinstance(item, dict)] if isinstance(applied, list) else []
    rejected = annotation.get("rejected")
    rejected = [item for item in rejected if isinstance(item, dict)] if isinstance(rejected, list) else []
    reinforce = [item for item in applied if item.get("direction") == "reinforce"]
    weaken = [item for item in applied if item.get("direction") == "weaken"]
    lines = [
        "教师归因结果",
        f"模型：{model}  |  状态：{status}  |  延迟：{latency_text}",
        f"通过：{len(applied)}（强化 {len(reinforce)} / 弱化 {len(weaken)}）  |  过滤：{len(rejected)}",
    ]
    for heading, items in (("可强化", reinforce), ("应弱化", weaken)):
        lines.extend(("", heading))
        if not items:
            lines.append("无通过校验的动作")
            continue
        for item in items:
            step = item.get("step_id")
            step_text = f"#{step}" if isinstance(step, int) else "#?"
            confidence = format_probability(item.get("confidence"))
            severity = item.get("severity")
            label = table_cell(item.get("action_label") or item.get("selected_alias") or "未知动作")
            evidence = table_cell(item.get("evidence"))
            lines.append(f"{step_text}  q={confidence}  严重度={severity}  {label}")
            if evidence:
                lines.append(f"  {evidence}")
    error = table_cell(annotation.get("error"))
    if error:
        lines.extend(("", f"错误：{error}"))
    elif rejected:
        reasons = []
        for item in rejected:
            reason = table_cell(item.get("reason"))
            if reason and reason not in reasons:
                reasons.append(reason)
        if reasons:
            lines.extend(("", "过滤原因：" + "；".join(reasons[:3])))
    return "\n".join(lines)


def prompt_evolution_review_text(review: Any) -> str:
    """Render a compact candidate comparison for the human gate."""

    if not isinstance(review, dict):
        return "没有可审核的 prompt 候选。"
    comparison = review.get("comparison") if isinstance(review.get("comparison"), dict) else {}
    phase = str(review.get("phase") or "validation")
    incumbent = comparison.get("incumbent") if isinstance(comparison.get("incumbent"), dict) else {}
    candidate = comparison.get("candidate") if isinstance(comparison.get("candidate"), dict) else {}
    left = incumbent.get("metrics") if isinstance(incumbent.get("metrics"), dict) else {}
    right = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    change = comparison.get("change_summary") if isinstance(comparison.get("change_summary"), dict) else {}
    candidate_payload = review.get("candidate") if isinstance(review.get("candidate"), dict) else {}
    if phase == "candidate":
        new_result = "候选尚未验证"
        delta = "等待人工批准后再运行同一种子"
        automatic = "不适用"
    else:
        new_result = (
            f"{right.get('screen_type', '?')}  Act {right.get('act', '?')} Floor {right.get('floor', '?')}  "
            f"HP {right.get('current_hp', '?')}/{right.get('max_hp', '?')}  分数 {right.get('score', '?')}"
        )
        delta = str(comparison.get("score_delta", "?"))
        automatic = "更好" if comparison.get("automatically_favorable") else "未超过旧版本"
    lines = [
        "PROMPT EVOLUTION REVIEW",
        f"阶段：{'候选验证前审核' if phase == 'candidate' else '验证结果审核'}",
        f"种子：{table_cell(comparison.get('seed') or '?')}",
        f"旧版本：{table_cell(incumbent.get('policy_id') or '?')}  |  新版本：{table_cell(candidate.get('policy_id') or '?')}",
        f"旧结果：{left.get('screen_type', '?')}  Act {left.get('act', '?')} Floor {left.get('floor', '?')}  HP {left.get('current_hp', '?')}/{left.get('max_hp', '?')}  分数 {left.get('score', '?')}",
        f"新结果：{new_result}",
        f"分数变化：{delta}  | 自动判断：{automatic}",
        "",
        f"修改场景：{table_cell(change.get('target_skill') or '?')}",
        f"问题诊断：{table_cell(change.get('diagnosis') or '未提供')}",
        f"预期效果：{table_cell(change.get('expected_effect') or '未提供')}",
        f"风险：{table_cell(change.get('risk') or '未提供')}",
        "",
        "原 skill（红色为候选中删除的部分）：",
        table_cell(change.get("previous_skill") or change.get("previous_instruction") or "（无）"),
        "",
        "候选 skill（绿色为相对原 skill 新增的部分）：",
        table_cell(change.get("skill") or change.get("instruction") or "（无）"),
        "",
        "证据步骤：" + ", ".join(f"#{item}" for item in change.get("evidence_step_ids", []) if isinstance(item, int)),
    ]
    if candidate_payload.get("analysis_input_truncated"):
        lines.extend(
            (
                "",
                "输入告警：候选分析账本超过配置上限，发送给教师的内容已截断。",
                f"实际 {candidate_payload.get('analysis_input_chars', '?')} 字符 / 上限 {candidate_payload.get('analysis_input_limit_chars', '?')} 字符；"
                f"原始 {candidate_payload.get('analysis_input_original_chars', '?')} 字符。",
            )
        )
    evidence = comparison.get("evidence") if isinstance(comparison.get("evidence"), dict) else {}
    candidate_evidence = evidence.get("candidate") if isinstance(evidence.get("candidate"), list) else []
    if candidate_evidence:
        lines.extend(("", "候选版本证据动作："))
        for item in candidate_evidence:
            if not isinstance(item, dict):
                continue
            lines.append(
                f"#{item.get('step_id', '?')} {table_cell(item.get('screen_type') or '?')} "
                f"{table_cell(item.get('action') or '?')}"
            )
    return "\n".join(lines)


def tag_prompt_evolution_skill_changes(widget: tk.Text, review: Any) -> None:
    """Color removed original and added candidate fragments in a review Text."""

    if not isinstance(review, dict):
        return
    comparison = review.get("comparison") if isinstance(review.get("comparison"), dict) else {}
    change = comparison.get("change_summary") if isinstance(comparison.get("change_summary"), dict) else {}
    previous = table_cell(change.get("previous_skill") or change.get("previous_instruction") or "（无）")
    candidate = table_cell(change.get("skill") or change.get("instruction") or "（无）")
    original_segments, candidate_segments = skill_text_segments(previous, candidate)

    def apply(header: str, segments: list[tuple[str, str]], tag: str) -> None:
        header_start = widget.search(header + "\n", "1.0", stopindex="end")
        if not header_start:
            return
        cursor = widget.index(f"{header_start}+{len(header) + 1}c")
        for fragment, kind in segments:
            end = widget.index(f"{cursor}+{len(fragment)}c")
            if kind != "unchanged":
                widget.tag_add(tag, cursor, end)
            cursor = end

    apply("原 skill（红色为候选中删除的部分）：", original_segments, "pe_removed")
    apply("候选 skill（绿色为相对原 skill 新增的部分）：", candidate_segments, "pe_added")


def teacher_panel_text(decision: dict[str, Any] | None) -> str:
    """Render the remote teacher's thinking/output without showing candidates."""
    if not isinstance(decision, dict) or decision.get("mode") not in {"teacher_collection", "prompt_evolution"}:
        return ""
    teacher = decision.get("teacher")
    teacher = teacher if isinstance(teacher, dict) else {}
    raw_status = str(decision.get("source") or decision.get("assist_status") or "等待教师")
    status = {"teacher_auto": "自动执行", "human_override": "等待人工接管"}.get(raw_status, raw_status)
    model = display_model_name(teacher.get("model_id") or decision.get("model_id"))
    action = str(decision.get("model_action") or "-")
    latency = teacher.get("latency_ms")
    latency_text = f"{latency} ms" if isinstance(latency, (int, float)) else "-"
    thinking_enabled = teacher.get("thinking_enabled", True)
    mode_label = "思考已开启" if thinking_enabled else "快速模式"
    title = "Prompt 演化教师" if decision.get("mode") == "prompt_evolution" else "远程教师模式"
    lines = [
        f"{title} · {mode_label}",
        f"模型：{model}    状态：{status}",
        f"动作：{action}    延迟：{latency_text}",
    ]
    if teacher.get("retrying"):
        retry_attempt = teacher.get("retry_attempt")
        retry_delay = teacher.get("retry_delay_seconds")
        attempt_text = f"第 {retry_attempt} 次" if isinstance(retry_attempt, int) else "下一次"
        delay_text = f"，等待 {retry_delay:.1f} 秒" if isinstance(retry_delay, (int, float)) else ""
        lines.append(f"网络重试：{attempt_text}{delay_text}后继续")
    thinking = str(teacher.get("thinking") or "").strip()
    output = str(teacher.get("output") or "").strip()
    lines.extend(("", "教师思考", thinking or "（未返回思考内容）", "", "最终输出", output or "（未返回输出）"))
    if teacher.get("truncated"):
        lines.extend(("", "警告：模型输出达到上限，思考或最终动作可能被截断。"))
    if decision.get("assist_reason") and not decision.get("model_candidates"):
        lines.extend(("", f"回退原因：{decision.get('assist_reason')}"))
    return "\n".join(lines)


def status_tab_stops(width: int) -> tuple[int, int, int]:
    """Return tab stops that split the status table into four equal columns."""
    usable_width = max(width - 20, 240)
    return usable_width // 4, usable_width // 2, usable_width * 3 // 4


class OverlayTerminal:
    def __init__(self, config: dict[str, Any], host: str, port: int, always_on_top: bool) -> None:
        self.config = config
        self.host = host
        self.port = port
        self.always_on_top = always_on_top
        self.layout_path = PROJECT_ROOT / "data" / "runtime" / "overlay-layout.json"
        self.saved_layout = self._load_layout()
        self.actions: queue.Queue[str] = queue.Queue()
        self.reflection_responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self.prompt_evolution_responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._prompt_evolution_warning_signature: str | None = None
        self.network_thread: threading.Thread | None = None
        self.closing = False
        self.pending_view = False
        self._save_after: str | None = None
        self._topmost_after: str | None = None
        self.reflection_dialog: tk.Toplevel | None = None
        self.attribution_dialog: tk.Toplevel | None = None

        self.root = tk.Tk()
        self.root.title("StS Agent - State")
        self.control = tk.Toplevel(self.root)
        self.control.title("StS Agent - Action Panel")
        self.online_rl = tk.Toplevel(self.root)
        self.online_rl.title("StS Agent - Online RL")
        self.prompt_evolution = tk.Toplevel(self.root)
        self.prompt_evolution.title("StS Agent - Prompt Evolution")
        self.connection_state = tk.StringVar()
        self.control_connection = tk.StringVar()
        self.ui_font = self._choose_font("Microsoft YaHei UI", "Noto Sans SC", "Microsoft YaHei", "Segoe UI")
        self.data_font = self._choose_font("Microsoft YaHei UI", "Noto Sans SC", "Microsoft YaHei", "Cascadia Mono")
        self._drag_origin: dict[str, tuple[int, int]] = {}
        self._resize_origin: dict[str, tuple[int, int, int, int, int, int, str]] = {}
        self._resize_handles: list[tk.Frame] = []
        # Borderless windows are not reliably represented in the Windows
        # taskbar. Track the ones temporarily handed back to the window
        # manager so the explicit minimize button always has a restore path.
        self._minimized_frameless: set[str] = set()
        self._minimize_in_progress: set[str] = set()
        self._configure_window(self.root, "status", 360, 260)
        self._configure_window(self.control, "control", 380, 260)
        self._configure_window(self.online_rl, "online_rl", 300, 210)
        self._configure_window(self.prompt_evolution, "prompt_evolution", 360, 250)
        for window in (self.root, self.control, self.online_rl, self.prompt_evolution):
            window.bind(
                "<Map>",
                lambda _event, target=window: self._restore_frameless_window(target),
                add="+",
            )
        self._build_status_window()
        self._build_control_window()
        self._build_online_rl_window()
        self._build_prompt_evolution_window()
        if self.config["overlay"]["frameless"]:
            self._install_resize_handles(self.root)
            self._install_resize_handles(self.control)
            self._install_resize_handles(self.online_rl)
            self._install_resize_handles(self.prompt_evolution)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.control.protocol("WM_DELETE_WINDOW", self.close)
        self.online_rl.protocol("WM_DELETE_WINDOW", self.close)
        self.prompt_evolution.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Configure>", self._schedule_layout_save, add="+")
        self.control.bind("<Configure>", self._schedule_layout_save, add="+")
        self.online_rl.bind("<Configure>", self._schedule_layout_save, add="+")
        self.prompt_evolution.bind("<Configure>", self._schedule_layout_save, add="+")
        self.root.bind("<FocusOut>", self._schedule_topmost_refresh, add="+")
        self.control.bind("<FocusOut>", self._schedule_topmost_refresh, add="+")
        self.online_rl.bind("<FocusOut>", self._schedule_topmost_refresh, add="+")
        self.prompt_evolution.bind("<FocusOut>", self._schedule_topmost_refresh, add="+")
        if not self.config["policy"]["enabled"] or self.config["policy"]["runtime"] != "remote_online_rl":
            self.online_rl.withdraw()
        if self.config["prompt_evolution"]["enabled"]:
            self._set_prompt_evolution_panel({}, self.config["teacher"]["model"])
        else:
            self.prompt_evolution.withdraw()
        self._set_controls_enabled(False)
        self._set_connection("正在连接 controller...", WARNING)
        if self.always_on_top:
            self.root.after(200, self._keep_topmost)
        self.root.after(50, self.connect)

    def _load_layout(self) -> dict[str, str]:
        try:
            value = json.loads(self.layout_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return {}
        if not isinstance(value, dict):
            return {}
        return {
            name: geometry
            for name, geometry in value.items()
            if name in {"status", "control", "online_rl", "prompt_evolution"} and isinstance(geometry, str) and GEOMETRY_PATTERN.fullmatch(geometry)
        }

    def _configure_window(self, window: tk.Misc, name: str, min_width: int, min_height: int) -> None:
        overlay = self.config["overlay"]
        window_config = overlay[name]
        geometry = self.saved_layout.get(name, window_config["geometry"])
        window.geometry(geometry)
        window.minsize(min_width, min_height)
        window.configure(background=BACKGROUND)
        if self.config["overlay"]["frameless"]:
            window.overrideredirect(True)
        window.attributes("-alpha", float(window_config["opacity"]))
        window.attributes("-topmost", self.always_on_top)

    def _raise_topmost_windows(self) -> None:
        """Use the native Windows topmost band without activating either Overlay."""
        if not self.always_on_top or self.closing:
            return
        for window in (self.root, self.control, self.online_rl, self.prompt_evolution):
            if str(window) in self._minimized_frameless:
                continue
            try:
                window.attributes("-topmost", True)
                if sys.platform == "win32":
                    ctypes.windll.user32.SetWindowPos(
                        window.winfo_id(),
                        -1,  # HWND_TOPMOST
                        0,
                        0,
                        0,
                        0,
                        0x0001 | 0x0002 | 0x0010,  # no size, move, or activation
                    )
            except (AttributeError, OSError, tk.TclError):
                continue

    def _keep_topmost(self) -> None:
        self._topmost_after = None
        self._raise_topmost_windows()
        if not self.closing and self.always_on_top:
            self._topmost_after = self.root.after(1200, self._keep_topmost)

    def _schedule_topmost_refresh(self, _event: tk.Event[tk.Misc]) -> None:
        if self.always_on_top and not self.closing:
            self.root.after_idle(self._raise_topmost_windows)

    def _choose_font(self, *preferred: str) -> str:
        available = {name.casefold(): name for name in tkfont.families(self.root)}
        for name in preferred:
            if name.casefold() in available:
                return available[name.casefold()]
        return "TkDefaultFont"

    def _title_bar(self, parent: tk.Misc, window: tk.Misc, title: str) -> tk.Frame:
        bar = tk.Frame(parent, background=PANEL, height=32, highlightbackground="#4c5053", highlightthickness=1)
        bar.grid_columnconfigure(0, weight=1)
        label = tk.Label(
            bar,
            text=title,
            foreground=ACCENT,
            background=PANEL,
            font=(self.ui_font, 10, "bold"),
            padx=10,
        )
        label.grid(row=0, column=0, sticky="w")
        if window is self.control:
            self.control_title_label = label
        if self.config["overlay"]["frameless"]:
            minimize_button = self._minimize_button(bar, window, background=PANEL)
            minimize_button.grid(row=0, column=2, sticky="e")
            close_button = tk.Button(
                bar,
                text="x",
                command=self.close,
                background=PANEL,
                foreground=MUTED,
                activebackground=ERROR,
                activeforeground=TEXT,
                relief="flat",
                borderwidth=0,
                padx=11,
                pady=5,
                font=(self.ui_font, 10, "bold"),
                cursor="hand2",
            )
            close_button.grid(row=0, column=3, sticky="e")
            self._bind_drag(bar, window)
            self._bind_drag(label, window)
        return bar

    def _minimize_button(self, parent: tk.Misc, window: tk.Misc, *, background: str) -> tk.Button:
        return tk.Button(
            parent,
            text="-",
            command=lambda: self._minimize_window(window),
            background=background,
            foreground=MUTED,
            activebackground=BUTTON_ACTIVE,
            activeforeground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=11,
            pady=5,
            font=(self.ui_font, 10, "bold"),
            cursor="hand2",
        )

    def _is_frameless_primary_window(self, window: tk.Misc) -> bool:
        return bool(
            self.config["overlay"]["frameless"]
            and window in (self.root, self.control, self.online_rl, self.prompt_evolution)
        )

    def _minimize_window(self, window: tk.Misc) -> None:
        """Minimize one panel while preserving a taskbar restore affordance."""

        if self.closing:
            return
        key = str(window)
        frameless = self._is_frameless_primary_window(window)
        try:
            if frameless:
                self._minimized_frameless.add(key)
                self._minimize_in_progress.add(key)
                window.overrideredirect(False)
            window.iconify()
        except tk.TclError:
            self._minimized_frameless.discard(key)
        finally:
            self._minimize_in_progress.discard(key)

    def _restore_frameless_window(self, window: tk.Misc) -> None:
        """Restore borderless styling after a taskbar-minimized panel returns."""

        key = str(window)
        if self.closing or key not in self._minimized_frameless or key in self._minimize_in_progress:
            return
        try:
            if window.state() == "iconic":
                return
            self._minimized_frameless.discard(key)
            window.overrideredirect(True)
            window.attributes("-alpha", float(self.config["overlay"][self._window_name(window)]["opacity"]))
            window.attributes("-topmost", self.always_on_top)
            self.root.after_idle(self._raise_topmost_windows)
        except tk.TclError:
            self._minimized_frameless.discard(key)

    def _window_name(self, window: tk.Misc) -> str:
        if window is self.root:
            return "status"
        if window is self.control:
            return "control"
        if window is self.online_rl:
            return "online_rl"
        return "prompt_evolution"

    def _show_state_panel_for_prompt(self) -> None:
        """Make a human-input prompt visible even when other panels are minimized."""

        key = str(self.root)
        if key in self._minimized_frameless:
            try:
                self.root.deiconify()
            except tk.TclError:
                return
            self._restore_frameless_window(self.root)
        try:
            self.root.lift()
        except tk.TclError:
            pass

    def _bind_drag(self, widget: tk.Misc, window: tk.Misc) -> None:
        key = str(window)

        def begin(event: tk.Event[tk.Misc]) -> None:
            self._drag_origin[key] = (event.x_root - window.winfo_x(), event.y_root - window.winfo_y())

        def drag(event: tk.Event[tk.Misc]) -> None:
            origin = self._drag_origin.get(key)
            if origin is not None:
                window.geometry(f"+{event.x_root - origin[0]}+{event.y_root - origin[1]}")

        widget.bind("<ButtonPress-1>", begin)
        widget.bind("<B1-Motion>", drag)

    def _install_resize_handles(self, window: tk.Misc) -> None:
        """Add edge/corner resize hit areas for the borderless Tk windows."""
        edge = 7
        handles = {
            "n": ("sb_v_double_arrow", {"x": 0, "y": 0, "relwidth": 1, "height": edge}),
            "s": ("sb_v_double_arrow", {"x": 0, "rely": 1, "relwidth": 1, "height": edge, "y": -edge}),
            "w": ("sb_h_double_arrow", {"x": 0, "y": 0, "width": edge, "relheight": 1}),
            "e": ("sb_h_double_arrow", {"relx": 1, "x": -edge, "y": 0, "width": edge, "relheight": 1}),
            "nw": ("size_nw_se", {"x": 0, "y": 0, "width": edge, "height": edge}),
            "se": ("size_nw_se", {"relx": 1, "rely": 1, "x": -edge, "y": -edge, "width": edge, "height": edge}),
            "ne": ("size_ne_sw", {"relx": 1, "x": -edge, "y": 0, "width": edge, "height": edge}),
            "sw": ("size_ne_sw", {"x": 0, "rely": 1, "y": -edge, "width": edge, "height": edge}),
        }
        for side, (cursor, place_options) in handles.items():
            handle = tk.Frame(window, background=BACKGROUND, cursor=cursor, bd=0, highlightthickness=0)
            handle.place(**place_options)
            handle.bind("<ButtonPress-1>", lambda event, item=side: self._begin_resize(event, window, item))
            handle.bind("<B1-Motion>", lambda event, item=side: self._resize_window(event, window, item))
            handle.bind("<ButtonRelease-1>", lambda _event, item=side: self._resize_origin.pop(f"{window}:{item}", None))
            self._resize_handles.append(handle)

    def _begin_resize(self, event: tk.Event[tk.Misc], window: tk.Misc, side: str) -> None:
        self._resize_origin[f"{window}:{side}"] = (
            event.x_root,
            event.y_root,
            window.winfo_rootx(),
            window.winfo_rooty(),
            window.winfo_width(),
            window.winfo_height(),
            side,
        )

    def _resize_window(self, event: tk.Event[tk.Misc], window: tk.Misc, side: str) -> None:
        origin = self._resize_origin.get(f"{window}:{side}")
        if origin is None:
            return
        start_x, start_y, left, top, start_width, start_height, _ = origin
        dx = event.x_root - start_x
        dy = event.y_root - start_y
        min_width, min_height = (int(value) for value in window.minsize())

        width = start_width + dx if "e" in side else start_width - dx if "w" in side else start_width
        height = start_height + dy if "s" in side else start_height - dy if "n" in side else start_height
        width = max(min_width, width)
        height = max(min_height, height)
        if "w" in side:
            left = left + start_width - width
        if "n" in side:
            top = top + start_height - height
        window.geometry(f"{width}x{height}+{left}+{top}")

    def _build_status_window(self) -> None:
        outer = tk.Frame(self.root, background=BACKGROUND, padx=10, pady=10)
        outer.grid(sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        self._title_bar(outer, self.root, "CURRENT STATE").grid(row=0, column=0, sticky="ew", pady=(0, 7))
        text_frame = tk.Frame(outer, background=FIELD)
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        self.status_text = tk.Text(
            text_frame,
            wrap="word",
            state="disabled",
            background=FIELD,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=10,
            font=(self.data_font, 10),
        )
        self.status_text.grid(row=0, column=0, sticky="nsew")
        self._configure_status_tags()
        self.status_text.bind("<Configure>", self._update_status_tabs, add="+")
        footer = tk.Frame(outer, background=BACKGROUND, height=16)
        footer.grid(row=2, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)

    def _build_control_window(self) -> None:
        outer = tk.Frame(self.control, background=BACKGROUND, padx=10, pady=10)
        outer.grid(sticky="nsew")
        self.control.grid_rowconfigure(0, weight=1)
        self.control.grid_columnconfigure(0, weight=1)
        # Let the model/teacher panel absorb vertical resize space. The
        # footer only contains the window's resize hit area and should stay
        # at its requested height.
        outer.grid_rowconfigure(2, weight=1)
        outer.grid_rowconfigure(5, weight=0, minsize=16)
        outer.grid_columnconfigure(0, weight=1)

        title_bar = self._title_bar(outer, self.control, "动作面板 · 人工操作")
        title_bar.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        self.reconnect_button = self._button(title_bar, "重连", self.connect, compact=True)
        self.reconnect_button.grid(row=0, column=1, sticky="e", padx=(0, 4))
        tk.Label(
            outer,
            textvariable=self.control_connection,
            foreground=MUTED,
            background=BACKGROUND,
            font=(self.ui_font, 9),
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))

        self.candidate_frame = tk.Frame(outer, background=BACKGROUND)
        self.candidate_frame.grid(
            row=2,
            column=0,
            sticky="nsew",
            padx=1,
            pady=1,
        )
        self.candidate_frame.grid_columnconfigure(0, weight=1)
        self.candidate_frame.configure(highlightbackground="#4c5053", highlightthickness=1)
        candidate_header = tk.Frame(self.candidate_frame, background=PANEL, height=30)
        candidate_header.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        candidate_header.grid_columnconfigure(0, weight=1)
        self.candidate_title = tk.Label(
            candidate_header,
            text="模型候选",
            foreground=ACCENT,
            background=PANEL,
            font=(self.ui_font, 10, "bold"),
            padx=9,
            pady=5,
        )
        self.candidate_title.grid(row=0, column=0, sticky="w")
        self.candidate_meta = tk.Label(
            candidate_header,
            text="等待模型反馈",
            foreground=MUTED,
            background=PANEL,
            font=(self.data_font, 9),
            padx=9,
            pady=5,
        )
        self.candidate_meta.grid(row=0, column=1, sticky="e")
        self.candidate_buttons: list[tk.Button] = []
        self.quick_buttons: list[tk.Button] = []

        self.teacher_frame = tk.Frame(
            outer,
            background=FIELD,
            highlightbackground="#4c5053",
            highlightthickness=1,
        )
        self.teacher_frame.grid(row=2, column=0, sticky="nsew")
        self.teacher_frame.grid_rowconfigure(0, weight=1)
        self.teacher_frame.grid_columnconfigure(0, weight=1)
        self.teacher_text = tk.Text(
            self.teacher_frame,
            height=10,
            wrap="word",
            state="disabled",
            background=FIELD,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=8,
            font=(self.data_font, 9),
        )
        self.teacher_text.grid(row=0, column=0, sticky="nsew")
        self._configure_teacher_tags()
        self.teacher_frame.grid_remove()

        action_frame = tk.Frame(outer, background=BACKGROUND)
        action_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        action_frame.grid_columnconfigure(0, weight=1)
        tk.Label(
            action_frame,
            text="手动动作",
            foreground=MUTED,
            background=BACKGROUND,
            font=(self.ui_font, 9, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 4))
        self.action_entry = tk.Entry(
            action_frame,
            background=FIELD,
            foreground=TEXT,
            insertbackground=TEXT,
            disabledbackground=FIELD,
            disabledforeground=DISABLED,
            relief="flat",
            font=(self.data_font, 11),
        )
        self.action_entry.grid(row=1, column=0, sticky="ew", ipady=6)
        self.action_entry.bind("<Return>", self._submit_entry)
        self.send_button = self._button(action_frame, "执行", self.submit_action)
        self.send_button.configure(
            background=ACCENT,
            foreground="#10221f",
            activebackground="#61ddc2",
            activeforeground="#10221f",
            font=(self.ui_font, 9, "bold"),
        )
        self.send_button.grid(row=1, column=1, padx=(8, 0), sticky="ew")

        quick = tk.Frame(outer, background=BACKGROUND)
        quick.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        quick.grid_columnconfigure(0, weight=1)
        tk.Label(
            quick,
            text="快捷操作",
            foreground=MUTED,
            background=BACKGROUND,
            font=(self.ui_font, 9, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))
        for index, (label, action) in enumerate((("结束回合", "e"), ("继续", "c"), ("返回", "x"))):
            quick.grid_columnconfigure(index, weight=1)
            button = self._button(quick, label, lambda value=action: self.submit_action(value), compact=True)
            button.grid(row=1, column=index, sticky="ew", padx=(0 if index == 0 else 5, 0))
            self.quick_buttons.append(button)
        quick.grid_columnconfigure(3, weight=1)
        resize_bar = tk.Frame(outer, background=BACKGROUND, height=16)
        resize_bar.grid(row=5, column=0, sticky="nsew")
        resize_bar.grid_columnconfigure(0, weight=1)

    def _build_online_rl_window(self) -> None:
        outer = tk.Frame(self.online_rl, background=BACKGROUND, padx=10, pady=10)
        outer.grid(sticky="nsew")
        self.online_rl.grid_rowconfigure(0, weight=1)
        self.online_rl.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        self._title_bar(outer, self.online_rl, "ONLINE RL").grid(row=0, column=0, sticky="ew", pady=(0, 7))
        text_frame = tk.Frame(outer, background=FIELD, highlightbackground="#4c5053", highlightthickness=1)
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        self.online_rl_text = tk.Text(
            text_frame,
            wrap="word",
            state="disabled",
            background=FIELD,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=9,
            font=(self.data_font, 9),
        )
        self.online_rl_text.grid(row=0, column=0, sticky="nsew")
        self._configure_online_rl_tags()
        footer = tk.Frame(outer, background=BACKGROUND, height=16)
        footer.grid(row=2, column=0, sticky="ew")

    def _configure_online_rl_tags(self) -> None:
        self.online_rl_text.tag_configure("rl_title", foreground=ACCENT, font=(self.ui_font, 11, "bold"), spacing3=4)
        self.online_rl_text.tag_configure("rl_meta", foreground=MUTED, font=(self.data_font, 9))
        self.online_rl_text.tag_configure("rl_progress", foreground=TEXT, font=(self.data_font, 9, "bold"))
        self.online_rl_text.tag_configure("rl_section", foreground=WARNING, font=(self.ui_font, 10, "bold"), spacing1=9, spacing3=2)
        self.online_rl_text.tag_configure("rl_stage", foreground=ACCENT, font=(self.data_font, 10, "bold"))
        self.online_rl_text.tag_configure("rl_device", foreground=MUTED, font=(self.data_font, 9))
        self.online_rl_text.tag_configure("rl_error", foreground=ERROR, font=(self.data_font, 9, "bold"))

    def _build_prompt_evolution_window(self) -> None:
        outer = tk.Frame(self.prompt_evolution, background=BACKGROUND, padx=10, pady=10)
        outer.grid(sticky="nsew")
        self.prompt_evolution.grid_rowconfigure(0, weight=1)
        self.prompt_evolution.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        self._title_bar(outer, self.prompt_evolution, "PROMPT EVOLUTION").grid(row=0, column=0, sticky="ew", pady=(0, 7))
        text_frame = tk.Frame(outer, background=FIELD, highlightbackground="#4c5053", highlightthickness=1)
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        self.prompt_evolution_text = tk.Text(
            text_frame,
            wrap="word",
            state="disabled",
            background=FIELD,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=9,
            font=(self.data_font, 9),
        )
        self.prompt_evolution_text.grid(row=0, column=0, sticky="nsew")
        self.prompt_evolution_text.tag_configure("pe_title", foreground=ACCENT, font=(self.ui_font, 11, "bold"), spacing3=4)
        self.prompt_evolution_text.tag_configure("pe_meta", foreground=MUTED, font=(self.data_font, 9))
        self.prompt_evolution_text.tag_configure("pe_section", foreground=WARNING, font=(self.ui_font, 10, "bold"), spacing1=9, spacing3=2)
        self.prompt_evolution_text.tag_configure("pe_stage", foreground=ACCENT, font=(self.data_font, 10, "bold"))
        self.prompt_evolution_text.tag_configure("pe_error", foreground=ERROR, font=(self.data_font, 9, "bold"))

    def _set_prompt_evolution_panel(self, status: Any, model_id: Any = None) -> None:
        if isinstance(model_id, str) and model_id.strip():
            self.prompt_evolution_model_id = model_id
        self.prompt_evolution.deiconify()
        widget = self.prompt_evolution_text
        value = status if isinstance(status, dict) else {}
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", prompt_evolution_panel_text(value, getattr(self, 'prompt_evolution_model_id', None)))
        widget.tag_add("pe_title", "1.0", "1.end")
        for line in ("当前阶段", "异常"):
            start = widget.search(line, "1.0", stopindex="end")
            if start:
                widget.tag_add("pe_section", start, f"{start}+{len(line)}c")
        stage_text = table_cell(value.get("stage") or "等待 controller 状态")
        stage = widget.search(stage_text, "1.0", stopindex="end")
        if stage:
            widget.tag_add("pe_stage", stage, f"{stage}+{len(stage_text)}c")
        widget.configure(state="disabled")

    def _show_prompt_evolution_update(self, packet: dict[str, Any]) -> None:
        status = packet.get("prompt_evolution")
        self._set_prompt_evolution_panel(status, packet.get("model_id"))
        if not isinstance(status, dict) or not status.get("analysis_input_truncated"):
            return
        input_chars = status.get("analysis_input_chars")
        input_limit = status.get("analysis_input_limit_chars")
        if not isinstance(input_chars, int) or not isinstance(input_limit, int):
            return
        signature = ":".join(
            str(status.get(key) or "")
            for key in ("candidate_id", "analysis_input_chars", "analysis_input_limit_chars")
        )
        if signature == self._prompt_evolution_warning_signature:
            return
        self._prompt_evolution_warning_signature = signature
        messagebox.showwarning(
            "Prompt 输入已截断",
            "候选分析输入超过配置上限，发送给 DeepSeek 的动作账本被截断。\n\n"
            f"当前约 {input_chars:,} 字符，上限 {input_limit:,} 字符。\n"
            "请提高 prompt_evolution.max_analysis_input_chars 后再进行本轮分析。",
            parent=self.prompt_evolution,
        )

    def _button(self, parent: tk.Misc, text: str, command: Callable[[], None], compact: bool = False) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            background=BUTTON,
            foreground=TEXT,
            activebackground=BUTTON_ACTIVE,
            activeforeground=TEXT,
            disabledforeground=DISABLED,
            relief="flat",
            borderwidth=0,
            padx=8 if compact else 12,
            pady=5,
            font=(self.ui_font, 9),
            cursor="hand2",
            highlightthickness=1,
            highlightbackground="#4c5053",
            highlightcolor=ACCENT,
        )

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _set_online_rl_panel(self, status: Any, model_id: Any = None) -> None:
        """Render asynchronous rollout/update progress without blocking the UI."""
        value = status if isinstance(status, dict) else {}
        if isinstance(model_id, str) and model_id.strip():
            self.online_rl_model_id = model_id
        model = display_model_name(getattr(self, "online_rl_model_id", None))
        version = table_cell(value.get("policy_version") or "等待远端 worker")
        iteration = value.get("iteration")
        iteration_text = str(iteration) if isinstance(iteration, int) else "-"
        group = table_cell(value.get("group_id") or "等待首个 rollout")
        group_size = value.get("group_size")
        rollout_index = value.get("rollout_index")
        collected = value.get("episodes_collected")
        progress = "-"
        if isinstance(group_size, int) and group_size > 0:
            current = rollout_index if isinstance(rollout_index, int) else "-"
            completed = collected if isinstance(collected, int) else 0
            progress = f"当前 {current}/{group_size}  |  已完成 {completed}/{group_size}"

        self.online_rl.deiconify()
        self.online_rl_text.configure(state="normal")
        self.online_rl_text.delete("1.0", "end")
        self.online_rl_text.insert("end", "远程在线强化学习", "rl_title")
        self.online_rl_text.insert("end", f"\n模型：{model}", "rl_meta")
        self.online_rl_text.insert("end", f"\n策略版本：{version}  |  参数更新：第 {iteration_text} 次", "rl_meta")
        self.online_rl_text.insert("end", f"\n分组：{group}", "rl_meta")
        self.online_rl_text.insert("end", f"\nRollout：{progress}", "rl_progress")
        self.online_rl_text.insert("end", "\n\n当前状态", "rl_section")
        self.online_rl_text.insert("end", f"\n{table_cell(value.get('stage') or '等待 controller 状态')}", "rl_stage")
        gpu = value.get("gpu")
        if isinstance(gpu, dict) and isinstance(gpu.get("index"), int):
            details = [f"GPU {gpu['index']}"]
            if isinstance(gpu.get("free_memory_mib"), int):
                details.append(f"剩余 {gpu['free_memory_mib']} MiB")
            if isinstance(gpu.get("utilization_percent"), int):
                details.append(f"利用率 {gpu['utilization_percent']}%")
            self.online_rl_text.insert("end", "\n\n远端设备", "rl_section")
            self.online_rl_text.insert("end", f"\n{' · '.join(details)}", "rl_device")
        error = table_cell(value.get("error"))
        if error:
            self.online_rl_text.insert("end", "\n\n异常", "rl_section")
            self.online_rl_text.insert("end", f"\n{error}", "rl_error")
        self.online_rl_text.configure(state="disabled")

    def _show_online_rl_update(self, packet: dict[str, Any]) -> None:
        self._set_online_rl_panel(packet.get("online_rl"), packet.get("model_id"))

    def _set_teacher_panel(self, decision: dict[str, Any] | None) -> None:
        if not isinstance(decision, dict) or decision.get("mode") not in {"teacher_collection", "prompt_evolution"}:
            self._set_text(self.teacher_text, teacher_panel_text(decision))
            return
        teacher = decision.get("teacher")
        teacher = teacher if isinstance(teacher, dict) else {}
        raw_status = str(decision.get("source") or decision.get("assist_status") or "等待教师")
        status = {"teacher_auto": "自动执行", "human_override": "等待人工接管"}.get(raw_status, raw_status)
        model = display_model_name(teacher.get("model_id") or decision.get("model_id"))
        action = str(decision.get("model_action") or "-")
        latency = teacher.get("latency_ms")
        latency_text = f"{latency} ms" if isinstance(latency, (int, float)) else "-"
        thinking_enabled = teacher.get("thinking_enabled", True)
        mode_label = "思考已开启" if thinking_enabled else "快速模式"
        thinking = str(teacher.get("thinking") or "").strip() or "（未返回思考内容）"
        output = str(teacher.get("output") or "").strip() or "（未返回输出）"
        retrying = bool(teacher.get("retrying"))
        retry_attempt = teacher.get("retry_attempt")
        retry_delay = teacher.get("retry_delay_seconds")
        retry_text = ""
        if retrying:
            attempt_text = f"第 {retry_attempt} 次" if isinstance(retry_attempt, int) else "下一次"
            delay_text = f"，等待 {retry_delay:.1f} 秒" if isinstance(retry_delay, (int, float)) else ""
            retry_text = f"网络重试：{attempt_text}{delay_text}后继续"
        title = "Prompt 演化教师" if decision.get("mode") == "prompt_evolution" else "远程教师模式"

        self.teacher_text.configure(state="normal")
        self.teacher_text.delete("1.0", "end")
        self.teacher_text.insert("end", f"{title} · {mode_label}", "teacher_title")
        self.teacher_text.insert("end", f"\n模型：{model}    状态：{status}", "teacher_meta")
        self.teacher_text.insert("end", f"\n动作：{action}    延迟：{latency_text}", "teacher_meta")
        if retry_text:
            self.teacher_text.insert("end", f"\n{retry_text}", "teacher_warning")
        self.teacher_text.insert("end", "\n\n教师思考", "teacher_section")
        self.teacher_text.insert("end", f"\n{thinking}", "teacher_body")
        self.teacher_text.insert("end", "\n\n最终输出", "teacher_section")
        self.teacher_text.insert("end", f"\n{output}", "teacher_output")
        if teacher.get("truncated"):
            self.teacher_text.insert("end", "\n\n警告：模型输出达到上限，思考或最终动作可能被截断。", "teacher_warning")
        if decision.get("assist_reason") and not decision.get("model_candidates"):
            self.teacher_text.insert("end", f"\n\n回退原因：{decision.get('assist_reason')}", "teacher_warning")
        # Streaming updates replace the whole read-only widget. Keep the
        # viewport on the newest reasoning instead of jumping back to line 1.
        self.teacher_text.see("end")
        self.teacher_text.configure(state="disabled")

    def _configure_teacher_tags(self) -> None:
        self.teacher_text.tag_configure(
            "teacher_title",
            foreground=ACCENT,
            font=(self.ui_font, 10, "bold"),
            spacing1=4,
            spacing3=5,
        )
        self.teacher_text.tag_configure(
            "teacher_meta",
            foreground=MUTED,
            font=(self.data_font, 9),
        )
        self.teacher_text.tag_configure(
            "teacher_section",
            foreground=WARNING,
            font=(self.ui_font, 10, "bold"),
            spacing1=7,
            spacing3=2,
        )
        self.teacher_text.tag_configure(
            "teacher_body",
            foreground=TEXT,
            font=(self.data_font, 9),
        )
        self.teacher_text.tag_configure(
            "teacher_output",
            foreground=ACCENT,
            font=(self.data_font, 9),
        )
        self.teacher_text.tag_configure(
            "teacher_warning",
            foreground=ERROR,
            font=(self.ui_font, 9, "bold"),
        )

    def _configure_status_tags(self) -> None:
        self.status_text.tag_configure("summary_run", foreground=ACCENT, font=(self.ui_font, 10, "bold"))
        self.status_text.tag_configure("summary_stats", foreground=TEXT, font=(self.data_font, 10, "bold"))
        self.status_text.tag_configure(
            "section_heading",
            foreground=ACCENT,
            font=(self.ui_font, 11, "bold"),
            spacing1=12,
            spacing3=3,
        )
        self.status_text.tag_configure(
            "table_header",
            foreground=MUTED,
            font=(self.ui_font, 9, "bold"),
            spacing3=3,
        )
        self.status_text.tag_configure("action_type", foreground=WARNING, font=(self.ui_font, 10, "bold"))
        self.status_text.tag_configure("action_selection", foreground=TEXT, font=(self.data_font, 10))
        self.status_text.tag_configure("action_target", foreground=MUTED, font=(self.data_font, 10))
        self.status_text.tag_configure("action_command", foreground=ACCENT, font=(self.data_font, 10, "bold"))
        self.status_text.tag_configure("deck_heading", foreground=ACCENT, font=(self.ui_font, 10, "bold"), spacing1=10, spacing3=2)
        self.status_text.tag_configure("deck_cards", foreground=TEXT, font=(self.data_font, 10))
        self.status_text.tag_configure("history_heading", foreground=ACCENT, font=(self.ui_font, 10, "bold"), spacing1=10, spacing3=2)
        self.status_text.tag_configure("history_text", foreground=MUTED, font=(self.data_font, 9))
        self.status_text.tag_configure("empty_actions", foreground=MUTED, font=(self.ui_font, 9))

    def _set_status_panel(self, overlay: Any) -> None:
        if not isinstance(overlay, dict):
            self._set_text(
                self.status_text,
                state_text_only(str(overlay or ""), self.config["overlay"]["status"]["divider_columns"]),
            )
            return

        summary = str(overlay.get("summary") or "等待当前状态。").strip()
        deck = overlay.get("deck")
        history = overlay.get("recent_actions")
        actions = overlay.get("legal_actions")
        self.status_text.configure(state="normal")
        self.status_text.delete("1.0", "end")
        summary_lines = summary.splitlines() or [summary]
        for index, line in enumerate(summary_lines):
            if index:
                self.status_text.insert("end", "\n")
            self.status_text.insert("end", line, "summary_run" if index == 0 else "summary_stats")

        if isinstance(deck, list):
            self.status_text.insert("end", "\n\n当前牌组", "deck_heading")
            deck_labels: list[str] = []
            for card in deck:
                if not isinstance(card, dict):
                    continue
                name = table_cell(card.get("name") or "未知卡牌")
                try:
                    upgrades = int(card.get("upgrades", 0) or 0)
                except (TypeError, ValueError):
                    upgrades = 0
                try:
                    count = int(card.get("count", 0) or 0)
                except (TypeError, ValueError):
                    count = 0
                if upgrades > 0:
                    name += f"+{upgrades}"
                deck_labels.append(f"{name}*{count}")
            self.status_text.insert("end", "\n" + ("  ".join(deck_labels) if deck_labels else "暂无牌组信息"), "deck_cards")

        if isinstance(history, list):
            self.status_text.insert("end", "\n\n当前房间历史（最近6步）", "history_heading")
            history_lines = [history_line(item) for item in history if isinstance(item, dict)]
            self.status_text.insert(
                "end",
                "\n" + ("\n".join(history_lines) if history_lines else "暂无历史动作"),
                "history_text",
            )

        self.status_text.insert("end", "\n\n合法动作", "section_heading")
        if not isinstance(actions, list) or not actions:
            self.status_text.insert("end", "\n当前没有可执行动作。", "empty_actions")
            self.status_text.configure(state="disabled")
            return

        self.status_text.insert("end", "\n动作类型\t选择\t目标\t动作指令", "table_header")
        for action in actions:
            if not isinstance(action, dict):
                continue
            self.status_text.insert("end", "\n")
            self.status_text.insert("end", table_cell(action.get("type")), "action_type")
            self.status_text.insert("end", "\t")
            self.status_text.insert("end", table_cell(action.get("selection")), "action_selection")
            self.status_text.insert("end", "\t")
            self.status_text.insert("end", table_cell(action.get("target")), "action_target")
            self.status_text.insert("end", "\t")
            self.status_text.insert("end", table_cell(action.get("command")), "action_command")
        self.status_text.configure(state="disabled")

    def _update_status_tabs(self, event: tk.Event[tk.Misc]) -> None:
        """Keep the three legal-action columns evenly spaced while resizing."""
        self.status_text.configure(tabs=status_tab_stops(event.width))

    def _set_connection(self, message: str, color: str = MUTED) -> None:
        self.connection_state.set(message)
        self.control_connection.set(message)
        # Color the status labels without duplicating widgets or protocol state.
        for widget in self.root.winfo_children() + self.control.winfo_children():
            self._set_descendant_status_color(widget, color)

    def _set_descendant_status_color(self, widget: tk.Misc, color: str) -> None:
        for child in widget.winfo_children():
            if isinstance(child, tk.Label) and str(child.cget("textvariable")) in {
                str(self.connection_state),
                str(self.control_connection),
            }:
                child.configure(foreground=color)
            self._set_descendant_status_color(child, color)

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self.action_entry.configure(state=state)
        self.send_button.configure(state=state)
        for button in self.quick_buttons:
            button.configure(state=state)
        for button in self.candidate_buttons:
            button.configure(state=state)

    def _clear_candidate_actions(self) -> None:
        for button in self.candidate_buttons:
            button.destroy()
        self.candidate_buttons.clear()

    def _show_view(self, packet: dict[str, Any]) -> None:
        if self.closing:
            return
        if packet.get("type") == "teacher_update":
            # A stream fragment intentionally has no game-state payload.  Do
            # not rebuild the status panel or candidate buttons for it: that
            # redraw path is expensive enough to make the game feel stalled.
            decision = packet.get("decision")
            teacher_mode = isinstance(decision, dict) and decision.get("mode") in {"teacher_collection", "prompt_evolution"}
            if not teacher_mode:
                return
            self.pending_view = True
            self._set_connection("教师正在思考", WARNING)
            mode_title, _mode_purpose = action_panel_mode_info(decision)
            self.control.title(f"StS Agent - {mode_title}")
            self.control_title_label.configure(text=mode_title)
            self.candidate_frame.grid_remove()
            self.teacher_frame.grid()
            self._set_teacher_panel(decision)
            if decision.get("mode") == "prompt_evolution":
                self.online_rl.withdraw()
                self._set_prompt_evolution_panel(packet.get("prompt_evolution"), decision.get("model_id"))
            self._set_controls_enabled(False)
            return
        self.pending_view = True
        self._set_connection("等待操作", WARNING)
        overlay = packet.get("overlay")
        if isinstance(overlay, dict):
            self._set_status_panel(overlay)
        else:
            self._set_text(
                self.status_text,
                state_text_only(
                    str(packet.get("text", "")), self.config["overlay"]["status"]["divider_columns"]
                ),
            )
        decision = packet.get("decision")
        self._clear_candidate_actions()
        mode_title, mode_purpose = action_panel_mode_info(decision)
        self.control.title(f"StS Agent - {mode_title}")
        self.control_title_label.configure(text=mode_title)
        teacher_mode = isinstance(decision, dict) and decision.get("mode") in {"teacher_collection", "prompt_evolution"}
        online_rl_mode = isinstance(decision, dict) and decision.get("mode") == "remote_online_rl"
        prompt_evolution_mode = isinstance(decision, dict) and decision.get("mode") == "prompt_evolution"
        environment_mode = isinstance(decision, dict) and decision.get("mode") == "environment_automation"
        if online_rl_mode:
            self.prompt_evolution.withdraw()
            self._set_online_rl_panel(decision.get("online_rl"), decision.get("model_id"))
        elif prompt_evolution_mode:
            self.online_rl.withdraw()
            self._set_prompt_evolution_panel(packet.get("prompt_evolution"), decision.get("model_id"))
        if teacher_mode:
            self.candidate_frame.grid_remove()
            self.teacher_frame.grid()
            self._set_teacher_panel(decision)
        else:
            self.teacher_frame.grid_remove()
            self.candidate_frame.grid()
            candidates = decision.get("model_candidates", []) if isinstance(decision, dict) else []
            model_id = display_model_name(decision.get("model_id")) if isinstance(decision, dict) else "-"
            count = len(candidates) if isinstance(candidates, list) else 0
            status = str(decision.get("assist_status") or "") if isinstance(decision, dict) else ""
            if online_rl_mode:
                self.candidate_title.configure(text="在线策略采样")
                meta = "正在采样" if packet.get("model_pending") or status == "请求中" else "已采样并自动执行"
                self.candidate_meta.configure(text=f"策略：{model_id}  ·  {meta}")
                if not packet.get("model_pending") and isinstance(decision, dict) and decision.get("model_action"):
                    selected = self._button(self.candidate_frame, online_rl_action_text(decision), lambda: None, compact=True)
                    selected.configure(
                        anchor="w",
                        padx=10,
                        state="disabled",
                        background="#256f62",
                        disabledforeground=TEXT,
                    )
                    selected.grid(row=1, column=0, sticky="ew", pady=(0, 5), padx=7)
                    self.candidate_buttons.append(selected)
                    if isinstance(candidates, list):
                        row = 2
                        for candidate in candidates:
                            if not isinstance(candidate, dict) or candidate.get("action_id") == decision.get("model_action"):
                                continue
                            item = self._button(
                                self.candidate_frame,
                                online_rl_distribution_text(candidate),
                                lambda: None,
                                compact=True,
                            )
                            item.configure(anchor="w", padx=10, state="disabled", disabledforeground=MUTED)
                            item.grid(row=row, column=0, sticky="ew", pady=(0, 5), padx=7)
                            self.candidate_buttons.append(item)
                            row += 1
                elif isinstance(decision, dict) and decision.get("assist_reason"):
                    fallback = self._button(
                        self.candidate_frame,
                        "策略不可用：" + table_cell(decision.get("assist_reason")),
                        lambda: None,
                        compact=True,
                    )
                    fallback.configure(anchor="w", padx=10, state="disabled", disabledforeground=ERROR)
                    fallback.grid(row=1, column=0, sticky="ew", pady=(0, 5), padx=7)
                    self.candidate_buttons.append(fallback)
            else:
                if environment_mode:
                    environment = decision.get("environment") if isinstance(decision, dict) else {}
                    reason = table_cell(environment.get("reason") if isinstance(environment, dict) else "推进界面")
                    self.candidate_title.configure(text="环境自动推进")
                    self.candidate_meta.configure(text="Controller  ·  已自动执行")
                    automatic = self._button(self.candidate_frame, reason, lambda: None, compact=True)
                    automatic.configure(anchor="w", padx=10, state="disabled", disabledforeground=TEXT)
                    automatic.grid(row=1, column=0, sticky="ew", pady=(0, 5), padx=7)
                    self.candidate_buttons.append(automatic)
                else:
                    title_by_mode = {
                        "baseline_inference": "基线模型动作",
                        "candidate_assist": "候选动作",
                        "confidence_auto": "置信度评分",
                    }
                    mode = decision.get("mode") if isinstance(decision, dict) else None
                    self.candidate_title.configure(text=title_by_mode.get(mode, "动作候选"))
                    if packet.get("model_pending") or status == "请求中":
                        meta = "正在计算"
                    elif mode == "confidence_auto":
                        confidence = decision.get("confidence") if isinstance(decision, dict) else {}
                        route = confidence.get("route") if isinstance(confidence, dict) else None
                        meta = "高置信度，已自动执行" if route == "auto_execute" else "低置信度，等待确认"
                    elif mode == "baseline_inference":
                        api = decision.get("api", {})
                        meta = f"{status} · {api.get('latency_ms', '-')} ms · 输入 {api.get('input_tokens', '-')} tokens"
                        if decision.get("assist_reason") and not candidates:
                            meta += " · " + table_cell(decision["assist_reason"])
                    else:
                        meta = f"{count} 个选项"
                    self.candidate_meta.configure(text=f"模型：{model_id}  ·  {meta}")
            if not teacher_mode and not environment_mode and not online_rl_mode and isinstance(decision, dict):
                self.candidate_meta.configure(
                    text=f"{self.candidate_meta.cget('text')}  |  {mode_purpose}"
                )
            elif not isinstance(decision, dict):
                self.candidate_title.configure(text="人工动作")
                self.candidate_meta.configure(text=mode_purpose)
            if not online_rl_mode and isinstance(candidates, list):
                for index, candidate in enumerate(candidates, start=1):
                    if not isinstance(candidate, dict):
                        continue
                    label = candidate_button_text(candidate, index)
                    command = "a" if index == 1 else f"pick {index}"
                    button = self._button(self.candidate_frame, label, lambda value=command: self.submit_action(value), compact=True)
                    button.configure(anchor="w", padx=10)
                    if index == 1:
                        button.configure(
                            background="#256f62",
                            activebackground="#349984",
                            font=(self.ui_font, 9, "bold"),
                        )
                    button.grid(row=index, column=0, sticky="ew", pady=(0, 5), padx=7)
                    self.candidate_buttons.append(button)
        if packet.get("model_pending"):
            self._set_controls_enabled(False)
            self._set_connection("模型正在计算候选", WARNING)
        elif packet.get("auto_execute"):
            self.pending_view = False
            self._set_controls_enabled(False)
            prefix = (
                "在线策略已执行"
                if online_rl_mode
                else "基线策略已执行"
                if decision.get("mode") == "baseline_inference"
                else "环境已自动执行"
                if environment_mode
                else "教师已自动执行"
            )
            self._set_connection(prefix + "：" + str(decision.get("model_action", "-")), ACCENT)
        else:
            self._set_controls_enabled(True)
            self.action_entry.focus_set()

    def _show_reflection_dialog(self, packet: dict[str, Any]) -> None:
        """Collect two short natural-language labels after an online rollout."""

        if self.closing:
            self.reflection_responses.put({"type": "posthoc_reflection", "skip": True})
            return
        if self.reflection_dialog is not None and self.reflection_dialog.winfo_exists():
            return
        self.pending_view = False
        self._set_controls_enabled(False)
        self._set_connection("等待本局终局反思", WARNING)
        self._show_state_panel_for_prompt()
        # Do not attach this prompt to MODEL / ACTION: that panel may be
        # minimized while an online rollout ends and needs human feedback.
        dialog = tk.Toplevel(self.root)
        self.reflection_dialog = dialog
        dialog.title("StS Agent - Rollout Reflection")
        dialog.geometry("780x760+540+120")
        dialog.minsize(580, 540)
        dialog.configure(background=BACKGROUND)
        dialog.attributes("-topmost", self.always_on_top)
        dialog.grab_set()
        outer = tk.Frame(dialog, background=BACKGROUND, padx=14, pady=14)
        outer.grid(sticky="nsew")
        dialog.grid_rowconfigure(0, weight=1)
        dialog.grid_columnconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(3, weight=3)
        outer.grid_rowconfigure(5, weight=1)
        outer.grid_rowconfigure(7, weight=1)
        header = tk.Frame(outer, background=BACKGROUND)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="ROLLOUT REFLECTION",
            foreground=ACCENT,
            background=BACKGROUND,
            font=(self.ui_font, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        if self.config["overlay"]["frameless"]:
            self._minimize_button(header, dialog, background=BACKGROUND).grid(row=0, column=1, sticky="e")
        tk.Label(
            outer,
            text=posthoc_reflection_text(packet),
            foreground=MUTED,
            background=BACKGROUND,
            font=(self.data_font, 9),
        ).grid(row=1, column=0, sticky="w", pady=(3, 12))
        tk.Label(
            outer,
            text="本局动作复盘",
            foreground=TEXT,
            background=BACKGROUND,
            font=(self.ui_font, 10, "bold"),
        ).grid(row=2, column=0, sticky="w", pady=(0, 4))
        review_frame = tk.Frame(outer, background=FIELD, highlightbackground="#4c5053", highlightthickness=1)
        review_frame.grid(row=3, column=0, sticky="nsew")
        review_frame.grid_rowconfigure(0, weight=1)
        review_frame.grid_columnconfigure(0, weight=1)
        review = tk.Text(
            review_frame,
            wrap="word",
            state="normal",
            background=FIELD,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=9,
            pady=7,
            font=(self.data_font, 9),
        )
        review.grid(row=0, column=0, sticky="nsew")
        review.insert("1.0", action_review_text(packet.get("action_review")))
        review.configure(state="disabled")
        review_scroll = tk.Scrollbar(
            review_frame,
            command=review.yview,
            relief="flat",
            borderwidth=0,
            background=BUTTON,
            activebackground=BUTTON_ACTIVE,
            troughcolor=FIELD,
        )
        review_scroll.grid(row=0, column=1, sticky="ns")
        review.configure(yscrollcommand=review_scroll.set)
        tk.Label(
            outer,
            text="可强化之处",
            foreground=ACCENT,
            background=BACKGROUND,
            font=(self.ui_font, 10, "bold"),
        ).grid(row=4, column=0, sticky="w", pady=(10, 4))
        reinforce = tk.Text(
            outer,
            height=6,
            wrap="word",
            background=FIELD,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=9,
            pady=7,
            font=(self.data_font, 10),
        )
        reinforce.grid(row=5, column=0, sticky="nsew")
        tk.Label(
            outer,
            text="应弱化之处",
            foreground=WARNING,
            background=BACKGROUND,
            font=(self.ui_font, 10, "bold"),
        ).grid(row=6, column=0, sticky="w", pady=(10, 4))
        weaken = tk.Text(
            outer,
            height=6,
            wrap="word",
            background=FIELD,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=9,
            pady=7,
            font=(self.data_font, 10),
        )
        weaken.grid(row=7, column=0, sticky="nsew")
        buttons = tk.Frame(outer, background=BACKGROUND)
        buttons.grid(row=8, column=0, sticky="e", pady=(12, 0))

        def finish(skip: bool) -> None:
            if self.reflection_dialog is not dialog:
                return
            response = {
                "type": "posthoc_reflection",
                "skip": skip,
                "reinforce": "" if skip else reinforce.get("1.0", "end-1c").strip(),
                "weaken": "" if skip else weaken.get("1.0", "end-1c").strip(),
            }
            try:
                dialog.grab_release()
                dialog.destroy()
            except tk.TclError:
                pass
            self.reflection_dialog = None
            self.reflection_responses.put(response)
            self._set_connection("终局反思已提交；正在请求教师归因", ACCENT)

        skip_button = self._button(buttons, "跳过本局", lambda: finish(True), compact=True)
        skip_button.grid(row=0, column=0, padx=(0, 8))
        submit = self._button(buttons, "提交反思", lambda: finish(False), compact=True)
        submit.configure(
            background=ACCENT,
            foreground="#10221f",
            activebackground="#61ddc2",
            activeforeground="#10221f",
            font=(self.ui_font, 9, "bold"),
        )
        submit.grid(row=0, column=1)
        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(True))
        reinforce.focus_set()
        dialog.lift()

    def _show_attribution_result(self, packet: dict[str, Any]) -> None:
        """Show the teacher's post-hoc selected steps without blocking rollout."""

        if self.closing:
            return
        if self.attribution_dialog is not None:
            try:
                self.attribution_dialog.destroy()
            except tk.TclError:
                pass
        # This result must remain visible even if MODEL / ACTION is minimized.
        dialog = tk.Toplevel(self.root)
        self.attribution_dialog = dialog
        dialog.title("StS Agent - Teacher Attribution")
        dialog.geometry("700x540+650+140")
        dialog.minsize(480, 340)
        dialog.configure(background=BACKGROUND)
        dialog.attributes("-topmost", self.always_on_top)
        outer = tk.Frame(dialog, background=BACKGROUND, padx=12, pady=12)
        outer.grid(sticky="nsew")
        dialog.grid_rowconfigure(0, weight=1)
        dialog.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        header = tk.Frame(outer, background=BACKGROUND)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="TEACHER ATTRIBUTION",
            foreground=ACCENT,
            background=BACKGROUND,
            font=(self.ui_font, 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        if self.config["overlay"]["frameless"]:
            self._minimize_button(header, dialog, background=BACKGROUND).grid(row=0, column=1, sticky="e")
        text_frame = tk.Frame(outer, background=FIELD, highlightbackground="#4c5053", highlightthickness=1)
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        text = tk.Text(
            text_frame,
            wrap="word",
            state="normal",
            background=FIELD,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=9,
            font=(self.data_font, 10),
        )
        text.grid(row=0, column=0, sticky="nsew")
        text.insert("1.0", attribution_result_text(packet))
        text.configure(state="disabled")
        scrollbar = tk.Scrollbar(
            text_frame,
            command=text.yview,
            relief="flat",
            borderwidth=0,
            background=BUTTON,
            activebackground=BUTTON_ACTIVE,
            troughcolor=FIELD,
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        dialog.protocol("WM_DELETE_WINDOW", lambda: self._close_attribution_dialog(dialog))


    def _show_prompt_evolution_review(self, packet: dict[str, Any]) -> None:
        """Show either the pre-validation candidate gate or result gate."""

        if self.closing:
            self.prompt_evolution_responses.put({"type": "prompt_evolution_review", "verdict": "abandon"})
            return
        if self.reflection_dialog is not None and self.reflection_dialog.winfo_exists():
            return
        self.pending_view = False
        self._set_controls_enabled(False)
        dialog = tk.Toplevel(self.root)
        self.reflection_dialog = dialog
        dialog.title("StS Agent - Prompt Evolution Review")
        dialog.geometry("820x760+520+100")
        dialog.minsize(620, 520)
        dialog.configure(background=BACKGROUND)
        dialog.attributes("-topmost", self.always_on_top)
        dialog.grab_set()
        outer = tk.Frame(dialog, background=BACKGROUND, padx=14, pady=14)
        outer.grid(sticky="nsew")
        dialog.grid_rowconfigure(0, weight=1)
        dialog.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)
        outer.grid_rowconfigure(3, weight=0)
        outer.grid_rowconfigure(5, weight=0)
        outer.grid_columnconfigure(0, weight=1)
        header = tk.Frame(outer, background=BACKGROUND)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        header.grid_columnconfigure(0, weight=1)
        review = packet.get("review") if isinstance(packet.get("review"), dict) else {}
        phase = str(review.get("phase") or packet.get("phase") or "validation")
        tk.Label(
            header,
            text="PROMPT EVOLUTION REVIEW",
            foreground=ACCENT,
            background=BACKGROUND,
            font=(self.ui_font, 12, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self._minimize_button(header, dialog, background=BACKGROUND).grid(row=0, column=1, sticky="e")
        text_frame = tk.Frame(outer, background=FIELD, highlightbackground="#4c5053", highlightthickness=1)
        text_frame.grid(row=1, column=0, sticky="nsew")
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        text = tk.Text(
            text_frame,
            wrap="word",
            state="disabled",
            background=FIELD,
            foreground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=9,
            font=(self.data_font, 10),
        )
        text.grid(row=0, column=0, sticky="nsew")
        text.configure(state="normal")
        text.insert("1.0", prompt_evolution_review_text(packet.get("review")))
        text.tag_configure("pe_removed", foreground="#ef7373", background="#4a2d32")
        text.tag_configure("pe_added", foreground="#7cdb9a", background="#284336")
        tag_prompt_evolution_skill_changes(text, packet.get("review"))
        text.configure(state="disabled")
        scrollbar = tk.Scrollbar(text_frame, command=text.yview, relief="flat", borderwidth=0, background=BUTTON, troughcolor=FIELD)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        tk.Label(
            outer,
            text="拒绝理由（拒绝时必填；会作为下一轮教师修改依据）",
            foreground=WARNING,
            background=BACKGROUND,
            font=(self.ui_font, 10, "bold"),
        ).grid(row=2, column=0, sticky="w", pady=(10, 4))
        reason = tk.Text(
            outer,
            height=4,
            wrap="word",
            background=FIELD,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=9,
            pady=7,
            font=(self.data_font, 10),
        )
        reason.grid(row=3, column=0, sticky="ew")
        buttons = tk.Frame(outer, background=BACKGROUND)
        buttons.grid(row=4, column=0, sticky="e", pady=(12, 0))

        def finish(verdict: str) -> None:
            if self.reflection_dialog is not dialog:
                return
            response = {
                "type": "prompt_evolution_review",
                "verdict": verdict,
                "reason": reason.get("1.0", "end-1c").strip(),
            }
            if verdict == "reject" and not response["reason"]:
                messagebox.showwarning("需要拒绝理由", "拒绝候选时请填写理由。", parent=dialog)
                return
            try:
                dialog.grab_release()
                dialog.destroy()
            except tk.TclError:
                pass
            self.reflection_dialog = None
            self.prompt_evolution_responses.put(response)
            self._set_connection("Prompt 候选审核已提交", ACCENT)

        abandon = self._button(buttons, "放弃本次", lambda: finish("abandon"), compact=True)
        abandon.grid(row=0, column=0, padx=(0, 8))
        reject = self._button(buttons, "拒绝并重写", lambda: finish("reject"), compact=True)
        reject.configure(background="#6b4b2d", activebackground="#8a623c")
        reject.grid(row=0, column=1, padx=(0, 8))
        accept_label = "开始验证候选" if phase == "candidate" else "接受并固化"
        accept = self._button(buttons, accept_label, lambda: finish("accept"), compact=True)
        accept.configure(background=ACCENT, foreground="#10221f", activebackground="#61ddc2", activeforeground="#10221f", font=(self.ui_font, 9, "bold"))
        accept.grid(row=0, column=2)
        dialog.protocol("WM_DELETE_WINDOW", lambda: finish("abandon"))
        dialog.lift()
        text.see("1.0")

    def _close_attribution_dialog(self, dialog: tk.Toplevel) -> None:
        if self.attribution_dialog is dialog:
            self.attribution_dialog = None
        try:
            dialog.destroy()
        except tk.TclError:
            pass

    def _submit_entry(self, _event: tk.Event[tk.Misc]) -> str:
        self.submit_action()
        return "break"

    def submit_action(self, action: str | None = None) -> None:
        if not self.pending_view:
            return
        command = (action if action is not None else self.action_entry.get()).strip()
        if not command:
            return
        if command.lower() in {"h", "help", "?"}:
            self.show_help()
            return
        self.pending_view = False
        self.action_entry.delete(0, "end")
        self._set_controls_enabled(False)
        self._set_connection(f"已发送：{command}", MUTED)
        self.actions.put(command)

    def show_help(self) -> None:
        messagebox.showinfo("Overlay 操作帮助", ACTION_HELP, parent=self.control)

    def connect(self) -> None:
        if self.closing or (self.network_thread is not None and self.network_thread.is_alive()):
            return
        self._drain_actions()
        self.pending_view = False
        self._set_controls_enabled(False)
        self._set_connection(f"连接 {self.host}:{self.port}...", WARNING)
        self.network_thread = threading.Thread(target=self._network_loop, name="sts-overlay-client", daemon=True)
        self.network_thread.start()

    def _network_loop(self) -> None:
        connection: socket.socket | None = None
        try:
            connection = socket.create_connection((self.host, self.port), timeout=30)
            connection.settimeout(None)
            self._post_ui(self._set_connection, f"已连接 {self.host}:{self.port}", ACCENT)
            with connection:
                reader = connection.makefile("r", encoding="utf-8", newline="\n")
                writer = connection.makefile("w", encoding="utf-8", newline="\n")
                try:
                    for line in reader:
                        try:
                            packet = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(packet, dict):
                            continue
                        packet_type = packet.get("type")
                        if packet_type == "hello":
                            self._post_ui(self._set_connection, str(packet.get("message", "已连接")), ACCENT)
                        elif packet_type in {"view", "teacher_update", "model_update"}:
                            self._post_ui(self._show_view, packet)
                            if packet_type in {"teacher_update", "model_update"} or packet.get("auto_execute"):
                                continue
                            action = self.actions.get()
                            writer.write(json.dumps({"type": "input", "text": action}, ensure_ascii=False) + "\n")
                            writer.flush()
                        elif packet_type == "online_rl_update":
                            self._post_ui(self._show_online_rl_update, packet)
                        elif packet_type == "prompt_evolution_update":
                            self._post_ui(self._show_prompt_evolution_update, packet)
                        elif packet_type == "posthoc_reflection_request":
                            self._post_ui(self._show_reflection_dialog, packet)
                            response = self.reflection_responses.get()
                            writer.write(json.dumps(response, ensure_ascii=False) + "\n")
                            writer.flush()
                        elif packet_type == "posthoc_attribution_result":
                            self._post_ui(self._show_attribution_result, packet)
                        elif packet_type == "prompt_evolution_review_request":
                            self._post_ui(self._show_prompt_evolution_review, packet)
                            response = self.prompt_evolution_responses.get()
                            writer.write(json.dumps(response, ensure_ascii=False) + "\n")
                            writer.flush()
                        elif packet_type == "shutdown":
                            self._post_ui(self._set_connection, str(packet.get("message", "controller 已停止")), ERROR)
                            break
                finally:
                    reader.close()
                    writer.close()
        except OSError as exc:
            if not self.closing:
                self._post_ui(self._connection_failed, str(exc))
        finally:
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
            if not self.closing:
                self._post_ui(self._connection_closed)

    def _connection_failed(self, reason: str) -> None:
        self.pending_view = False
        self._set_controls_enabled(False)
        self._set_connection(f"连接失败：{reason}", ERROR)

    def _connection_closed(self) -> None:
        self.pending_view = False
        self._set_controls_enabled(False)
        if self.connection_state.get().startswith("已发送：q"):
            self._set_connection("Overlay 已断开；controller 仍在运行", MUTED)
        elif not self.connection_state.get().startswith("连接失败"):
            self._set_connection("连接已关闭；可点击重连", ERROR)

    def _post_ui(self, callback: Callable[..., None], *args: Any) -> None:
        if self.closing:
            return
        try:
            self.root.after(0, callback, *args)
        except tk.TclError:
            pass

    def _drain_actions(self) -> None:
        while True:
            try:
                self.actions.get_nowait()
            except queue.Empty:
                return

    def _schedule_layout_save(self, _event: tk.Event[tk.Misc]) -> None:
        if self.closing:
            return
        if self._save_after is not None:
            self.root.after_cancel(self._save_after)
        self._save_after = self.root.after(400, self._save_layout)

    def _save_layout(self) -> None:
        self._save_after = None
        try:
            self.layout_path.parent.mkdir(parents=True, exist_ok=True)
            self.layout_path.write_text(
                json.dumps(
                    {
                        "status": self.root.winfo_geometry(),
                        "control": self.control.winfo_geometry(),
                        "online_rl": self.online_rl.winfo_geometry(),
                        "prompt_evolution": self.prompt_evolution.winfo_geometry(),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except (OSError, tk.TclError):
            pass

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        for callback in (self._save_after, self._topmost_after):
            if callback is not None:
                try:
                    self.root.after_cancel(callback)
                except tk.TclError:
                    pass
        self._save_after = None
        self._topmost_after = None
        self._save_layout()
        self.reflection_responses.put({"type": "posthoc_reflection", "skip": True})
        self.prompt_evolution_responses.put({"type": "prompt_evolution_review", "verdict": "abandon"})
        self.actions.put("q")
        if self.reflection_dialog is not None:
            try:
                self.reflection_dialog.destroy()
            except tk.TclError:
                pass
            self.reflection_dialog = None
        if self.attribution_dialog is not None:
            try:
                self.attribution_dialog.destroy()
            except tk.TclError:
                pass
            self.attribution_dialog = None
        try:
            self.control.destroy()
        except tk.TclError:
            pass
        try:
            self.online_rl.destroy()
        except tk.TclError:
            pass
        try:
            self.prompt_evolution.destroy()
        except tk.TclError:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    enable_windows_dpi_awareness()
    parser = argparse.ArgumentParser(description="Translucent two-window overlay client for the StS controller")
    parser.add_argument("--config", default="config/agent.json", help="项目根目录下的配置文件路径")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-topmost", action="store_true", help="本次启动不置顶窗口")
    args = parser.parse_args()
    try:
        config = load_agent_config(config_file=args.config)
    except ConfigError as exc:
        print(f"配置文件错误：{exc}", file=sys.stderr)
        return 2
    host = args.host or config["server"]["host"]
    port = args.port or config["server"]["port"]
    try:
        app = OverlayTerminal(config, host, port, config["overlay"]["always_on_top"] and not args.no_topmost)
    except tk.TclError as exc:
        print(f"无法启动 Overlay：{exc}", file=sys.stderr)
        return 2
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
