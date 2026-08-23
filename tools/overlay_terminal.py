"""Two-window translucent desktop overlay for the StS controller.

The controller owns game actions and only accepts one TCP client at a time.
This is an alternative to human_terminal.py: it renders the same protocol in
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

from agent_config import ConfigError, load_agent_config


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
    """Keep status and legal actions out of the model/action window."""
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


def model_feedback(decision: Any) -> str:
    if not isinstance(decision, dict):
        return "模型反馈\n等待当前状态。"
    lines = ["模型反馈"]
    status = decision.get("assist_status")
    reason = decision.get("assist_reason")
    if status:
        lines.append(f"状态：{status}")
    if reason:
        lines.append(str(reason))
    confidence = decision.get("confidence")
    if isinstance(confidence, dict):
        route = "自动执行" if confidence.get("route") == "auto_execute" else "等待人工确认"
        lines.append(
            f"p1 {format_probability(confidence.get('top_probability'))}  "
            f"margin {format_probability(confidence.get('margin'))}  "
            f"entropy {confidence.get('entropy', '-')}"
        )
        lines.append(f"路由：{route}")
    candidates = decision.get("model_candidates")
    if isinstance(candidates, list) and candidates:
        lines.append("候选动作：")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            rank = candidate.get("rank", "?")
            label = candidate.get("label") or candidate.get("action_id") or "未知动作"
            score = candidate.get("score")
            score_text = f"  p={format_probability(score)}" if isinstance(score, (int, float)) else ""
            lines.append(f"[{rank}] {label}{score_text}")
    elif not status:
        lines.append("当前没有可显示的模型候选。")
    return "\n".join(lines)


class OverlayTerminal:
    def __init__(self, config: dict[str, Any], host: str, port: int, always_on_top: bool) -> None:
        self.config = config
        self.host = host
        self.port = port
        self.always_on_top = always_on_top
        self.layout_path = Path(__file__).resolve().parents[1] / "data" / "runtime" / "overlay-layout.json"
        self.saved_layout = self._load_layout()
        self.actions: queue.Queue[str] = queue.Queue()
        self.network_thread: threading.Thread | None = None
        self.closing = False
        self.pending_view = False
        self._save_after: str | None = None

        self.root = tk.Tk()
        self.root.title("StS Agent - State")
        self.control = tk.Toplevel(self.root)
        self.control.title("StS Agent - Model & Action")
        self.ui_font = self._choose_font("Microsoft YaHei UI", "Noto Sans SC", "Microsoft YaHei", "Segoe UI")
        self.data_font = self._choose_font("Microsoft YaHei UI", "Noto Sans SC", "Microsoft YaHei", "Cascadia Mono")
        self._drag_origin: dict[str, tuple[int, int]] = {}
        self._resize_origin: dict[str, tuple[int, int, int, int]] = {}
        self._configure_window(self.root, "status", 360, 260)
        self._configure_window(self.control, "control", 380, 260)
        self._build_status_window()
        self._build_control_window()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.control.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Configure>", self._schedule_layout_save, add="+")
        self.control.bind("<Configure>", self._schedule_layout_save, add="+")
        self._set_controls_enabled(False)
        self._set_connection("正在连接 controller...", WARNING)
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
            if name in {"status", "control"} and isinstance(geometry, str) and GEOMETRY_PATTERN.fullmatch(geometry)
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
        close_button.grid(row=0, column=2, sticky="e")
        self._bind_drag(bar, window)
        self._bind_drag(label, window)
        return bar

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

    def _resize_grip(self, parent: tk.Misc, window: tk.Misc) -> tk.Frame:
        grip = tk.Frame(parent, background=BACKGROUND, width=16, height=16, cursor="size_nw_se")
        grip.grid_propagate(False)
        key = str(window)

        def begin(event: tk.Event[tk.Misc]) -> None:
            self._resize_origin[key] = (event.x_root, event.y_root, window.winfo_width(), window.winfo_height())

        def resize(event: tk.Event[tk.Misc]) -> None:
            origin = self._resize_origin.get(key)
            if origin is None:
                return
            x, y, width, height = origin
            min_width, min_height = window.minsize()
            window.geometry(f"{max(min_width, width + event.x_root - x)}x{max(min_height, height + event.y_root - y)}")

        grip.bind("<ButtonPress-1>", begin)
        grip.bind("<B1-Motion>", resize)
        return grip

    def _build_status_window(self) -> None:
        outer = tk.Frame(self.root, background=BACKGROUND, padx=10, pady=10)
        outer.grid(sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(2, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        self._title_bar(outer, self.root, "CURRENT STATE").grid(row=0, column=0, sticky="ew", pady=(0, 7))
        self.connection_state = tk.StringVar()
        tk.Label(
            outer,
            textvariable=self.connection_state,
            foreground=MUTED,
            background=BACKGROUND,
            font=(self.ui_font, 9),
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))

        text_frame = tk.Frame(outer, background=PANEL, highlightbackground="#4c5053", highlightthickness=1)
        text_frame.grid(row=2, column=0, sticky="nsew")
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
        footer = tk.Frame(outer, background=BACKGROUND, height=16)
        footer.grid(row=3, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)
        self._resize_grip(footer, self.root).grid(row=0, column=1, sticky="se")

    def _build_control_window(self) -> None:
        outer = tk.Frame(self.control, background=BACKGROUND, padx=10, pady=10)
        outer.grid(sticky="nsew")
        self.control.grid_rowconfigure(0, weight=1)
        self.control.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(2, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        title_bar = self._title_bar(outer, self.control, "MODEL / ACTION")
        title_bar.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        self.reconnect_button = self._button(title_bar, "重连", self.connect, compact=True)
        self.reconnect_button.grid(row=0, column=1, sticky="e", padx=(0, 4))
        self.control_connection = tk.StringVar()
        tk.Label(
            outer,
            textvariable=self.control_connection,
            foreground=MUTED,
            background=BACKGROUND,
            font=(self.ui_font, 9),
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))

        feedback_frame = tk.Frame(outer, background=PANEL, highlightbackground="#4c5053", highlightthickness=1)
        feedback_frame.grid(row=2, column=0, sticky="nsew")
        feedback_frame.grid_rowconfigure(0, weight=1)
        feedback_frame.grid_columnconfigure(0, weight=1)
        self.feedback_text = tk.Text(
            feedback_frame,
            wrap="word",
            height=10,
            state="disabled",
            background=FIELD,
            foreground=TEXT,
            relief="flat",
            borderwidth=0,
            padx=10,
            pady=8,
            font=(self.ui_font, 10),
        )
        self.feedback_text.grid(row=0, column=0, sticky="nsew")

        self.candidate_frame = tk.Frame(outer, background=BACKGROUND)
        self.candidate_frame.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        self.candidate_frame.grid_columnconfigure(0, weight=1)
        self.candidate_frame.grid_columnconfigure(1, weight=1)
        self.candidate_buttons: list[tk.Button] = []
        self.quick_buttons: list[tk.Button] = []

        action_frame = tk.Frame(outer, background=BACKGROUND)
        action_frame.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        action_frame.grid_columnconfigure(0, weight=1)
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
        self.action_entry.grid(row=0, column=0, sticky="ew", ipady=6)
        self.action_entry.bind("<Return>", self._submit_entry)
        self.send_button = self._button(action_frame, "执行", self.submit_action)
        self.send_button.grid(row=0, column=1, padx=(8, 0), sticky="ew")

        quick = tk.Frame(outer, background=BACKGROUND)
        quick.grid(row=5, column=0, sticky="ew", pady=(8, 0))
        for index, (label, action) in enumerate((("结束回合", "e"), ("继续", "c"), ("返回", "x"))):
            quick.grid_columnconfigure(index, weight=1)
            button = self._button(quick, label, lambda value=action: self.submit_action(value), compact=True)
            button.grid(row=0, column=index, sticky="ew", padx=(0 if index == 0 else 5, 0))
            self.quick_buttons.append(button)
        resize_bar = tk.Frame(outer, background=BACKGROUND, height=16)
        resize_bar.grid(row=6, column=0, sticky="ew")
        resize_bar.grid_columnconfigure(0, weight=1)
        self._resize_grip(resize_bar, self.control).grid(row=0, column=1, sticky="se")

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
        )

    def _set_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

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
        self.pending_view = True
        self._set_connection("等待操作", WARNING)
        self._set_text(
            self.status_text,
            state_text_only(str(packet.get("text", "")), self.config["overlay"]["status"]["divider_columns"]),
        )
        decision = packet.get("decision")
        self._set_text(self.feedback_text, model_feedback(decision))
        self._clear_candidate_actions()
        candidates = decision.get("model_candidates", []) if isinstance(decision, dict) else []
        if isinstance(candidates, list):
            for index, candidate in enumerate(candidates, start=1):
                if not isinstance(candidate, dict):
                    continue
                action_id = str(candidate.get("action_id") or candidate.get("label") or f"候选 {index}")
                label = f"接受 {index}: {action_id}"
                command = "a" if index == 1 else f"pick {index}"
                button = self._button(self.candidate_frame, label, lambda value=command: self.submit_action(value), compact=True)
                button.grid(row=(index - 1) // 2, column=(index - 1) % 2, sticky="ew", padx=((index - 1) % 2 * 5, 0), pady=(0, 5))
                self.candidate_buttons.append(button)
        self._set_controls_enabled(True)
        self.action_entry.focus_set()

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
                        elif packet_type == "view":
                            self._post_ui(self._show_view, packet)
                            action = self.actions.get()
                            writer.write(json.dumps({"type": "input", "text": action}, ensure_ascii=False) + "\n")
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
                    {"status": self.root.winfo_geometry(), "control": self.control.winfo_geometry()},
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
        self._save_layout()
        self.actions.put("q")
        try:
            self.control.destroy()
        except tk.TclError:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    enable_windows_dpi_awareness()
    try:
        config = load_agent_config()
    except ConfigError as exc:
        print(f"配置文件错误：{exc}", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description="Translucent two-window overlay client for the StS controller")
    parser.add_argument("--host", default=config["server"]["host"])
    parser.add_argument("--port", type=int, default=config["server"]["port"])
    parser.add_argument("--no-topmost", action="store_true", help="本次启动不置顶窗口")
    args = parser.parse_args()
    try:
        app = OverlayTerminal(config, args.host, args.port, config["overlay"]["always_on_top"] and not args.no_topmost)
    except tk.TclError as exc:
        print(f"无法启动 Overlay：{exc}", file=sys.stderr)
        return 2
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
