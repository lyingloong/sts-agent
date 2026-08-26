"""OpenAI-compatible remote teacher for StS SFT data collection.

The client receives only an already-normalized observation and controller
provided legal action aliases.  It never receives an API key through a prompt
and never communicates with CommunicationMod directly.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, TextIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from teacher_prompts import PROMPT_VERSION, build_system_prompt


class TeacherError(RuntimeError):
    """A recoverable remote-teacher error that should fall back to a human."""


def compact_json(value: Any, limit: int = 32000) -> str:
    """Serialize a public observation, with an optional configured hard cap."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def action_id(action: dict[str, Any]) -> str:
    return str(action.get("id") or action.get("action_id") or "")


def action_label(action: dict[str, Any]) -> str:
    return str(action.get("label") or action.get("kind") or action.get("command") or "")


def read_api_key(path: Path) -> str:
    """Read ``key: ...`` or a bare key from an ignored local file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TeacherError(f"cannot read teacher.api_key_file: {path}") from exc
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in stripped:
            name, value = stripped.split(":", 1)
            if name.strip().lower() in {"key", "api_key", "api-key"}:
                candidate = value.strip()
                if candidate:
                    return candidate
        elif stripped:
            return stripped
    raise TeacherError(f"teacher.api_key_file contains no key: {path}")


def build_prompt(
    observation: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    max_input_chars: int = 32000,
) -> str:
    lines = [
        "You are an expert Slay the Spire teacher producing an imitation-learning action label.",
        "Choose one action that maximizes long-term win probability from the supplied legal actions.",
        "The observation contains only public game information. recent_actions, event_history, and selection_context are context; the current legal actions are authoritative.",
        'Return exactly one JSON object: {"action_id":"A1"}. Do not include reasoning, prose, markdown, or an action outside the list.',
        "Observation:",
        compact_json(observation, limit=max_input_chars),
        "Legal actions:",
    ]
    for action in legal_actions:
        lines.append(f"- {action_id(action)} | {action_label(action)}")
    return "\n".join(lines)


def parse_action_id(content: str, legal_ids: set[str]) -> str:
    """Accept strict JSON first, then a single exact legal alias as fallback."""
    match = re.search(r"\{.*?\}", content, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            candidate = str(payload.get("action_id", ""))
            if candidate in legal_ids:
                return candidate
    exact = content.strip()
    if exact in legal_ids:
        return exact
    matches = [
        candidate
        for candidate in legal_ids
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(candidate)}(?![A-Za-z0-9_])", content)
    ]
    if len(matches) == 1:
        return matches[0]
    raise TeacherError("teacher response did not contain one current legal action ID")


def _usage_value(usage: Any, name: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(name)
    return value if isinstance(value, int) and value >= 0 else None


def display_text(value: Any, limit: int = 4000) -> str:
    """Bound remote text before it is shown in the overlay or trace."""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    return text if len(text) <= limit else text[:limit] + "...<truncated>"


TeacherUpdate = Callable[[dict[str, Any]], None]


class RemoteTeacherClient:
    """Select exactly one action through an OpenAI-compatible chat endpoint."""

    def __init__(self, project_root: Path, config: dict[str, Any]) -> None:
        self.enabled = bool(config["enabled"])
        self.base_url = str(config["base_url"]).rstrip("/")
        self.model = str(config["model"])
        self.api_key_path = project_root / str(config["api_key_file"])
        self.timeout_seconds = float(config["timeout_seconds"])
        self.max_retries = int(config["max_retries"])
        self.temperature = float(config["temperature"])
        self.max_tokens = int(config["max_tokens"])
        self.max_input_chars = int(config["max_input_chars"])
        self.thinking = bool(config["thinking"])
        self.json_mode = bool(config["json_mode"])
        self.streaming = bool(config["streaming"])
        self.screens = tuple(item.strip().upper() for item in config["screens"])
        self.request_id = 0
        self.last_error: str | None = None
        self._api_key: str | None = None

    def preflight(self) -> str | None:
        """Validate local credentials before a game run, without network I/O."""
        if not self.enabled:
            return None
        try:
            self._api_key = read_api_key(self.api_key_path)
        except TeacherError as exc:
            self.last_error = str(exc)
            return self.last_error
        self.last_error = None
        return None

    def _api_key_value(self) -> str:
        if self._api_key is None:
            self._api_key = read_api_key(self.api_key_path)
        return self._api_key

    def _payload(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": build_system_prompt(observation),
                },
                {
                    "role": "user",
                    "content": build_prompt(observation, legal_actions, self.max_input_chars),
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": self.streaming,
        }
        if self.streaming:
            payload["stream_options"] = {"include_usage": True}
        # USTC's gateway suppresses reasoning_content when JSON mode is
        # requested together with thinking. Keep strict JSON in the prompt
        # and parse/validate the returned action instead, so the live panel
        # can receive the reasoning stream.
        if self.json_mode and not self.thinking:
            payload["response_format"] = {"type": "json_object"}
        # DeepSeek accepts this field on its OpenAI-compatible endpoint.
        # Thinking is shown in the live teacher panel, but stripped before
        # trajectory export; it is not an SFT target.
        payload["thinking"] = {"type": "enabled" if self.thinking else "disabled"}
        return payload

    @staticmethod
    def _delta_text(value: Any) -> str:
        """Normalize OpenAI string or content-part deltas."""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts)
        return ""

    def _emit_update(
        self,
        callback: TeacherUpdate | None,
        thinking: str,
        output: str,
        done: bool,
        last_emit: float,
    ) -> float:
        if callback is None:
            return last_emit
        now = time.monotonic()
        if not done and now - last_emit < 0.12:
            return last_emit
        callback(
            {
                "thinking": display_text(thinking),
                "output": display_text(output, limit=1000),
                "done": done,
            }
        )
        return now

    def _read_stream(self, response: Any, callback: TeacherUpdate | None) -> dict[str, Any]:
        """Consume OpenAI SSE chunks and rebuild a normal completion object."""
        reasoning_parts: list[str] = []
        output_parts: list[str] = []
        model = self.model
        finish_reason = ""
        usage: Any = None
        last_emit = 0.0
        while True:
            line = response.readline()
            if not line:
                break
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            line = str(line).strip()
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                # Some proxies split SSE data over multiple lines. Ignore
                # keep-alive/non-JSON lines; the final action is still checked.
                continue
            if not isinstance(chunk, dict):
                continue
            model = str(chunk.get("model") or model)
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            reasoning_parts.append(
                self._delta_text(delta.get("reasoning_content") or delta.get("reasoning") or delta.get("thinking"))
            )
            output_parts.append(self._delta_text(delta.get("content")))
            last_emit = self._emit_update(
                callback,
                "".join(reasoning_parts),
                "".join(output_parts),
                False,
                last_emit,
            )
        thinking = "".join(reasoning_parts)
        output = "".join(output_parts)
        self._emit_update(callback, thinking, output, True, last_emit)
        return {
            "model": model,
            "choices": [{
                "message": {"reasoning_content": thinking, "content": output},
                "finish_reason": finish_reason,
            }],
            "usage": usage,
        }

    def _request(self, payload: dict[str, Any], callback: TeacherUpdate | None = None) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key_value()}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                if payload.get("stream") and "text/event-stream" in str(response.headers.get("content-type", "")):
                    return self._read_stream(response, callback)
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:300]
            except OSError:
                pass
            raise TeacherError(f"teacher HTTP {exc.code}: {detail or exc.reason}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise TeacherError(f"teacher request failed: {exc}") from exc
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TeacherError("teacher returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise TeacherError("teacher returned a non-object response")
        return value

    def select(
        self,
        observation: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        on_update: TeacherUpdate | None = None,
    ) -> dict[str, Any]:
        legal_ids = {action_id(action) for action in legal_actions if action_id(action)}
        if not legal_ids:
            raise TeacherError("teacher called without legal actions")
        self.request_id += 1
        payload = self._payload(observation, legal_actions)
        started = time.monotonic()
        last_error: TeacherError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._request(payload, on_update)
                choices = response.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise TeacherError("teacher response contains no choice")
                message = choices[0].get("message")
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str):
                    raise TeacherError("teacher response contains no text content")
                selected = parse_action_id(content, legal_ids)
                reasoning = ""
                if isinstance(message, dict):
                    reasoning = display_text(
                        message.get("reasoning_content")
                        or message.get("reasoning")
                        or message.get("thinking")
                    )
                if not reasoning:
                    reasoning = display_text(choices[0].get("reasoning_content"))
                usage = response.get("usage")
                self.last_error = None
                return {
                    "request_id": self.request_id,
                    "action_id": selected,
                    "model_id": str(response.get("model") or self.model),
                    "thinking_enabled": self.thinking,
                    "prompt_version": PROMPT_VERSION,
                    "latency_ms": round((time.monotonic() - started) * 1000),
                    "attempt": attempt + 1,
                    "finish_reason": str(choices[0].get("finish_reason") or ""),
                    "output": display_text(content, limit=1000),
                    "thinking": reasoning,
                    "usage": {
                        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
                        "completion_tokens": _usage_value(usage, "completion_tokens"),
                        "total_tokens": _usage_value(usage, "total_tokens"),
                    },
                }
            except TeacherError as exc:
                last_error = exc
        assert last_error is not None
        self.last_error = str(last_error)
        raise last_error
