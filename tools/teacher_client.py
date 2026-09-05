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
from experience_knowledge import visible_entity_references


class TeacherError(RuntimeError):
    """A recoverable remote-teacher error that should fall back to a human."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


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
        raise TeacherError(f"cannot read teacher.api_key_file: {path}", retryable=False) from exc
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
    raise TeacherError(f"teacher.api_key_file contains no key: {path}", retryable=False)


def build_prompt(
    observation: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    max_input_chars: int = 32000,
    allow_knowledge_request: bool = False,
    knowledge_request_style: str = "encouraged",
) -> str:
    choose_instruction = (
        "Choose the best next response for long-term win probability: normally a supplied legal action, or a knowledge_request when a currently public entity needs clarification."
        if allow_knowledge_request
        else "Choose one supplied legal action that maximizes long-term win probability."
    )
    response_instruction = (
        'Return exactly one JSON object: either {"action_id":"A1"} for a legal action, or {"knowledge_request":{"entity_type":"card|relic|potion|monster|event|power","entity_id":"...","question":"..."}} for one currently public entity. Never include both, prose, markdown, or an action outside the list.'
        if allow_knowledge_request
        else 'Return exactly one JSON object: {"action_id":"A1"}. Do not include prose, markdown, or an action outside the list.'
    )
    lines = [
        choose_instruction,
        "The observation contains only public game information. recent_actions, event_history, and selection_context are context; the current legal actions are authoritative.",
        response_instruction,
        "Observation:",
        compact_json(observation, limit=max_input_chars),
    ]
    if allow_knowledge_request:
        lines.append(
            "知识请求白名单（第一列是稳定 entity_id，第三列名称仅用于识别；ID 忽略大小写、空格和下划线）："
        )
        references = visible_entity_references(observation)
        if references:
            lines.extend(
                f"- {item['entity_type']} | {item['entity_id']} | {item.get('name') or '-'}"
                for item in references
            )
        else:
            lines.append("- (empty: do not return knowledge_request)")
    else:
        lines.append("- (knowledge requests disabled)")
    lines.extend([
        "Legal actions:",
    ])
    for action in legal_actions:
        lines.append(f"- {action_id(action)} | {action_label(action)}")
    if allow_knowledge_request:
        request_instruction = (
            "If anything about a currently public entity's mechanics, interactions, risks, or strategic meaning is unclear, prefer a knowledge_request instead of guessing; you need not first prove that no reliable action is possible."
            if str(knowledge_request_style).strip().casefold() == "encouraged"
            else "If anything about a currently public entity's mechanics, interactions, risks, or strategic meaning is unclear, you may return a knowledge_request instead of an action."
        )
        lines.insert(2, request_instruction + " 只返回包含 entity_type、entity_id、question 的 knowledge_request，不能同时返回 action_id。entity_type 和 entity_id 必须从下面的白名单中选择；不要把事件选项、选项标签或自由填写的名称用作 entity_id。")
    return "\n".join(lines)


def parse_action_id(content: str, legal_ids: set[str]) -> str:
    """Accept strict JSON first, then a single exact legal alias as fallback."""
    payload = _extract_json_object(content)
    if payload is not None:
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


def parse_action_response(content: str, legal_ids: set[str]) -> tuple[str | None, dict[str, str] | None]:
    """Parse an action plus an optional entity-specific knowledge request.

    A knowledge request is deliberately out-of-band: the controller must
    obtain human input and approve the resulting experience before asking for
    an executable action again.
    """
    payload = _extract_json_object(content)
    request = payload.get("knowledge_request") if isinstance(payload, dict) else None
    normalized_request: dict[str, str] | None = None
    if isinstance(request, dict):
        entity_type = str(request.get("entity_type") or request.get("type") or "").strip().casefold()
        entity_id = str(request.get("entity_id") or request.get("id") or "").strip()
        question = str(request.get("question") or "").strip()
        if entity_type and entity_id and question and len(question) <= 2000:
            normalized_request = {"entity_type": entity_type, "entity_id": entity_id, "question": question}
        else:
            raise TeacherError("teacher knowledge_request is missing entity_type, entity_id, or question")
    candidate = str(payload.get("action_id", "")) if isinstance(payload, dict) else ""
    if candidate:
        if candidate not in legal_ids:
            raise TeacherError("teacher response selected an action outside the legal set")
        if normalized_request is not None:
            raise TeacherError("teacher response must contain an action or a knowledge_request, not both")
        return candidate, normalized_request
    if normalized_request is not None:
        return None, normalized_request
    return parse_action_id(content, legal_ids), None


def _extract_json_object(content: str) -> dict[str, Any] | None:
    """Decode the first complete JSON object, including nested request data."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(content):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


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
        self.retry_backoff_seconds = float(config.get("retry_backoff_seconds", 2.0))
        self.retry_max_backoff_seconds = float(config.get("retry_max_backoff_seconds", 8.0))
        self.temperature = float(config["temperature"])
        self.max_tokens = int(config["max_tokens"])
        self.max_input_chars = int(config["max_input_chars"])
        self.thinking = bool(config["thinking"])
        self.json_mode = bool(config["json_mode"])
        self.streaming = bool(config["streaming"])
        self.screens = tuple(item.strip().upper() for item in config["screens"])
        self.allow_knowledge_request = False
        self.knowledge_request_style = "encouraged"
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

    def _payload(
        self,
        observation: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        allow_knowledge_request = bool(getattr(self, "allow_knowledge_request", False))
        knowledge_request_style = str(getattr(self, "knowledge_request_style", "encouraged"))
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt or build_system_prompt(
                        observation,
                        allow_knowledge_request=allow_knowledge_request,
                        knowledge_request_style=knowledge_request_style,
                    ),
                },
                {
                    "role": "user",
                    "content": build_prompt(
                        observation,
                        legal_actions,
                        self.max_input_chars,
                        allow_knowledge_request,
                        knowledge_request_style,
                    ),
                },
            ],
            "temperature": self.temperature if temperature is None else float(temperature),
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
            retryable = exc.code in {408, 409, 425, 429} or exc.code >= 500
            raise TeacherError(f"teacher HTTP {exc.code}: {detail or exc.reason}", retryable=retryable) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise TeacherError(f"teacher request failed: {exc}") from exc
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TeacherError("teacher returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise TeacherError("teacher returned a non-object response")
        return value

    def _retry_delay(self, retry_index: int) -> float:
        """Return bounded exponential backoff before the next HTTP attempt."""

        base = float(getattr(self, "retry_backoff_seconds", 2.0))
        maximum = float(getattr(self, "retry_max_backoff_seconds", 8.0))
        if base <= 0.0:
            return 0.0
        return min(
            maximum,
            base * (2.0 ** max(0, retry_index)),
        )

    def _wait_before_retry(
        self,
        error: TeacherError,
        retry_index: int,
        callback: TeacherUpdate | None,
    ) -> None:
        delay = self._retry_delay(retry_index)
        if callback is not None:
            callback(
                {
                    "thinking": "",
                    "output": "",
                    "done": False,
                    "retrying": True,
                    "retry_attempt": retry_index + 1,
                    "retry_delay_seconds": delay,
                    "retry_reason": str(error),
                }
            )
        # Wait before every retry, including parse/transport failures, so a
        # transient response or rate-limit error is not immediately retried.
        if delay > 0.0:
            time.sleep(delay)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        prompt_version: str,
        thinking: bool | None = None,
        json_mode: bool | None = None,
        streaming: bool | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_update: TeacherUpdate | None = None,
    ) -> dict[str, Any]:
        """Run a bounded generic completion through the configured teacher.

        Action selection remains the primary public API of this class.  This
        narrow method lets post-hoc evaluators share the same credential,
        retry, timeout, and OpenAI-compatible transport without inheriting the
        action-label parser or storing reasoning traces.
        """

        if not self.enabled:
            raise TeacherError("teacher is disabled")
        if not messages or not all(isinstance(item, dict) for item in messages):
            raise TeacherError("teacher completion requires non-empty messages")
        use_thinking = self.thinking if thinking is None else bool(thinking)
        use_json_mode = self.json_mode if json_mode is None else bool(json_mode)
        use_streaming = self.streaming if streaming is None else bool(streaming)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else float(temperature),
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
            "stream": use_streaming,
            "thinking": {"type": "enabled" if use_thinking else "disabled"},
        }
        if use_streaming:
            payload["stream_options"] = {"include_usage": True}
        if use_json_mode and not use_thinking:
            payload["response_format"] = {"type": "json_object"}

        self.request_id += 1
        started = time.monotonic()
        last_error: TeacherError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._request(payload, on_update)
                choices = response.get("choices")
                if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                    raise TeacherError("teacher response contains no choice")
                choice = choices[0]
                message = choice.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, str) or not content.strip():
                    raise TeacherError("teacher response contains no text content")
                usage = response.get("usage")
                self.last_error = None
                return {
                    "request_id": self.request_id,
                    "model_id": str(response.get("model") or self.model),
                    "prompt_version": prompt_version,
                    "latency_ms": round((time.monotonic() - started) * 1000),
                    "attempt": attempt + 1,
                    "finish_reason": str(choice.get("finish_reason") or ""),
                    "output": display_text(content, limit=16000),
                    "usage": {
                        "prompt_tokens": _usage_value(usage, "prompt_tokens"),
                        "completion_tokens": _usage_value(usage, "completion_tokens"),
                        "total_tokens": _usage_value(usage, "total_tokens"),
                    },
                }
            except TeacherError as exc:
                last_error = exc
                if attempt >= self.max_retries or not exc.retryable:
                    break
                self._wait_before_retry(exc, attempt, on_update)
        assert last_error is not None
        self.last_error = str(last_error)
        raise last_error

    def select(
        self,
        observation: dict[str, Any],
        legal_actions: list[dict[str, Any]],
        on_update: TeacherUpdate | None = None,
        *,
        system_prompt: str | None = None,
        prompt_version: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        legal_ids = {action_id(action) for action in legal_actions if action_id(action)}
        if not legal_ids:
            raise TeacherError("teacher called without legal actions")
        self.request_id += 1
        payload = self._payload(
            observation,
            legal_actions,
            system_prompt=system_prompt,
            temperature=temperature,
        )
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
                selected, knowledge_request = parse_action_response(content, legal_ids)
                if knowledge_request is not None and not bool(getattr(self, "allow_knowledge_request", False)):
                    raise TeacherError("teacher returned knowledge_request while knowledge assistance is disabled", retryable=False)
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
                    "knowledge_request": knowledge_request,
                    "model_id": str(response.get("model") or self.model),
                    "thinking_enabled": self.thinking,
                    "prompt_version": prompt_version or PROMPT_VERSION,
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
                if attempt >= self.max_retries or not exc.retryable:
                    break
                self._wait_before_retry(exc, attempt, on_update)
        assert last_error is not None
        self.last_error = str(last_error)
        raise last_error
