"""Auditable single-action baseline client for a persistent vLLM service.

This module has no torch/transformers dependency and never truncates prompts.
Generated actions require a subsequent fresh CommunicationMod observation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

from sts_agent.config import load_agent_config
from sts_agent.core.action_aliases import model_action_alias
from sts_agent.evaluation.progress import METRIC_VERSION, ProgressTracker, aggregate_summaries
from sts_agent.paths import PROJECT_ROOT
from sts_agent.policy.formats import require_prompt_format
from sts_agent.policy.local_qwen import build_scoring_prompt
from sts_agent.policy.ssh_tunnel import SshTunnel


def fingerprint(observation: dict, actions: list[dict], run_id: str | None) -> str:
    raw = json.dumps([run_id, observation, actions], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ApiPolicyClient:
    runtime = "remote_api"

    def __init__(self, root: Path, config: dict[str, Any]):
        self.root, self.config = root, config
        policy = config["policy"]
        self.settings = policy["api"]
        self.prompt_format = require_prompt_format(self.settings["prompt_format"])
        self.model = policy["model"]["base"]
        self.screens = tuple(policy["request"]["screens"])
        self.enabled = policy["enabled"]
        session = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.output_dir = root / self.settings["output_dir"] / session
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "requests.jsonl"
        self.base_url = f"http://127.0.0.1:{self.settings['local_port']}"
        self.opener = request.build_opener(request.ProxyHandler({}))
        remote = policy["remote"]
        self.tunnel = SshTunnel(remote["ssh_host"], self.settings["local_port"], self.settings["remote_port"],
                                remote["connect_timeout_seconds"], self.output_dir / "ssh.log") if self.settings["ssh_tunnel"] else None
        self.ready = False
        self.pending: tuple[str, dict] | None = None
        self.request_id = 0
        self.run_summaries: dict[str, dict] = {}
        self.progress_trackers: dict[str, ProgressTracker] = {}
        try:
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            code_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                                           text=True, timeout=5, check=True, creationflags=flags).stdout.strip()
            dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                                       text=True, timeout=5, check=True, creationflags=flags).stdout.strip())
        except (OSError, subprocess.SubprocessError):
            code_revision, dirty = None, None
        self.manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(), "mode": "baseline_inference",
            "actor": "model", "model": self.model, "requested_revision": policy["model"]["revision"],
            "adapter": None, "code_revision": code_revision, "dirty_worktree": dirty,
            "settings": self.settings, "prompt_builder": "build_scoring_prompt",
            "prompt_code_sha256": hashlib.sha256(Path(__file__).with_name("local_qwen.py").read_bytes()).hexdigest(),
            "human_training_data": False,
            "evaluation_metric": METRIC_VERSION,
        }
        self._save_manifest()

    def _save_manifest(self) -> None:
        (self.output_dir / "manifest.json").write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _log(self, event: str, **data: Any) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, "at": datetime.now(timezone.utc).isoformat(), **data}, ensure_ascii=False) + "\n")

    def _http(self, path: str, body: dict | None = None) -> dict:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        req = request.Request(self.base_url + path, data=payload, headers={"Content-Type": "application/json"})
        for attempt in range(self.settings["max_retries"] + 1):
            try:
                with self.opener.open(req, timeout=self.settings["timeout_seconds"]) as response:
                    value = json.loads(response.read().decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("API response must be an object")
                return value
            except (error.URLError, TimeoutError, ConnectionError) as exc:
                retryable = not isinstance(exc, error.HTTPError) or exc.code in {429, 500, 502, 503, 504}
                detail = exc.read().decode("utf-8", errors="replace") if isinstance(exc, error.HTTPError) else str(exc)
                self._log("transport_error", path=path, attempt=attempt, error=detail, retryable=retryable)
                if not retryable or attempt == self.settings["max_retries"]:
                    raise RuntimeError(f"API request failed: {detail}") from exc
                time.sleep(self.settings["retry_backoff_seconds"] * (attempt + 1))
        raise AssertionError("unreachable")

    def connect(self) -> None:
        if self.tunnel is not None:
            self.tunnel.start()
        if not self.ready:
            models = self._http("/v1/models")
            data = models.get("data")
            if not isinstance(data, list):
                raise ValueError("/v1/models did not return a model list")
            match = next((m for m in data if isinstance(m, dict) and m.get("id") == self.model), None)
            if match is None:
                raise RuntimeError(f"service does not expose configured model {self.model}")
            self.manifest["server_model_metadata"] = match
            root = str(match.get("root") or "")
            resolved = re.search(r"(?:^|/)snapshots/([0-9a-f]{40})(?:/|$)", root)
            self.manifest["resolved_model_revision"] = resolved.group(1) if resolved else None
            expected = self.config["policy"]["model"]["revision"]
            if re.fullmatch(r"[0-9a-f]{40}", expected) and (resolved is None or resolved.group(1) != expected):
                raise ValueError("service snapshot revision does not match the configured pinned SHA")
            server_limit = match.get("max_model_len")
            if isinstance(server_limit, int) and self.settings["serving"]["max_model_len"] > server_limit:
                raise ValueError(f"configured context exceeds server max_model_len={server_limit}")
            self._save_manifest()
            self.ready = True

    def status_decision(self) -> dict:
        return {"mode": "baseline_inference", "model_id": self.model, "candidate_source": "remote_api",
                "model_action": None, "model_candidates": [], "assist_status": "就绪" if self.ready else "待连接",
                "assist_reason": "远端服务需独立启动；不加载本机模型", "executed_action": None, "source": None}

    def requesting_decision(self) -> dict:
        return {**self.status_decision(), "assist_status": "请求中", "assist_reason": "正在请求单一动作（非思考模式）"}

    def prewarm(self) -> dict:
        # Do not block the game's mod-loading handshake on SSH/model startup.
        return self.status_decision()

    def rank(self, observation: dict, actions: list[dict], run_id: str | None = None, run_config: dict | None = None) -> dict:
        if not actions:
            return self.status_decision()
        state_key = fingerprint(observation, actions, run_id)
        if self.pending is not None:
            previous_key, previous = self.pending
            self.pending = None
            if previous_key == state_key:
                self._log("action_revalidated", request_id=previous["api"]["request_id"], state_key=state_key)
                return {**previous, "assist_status": "就绪", "api": {**previous["api"], "revalidated": True}}
            self._log("stale_action_discarded", old_state_key=previous_key, new_state_key=state_key,
                      request_id=previous["api"]["request_id"])
        aliases = {f"A{i}": action for i, action in enumerate(actions, 1)}
        model_actions = [model_action_alias(action, alias) for alias, action in aliases.items()]
        prompt = build_scoring_prompt(observation, model_actions, self.settings["prompt_template"],
                                      prompt_format=self.prompt_format)
        messages = [{"role": "user", "content": prompt}]
        template = {"enable_thinking": self.settings["thinking"]}
        self.request_id += 1
        req_id = self.request_id
        started = time.monotonic()
        body = {"model": self.model, "messages": messages, "temperature": self.settings["temperature"],
                "max_tokens": self.settings["max_output_tokens"], "stream": False, "chat_template_kwargs": template}
        self._log("request", request_id=req_id, run_id=run_id, run_config=run_config, state_key=state_key,
                  prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(), request=body, aliases=aliases)
        try:
            self.connect()
            tokenized = self._http("/tokenize", {"model": self.model, "messages": messages,
                                   "add_generation_prompt": True, "chat_template_kwargs": template})
            count = tokenized.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("/tokenize did not return a valid token count")
            limit = self.settings["serving"]["max_model_len"]
            if count + self.settings["max_output_tokens"] > limit:
                raise ValueError(f"context overflow: {count} input + {self.settings['max_output_tokens']} output > {limit}; no truncation")
            response = self._http("/v1/chat/completions", body)
            latency = round((time.monotonic() - started) * 1000)
            self._log("response", request_id=req_id, response=response, input_tokens=count, latency_ms=latency)
            choices = response.get("choices")
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("expected exactly one model completion")
            choice = choices[0]
            if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
                raise ValueError("malformed model completion")
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"incomplete completion: finish_reason={choice.get('finish_reason')}")
            content = choice.get("message", {}).get("content")
            if not isinstance(content, str) or content.strip() not in aliases:
                raise ValueError("model must output exactly one current legal alias, such as A1")
            alias = content.strip()
            action = aliases[alias]
            decision = {**self.status_decision(), "model_action": action["id"],
                        "model_candidates": [{"action_id": action["id"], "label": action.get("label"), "rank": 1}],
                        "assist_status": "等待状态复核", "assist_reason": None,
                        "api": {"request_id": req_id, "state_key": state_key, "alias": alias, "revalidated": False,
                                "latency_ms": latency, "input_tokens": count, "usage": response.get("usage"),
                                "output": content, "thinking": choice.get("message", {}).get("reasoning_content")}}
            self.pending = (state_key, decision)
            return decision
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            self._log("decision_error", request_id=req_id, error=str(exc), latency_ms=round((time.monotonic() - started) * 1000))
            return {**self.status_decision(), "assist_status": "暂停，需人工处理", "assist_reason": str(exc)}

    def record_result(self, record: dict) -> None:
        self._log("execution_result", run_id=record.get("run_id"), step_id=record.get("step_id"),
                  action=record.get("action"), accepted=record.get("accepted"), error=record.get("error"),
                  terminal=record.get("terminal"), decision=record.get("decision"))
        run_id = record.get("run_id")
        if not run_id:
            return
        summary = self._run_summary(run_id, record.get("run_config"))
        if record.get("accepted"):
            summary["accepted_actions"] += 1
            source = (record.get("decision") or {}).get("source")
            key = "model_actions" if source == "baseline_model" else "environment_actions" if source == "environment_auto" else "human_actions"
            summary[key] += 1
        else:
            summary["rejected_actions"] += 1
        tracker = self.progress_trackers[run_id]
        was_terminal = tracker.terminal
        pre_state = record.get("pre_state") or {}
        tracker.observe(pre_state)
        if tracker.terminal and not was_terminal:
            summary["terminal_state"] = pre_state
        self.record_observation(run_id, record.get("post_state") or {}, record.get("run_config"))

    def _run_summary(self, run_id: str, run_config: dict | None) -> dict:
        summary = self.run_summaries.setdefault(run_id, {"run_id": run_id, "accepted_actions": 0,
                "model_actions": 0, "environment_actions": 0, "human_actions": 0, "rejected_actions": 0,
                "terminal": False, "won": None, "beat_heart": None})
        self.progress_trackers.setdefault(run_id, ProgressTracker())
        if run_config is not None:
            summary["run_config"] = run_config.copy()
        return summary

    def record_observation(self, run_id: str, state: dict, run_config: dict | None = None) -> None:
        """Score natural ready observations, including terminals with no pending action."""
        summary = self._run_summary(run_id, run_config)
        tracker = self.progress_trackers[run_id]
        was_terminal = tracker.terminal
        tracker.observe(state)
        if tracker.terminal:
            summary.update(terminal=True, won=tracker.won, beat_heart=tracker.beat_heart)
            if not was_terminal:
                summary["terminal_state"] = state
        evaluation = tracker.result()
        evaluation["autonomous_eligible"] = (tracker.score_final and evaluation["progress_score"] is not None
                and not summary["human_actions"] and not (summary.get("run_config") or {}).get("partial_run"))
        summary["evaluation"] = evaluation
        (self.output_dir / "summary.json").write_text(json.dumps(self.run_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.output_dir / "evaluation.json").write_text(json.dumps(aggregate_summaries(self.run_summaries),
                ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self) -> None:
        self.pending = None
        if self.tunnel is not None:
            self.tunnel.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the remote baseline API without starting a game")
    parser.add_argument("--config", default="config/agent.json")
    parser.add_argument("--replay", type=Path, help="offline inference on a sts-agent human trajectory; never sends game commands")
    parser.add_argument("--max-states", type=int, default=20)
    args = parser.parse_args()
    if args.max_states < 1:
        parser.error("--max-states must be positive")
    client = ApiPolicyClient(PROJECT_ROOT, load_agent_config(config_file=args.config, mode="baseline_inference"))
    try:
        if args.replay is not None:
            from sts_agent.runtime.controller import route_policy_observation

            evaluated, correct, errors = 0, 0, 0
            with args.replay.open(encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    observation, actions = record.get("pre_state"), record.get("legal_actions")
                    if record.get("accepted") is not True or not isinstance(observation, dict) or not actions:
                        continue
                    if evaluated >= args.max_states:
                        break
                    # Each entry is an independent historical decision, not a
                    # state confirmation of the preceding replay entry.
                    client.pending = None
                    decision = client.rank(route_policy_observation(observation), actions,
                                           run_id=record.get("run_id"), run_config=record.get("run_config"))
                    expected = (record.get("action") or {}).get("id")
                    actual = decision.get("model_action")
                    evaluated += 1
                    correct += int(actual is not None and actual == expected)
                    errors += int(actual is None)
                    client._log("replay_comparison", step_id=record.get("step_id"), expected=expected,
                                actual=actual, matched=actual == expected, decision=decision)
                    print(json.dumps({"step_id": record.get("step_id"), "human": expected, "model": actual,
                                      "error": decision.get("assist_reason") if actual is None else None}, ensure_ascii=False))
                    if actual is None:
                        break
            summary = {"evaluated": evaluated, "human_action_matches": correct, "errors": errors,
                       "note": "Human agreement is diagnostic, not an optimality or win-rate metric.", "artifacts": str(client.output_dir)}
            client._log("replay_summary", **summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if evaluated > 0 and errors == 0 else 2
        decision = client.rank({"screen_type": "MAP", "in_game": True}, [{"id": "CHECK", "label": "选择唯一合法动作"}], run_id="preflight")
        print(json.dumps({"decision": decision, "artifacts": str(client.output_dir)}, ensure_ascii=False, indent=2))
        return 0 if decision.get("model_action") == "CHECK" else 2
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
