"""Supervise fixed-seed baseline runs without choosing any in-game action.

The controller remains the sole game player. This client only attaches at a
ready main menu, starts each planned seed once, and detaches after the final
terminal result has returned to that menu. A failed batch is never resumed or
silently retried; its report includes every planned seed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sts_agent.config import ConfigError, load_agent_config
from sts_agent.core.modes import MODE_FRAMEWORKS
from sts_agent.evaluation.progress import aggregate_summaries
from sts_agent.evaluation.reuse import expected_fingerprint, find_reusable_runs
from sts_agent.paths import PROJECT_ROOT
from sts_agent.runtime.batch_client import PROTOCOL_VERSION


DEFAULT_SUITE = "config/evaluation/a0_ironclad_10.json"
SEED_ALPHABET = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
LIMIT_DEFAULTS = {"connection_timeout_seconds": 30, "idle_timeout_seconds": 600,
                  "run_timeout_seconds": 3600, "max_actions_per_run": 4000}
MAX_PACKET_BYTES = 8 * 1024 * 1024


class BatchError(ValueError):
    """An unsafe, incomplete, or incompatible batch must stop immediately."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BatchError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchError(f"{path} must contain one JSON object")
    return value


def seed_to_internal(seed: str) -> int:
    """Decode a canonical StS display seed, including Java's signed long wrap."""
    if not isinstance(seed, str) or not re.fullmatch(r"[0-9A-NP-Z]+", seed):
        raise BatchError("seeds must use uppercase digits/letters, excluding O")
    if len(seed) > 1 and seed.startswith("0"):
        raise BatchError(f"seed is not canonical (leading zero): {seed}")
    value = 0
    for character in seed:
        value = value * len(SEED_ALPHABET) + SEED_ALPHABET.index(character)
    if value == 0:
        raise BatchError("zero seed is unsupported: the game displays an empty seed string")
    if value >= 1 << 64:
        raise BatchError(f"seed exceeds the game's unsigned 64-bit range: {seed}")
    return value - (1 << 64) if value >= 1 << 63 else value


def validate_suite(value: dict) -> dict:
    """Validate strictly before creating output or making a connection."""
    if not isinstance(value, dict):
        raise BatchError("suite must be an object")
    required = {"schema_version", "suite_id", "character", "ascension", "seeds", "limits", "output_dir"}
    unknown = set(value) - required - {"description", "seed_generation"}
    if unknown or required - set(value):
        raise BatchError(f"invalid suite fields: unknown={sorted(unknown)}, missing={sorted(required - set(value))}")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise BatchError("suite schema_version must be 1")
    if not isinstance(value["suite_id"], str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", value["suite_id"]):
        raise BatchError("suite_id must be a safe, non-empty filename token")
    if value["character"] != "IRONCLAD" or type(value["ascension"]) is not int or value["ascension"] != 0:
        raise BatchError("batch evaluation is fixed to IRONCLAD ascension 0")
    seeds = value["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise BatchError("seeds must be a non-empty list")
    for seed in seeds:
        seed_to_internal(seed)
    if len(set(seeds)) != len(seeds):
        raise BatchError("duplicate seeds are not allowed")
    limits = value["limits"]
    if not isinstance(limits, dict) or set(limits) != set(LIMIT_DEFAULTS):
        raise BatchError(f"limits must contain exactly {sorted(LIMIT_DEFAULTS)}")
    for key, number in limits.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
            raise BatchError(f"limits.{key} must be a positive finite number")
    if type(limits["max_actions_per_run"]) is not int:
        raise BatchError("limits.max_actions_per_run must be a positive integer")
    output = value["output_dir"]
    if not isinstance(output, str) or not output.strip() or Path(output).anchor or ".." in Path(output).parts:
        raise BatchError("output_dir must be a project-relative directory without '..'")
    if "description" in value and not isinstance(value["description"], str):
        raise BatchError("description must be a string")
    if "seed_generation" in value and not isinstance(value["seed_generation"], dict):
        raise BatchError("seed_generation must be an object")
    return copy.deepcopy(value)


def load_suite(path: Path) -> dict:
    return validate_suite(_read_json(path))


def differing_fields(actual: Any, expected: Any, prefix: str = "") -> list[str]:
    """Report precise mismatches without replacing configuration validation."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        differences = []
        for key in sorted(actual.keys() | expected.keys()):
            name = f"{prefix}.{key}" if prefix else key
            if key not in actual or key not in expected:
                differences.append(name)
            else:
                differences.extend(differing_fields(actual[key], expected[key], name))
        return differences
    return [] if actual == expected else [prefix]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def api_performance(path: Path, run_ids: set[str]) -> dict:
    """Join response/error/discard records to their run's request IDs.

    HTTP transport errors lack request_id in older logs. Calls are sequential,
    so those errors belong to the most recently logged request, never another
    run. Execution results also embed the associated decision request ID.
    """
    def records():
        if not path.exists():
            return
        try:
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise BatchError(f"non-object API log record at line {line_number}")
                    yield record
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise BatchError(f"cannot read API request log {path}: {exc}") from exc

    # Stream the source twice instead of retaining every full prompt in RAM.
    # The first pass collects only small request IDs needed for the join.
    request_ids = set()
    for record in records():
        if record.get("run_id") not in run_ids:
            continue
        request_id = record.get("request_id")
        if record.get("event") == "execution_result":
            decision = record.get("decision")
            api = decision.get("api") if isinstance(decision, dict) else None
            request_id = api.get("request_id") if isinstance(api, dict) else None
        if type(request_id) in (str, int):
            request_ids.add(request_id)
    events = {key: 0 for key in ("request", "response", "decision_error", "transport_error", "stale_action_discarded")}
    active_request = None
    latencies = []
    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage_records = input_tokens = 0
    for record in records():
        event = record.get("event")
        if event == "request":
            active_request = record.get("request_id")
        associated = record.get("request_id", active_request if event == "transport_error" else None)
        if record.get("run_id") not in run_ids and associated not in request_ids:
            continue
        if event in events:
            events[event] += 1
        if event == "response":
            latency = record.get("latency_ms")
            if type(latency) in (int, float) and math.isfinite(latency) and latency >= 0:
                latencies.append(float(latency))
            if type(record.get("input_tokens")) is int:
                input_tokens += record["input_tokens"]
            response = record.get("response")
            usage = response.get("usage") if isinstance(response, dict) else None
            if isinstance(usage, dict) and usage:
                usage_records += 1
                for key in tokens:
                    if type(usage.get(key)) is int and usage[key] >= 0:
                        tokens[key] += usage[key]
    return {"calls": events["request"], "responses": events["response"],
            "errors": events["decision_error"], "transport_errors": events["transport_error"],
            "discards": events["stale_action_discarded"],
            "token_usage": tokens if usage_records else None, "usage_records": usage_records,
            "input_tokens": input_tokens,
            "latency_ms_p50": percentile(latencies, .5), "latency_ms_p95": percentile(latencies, .95),
            "request_ids": sorted(request_ids, key=str), "source": str(path.resolve())}


class BatchReport:
    """Write only the unique batch directory; keep controller artifacts intact."""

    def __init__(self, root: Path, suite: dict, suite_path: Path, config_path: Path,
                 host: str, port: int, clock: Callable[[], float] = time.monotonic,
                 expected_policy: dict | None = None, reused: dict[str, dict] | None = None):
        self.clock = clock
        self.started = clock()
        self.suite = copy.deepcopy(suite)
        self.expected_policy = copy.deepcopy(expected_policy)
        identifier = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.output_dir = (root / suite["output_dir"] / f"{suite['suite_id']}-{identifier}").resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.experiment_dir: Path | None = None
        self.model_manifest: dict | None = None
        self.completed_performance: dict | None = None
        self.manifest = {"schema_version": 1, "protocol_version": PROTOCOL_VERSION,
                         "suite_id": suite["suite_id"], "created_at": utc_now(),
                         "suite_path": str(suite_path.resolve()), "config_path": str(config_path.resolve()),
                         "controller": {"host": host, "port": port}, "model_manifest": None,
                         "experiment_dir": None, "status": "in_progress", "failure": None,
                         "suite_sha256": hashlib.sha256(json.dumps(suite, sort_keys=True).encode()).hexdigest()}
        self.rows = []
        reused = reused or {}
        for i, seed in enumerate(suite["seeds"]):
            row = {"index": i + 1, "seed": seed, "status": "pending", "run_id": None,
                   "internal_seed": None, "attempted_at": None, "started_at": None, "finished_at": None,
                   "elapsed_seconds": None, "summary": None, "api_performance": None,
                   "artifacts": None, "failure": None}
            item = reused.get(seed)
            if item is not None:
                summary = item["summary"]
                row.update(status="completed", run_id=item.get("run_id"),
                           internal_seed=(summary.get("run_config") or {}).get("internal_seed"),
                           attempted_at=item.get("finished_at"), started_at=item.get("finished_at"),
                           finished_at=item.get("finished_at"), summary=summary,
                           api_performance=item.get("api_performance"), artifacts=item.get("artifacts"),
                           reused_from={"batch": item.get("batch"), "suite_id": item.get("suite_id"),
                                        "run_id": item.get("run_id")})
            self.rows.append(row)
        self.manifest["reused_runs"] = [{"seed": row["seed"], "run_id": row["run_id"],
                                         **(row.get("reused_from") or {})}
                                        for row in self.rows if row.get("reused_from")]
        self._write_json("suite.json", self.suite)
        self.save()

    def _write_json(self, filename: str, value: Any) -> None:
        path = self.output_dir / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)

    def freeze_experiment(self, directory: str, model: str) -> None:
        candidate = Path(directory)
        if not candidate.is_absolute():
            raise BatchError("controller experiment_dir must be absolute")
        candidate = candidate.resolve()
        if self.experiment_dir is not None and candidate != self.experiment_dir:
            raise BatchError("controller experiment directory changed during batch")
        manifest = _read_json(candidate / "manifest.json")
        # Retain what was actually received even if preflight refuses it.
        self.manifest["preflight"] = {"observed_experiment_dir": str(candidate),
                                      "observed_model_manifest": copy.deepcopy(manifest),
                                      "configuration_differences": []}
        if manifest.get("mode") != "baseline_inference" or manifest.get("model") != model or not isinstance(manifest.get("settings"), dict):
            raise BatchError("experiment manifest does not describe this baseline model/configuration")
        if self.expected_policy is not None:
            expected_model = self.expected_policy["model"]
            # Config uses a string path (empty means no adapter); the official
            # model manifest uses JSON null for that same absent adapter.
            expected = {"requested_revision": expected_model["revision"],
                        "adapter": expected_model["initial_adapter"] or None,
                        "settings": self.expected_policy["api"]}
            actual = {key: manifest.get(key) for key in expected}
            differences = differing_fields(actual, expected)
            self.manifest["preflight"]["configuration_differences"] = differences
            if differences:
                raise BatchError("controller model/settings differ from current config: " + ", ".join(differences)
                                 + "; restart game/controller before evaluating")
        if self.model_manifest is not None:
            for key in ("model", "requested_revision", "adapter", "settings", "prompt_builder", "prompt_code_sha256",
                        "model_config", "code_revision", "evaluation_metric", "resolved_model_revision"):
                if key in self.model_manifest and manifest.get(key) != self.model_manifest[key]:
                    raise BatchError(f"experiment model/configuration changed during batch: {key}")
        self.experiment_dir, self.model_manifest = candidate, copy.deepcopy(manifest)
        self.manifest.update(experiment_dir=str(candidate), model_manifest=copy.deepcopy(manifest))

    def artifacts(self, run_id: str) -> dict:
        assert self.experiment_dir is not None
        directory = self.experiment_dir
        return {"experiment_dir": str(directory), "manifest": str(directory / "manifest.json"),
                "summary": str(directory / "summary.json"), "evaluation": str(directory / "evaluation.json"),
                "requests": str(directory / "requests.jsonl"),
                "action_traces": [str(path) for path in sorted(directory.glob(f"action-trace-*{run_id}*.jsonl"))],
                "run_events": [str(path) for path in sorted(directory.glob(f"run-events-*{run_id}*.jsonl"))],
                "messages": [str(path) for path in sorted(directory.glob("messages-*.jsonl"))]}

    def save(self) -> None:
        completed = {row["run_id"]: row["summary"] for row in self.rows if row["status"] == "completed"}
        evaluation = aggregate_summaries(completed)
        reused_rows = [row for row in self.rows if row.get("reused_from")]
        evaluation.update(planned_runs=len(self.rows), completed_runs=len(completed),
                          runs_attempted=sum(row["attempted_at"] is not None for row in self.rows),
                          runs_started=sum(row["started_at"] is not None for row in self.rows),
                          not_started_runs=sum(row["started_at"] is None for row in self.rows),
                          unfinished_runs=sum(row["started_at"] is not None and row["status"] != "completed" for row in self.rows),
                          pending_runs=sum(row["status"] == "pending" for row in self.rows),
                          technical_aborts=sum(row["status"] == "technical_abort" for row in self.rows),
                          reused_runs=len(reused_rows),
                          new_runs=sum(row["status"] == "completed" for row in self.rows) - len(reused_rows),
                          batch_status=self.manifest["status"],
                          elapsed_seconds=round(self.clock() - self.started, 3))
        # Full prompts make requests.jsonl large. Never rescan it for every
        # progress packet; update the cached aggregate only at run boundaries.
        if self.completed_performance is not None:
            evaluation["api_performance"] = self.completed_performance
            if reused_rows:
                evaluation["api_performance"]["scope"] = "new runs in this batch; reused rows keep their own api_performance"
        elif reused_rows:
            evaluation["api_performance"] = {"scope": "reused rows only; see each row in results.json"}
        self._write_json("manifest.json", self.manifest)
        self._write_json("results.json", {"schema_version": 1, "suite_id": self.suite["suite_id"],
                                          "status": self.manifest["status"], "results": self.rows})
        self._write_json("evaluation.json", evaluation)
        fields = ["index", "seed", "status", "run_id", "internal_seed", "elapsed_seconds", "won", "beat_heart",
                  "progress_score", "score_final", "highest_floor", "highest_act", "accepted_actions", "model_actions",
                  "environment_actions", "human_actions", "rejected_actions", "api_calls", "api_errors",
                  "api_discards", "prompt_tokens", "completion_tokens", "latency_ms_p50", "latency_ms_p95", "failure"]
        temporary = self.output_dir / "results.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                summary, metrics, performance = row["summary"] or {}, (row["summary"] or {}).get("evaluation") or {}, row["api_performance"] or {}
                flat = {key: row.get(key) for key in fields}
                flat.update({key: summary.get(key) for key in ("won", "beat_heart", "accepted_actions", "model_actions", "environment_actions", "human_actions", "rejected_actions")})
                flat.update({key: metrics.get(key) for key in ("progress_score", "score_final", "highest_floor", "highest_act")})
                flat.update(api_calls=performance.get("calls"), api_errors=performance.get("errors"), api_discards=performance.get("discards"))
                flat.update({key: performance.get(key) for key in ("latency_ms_p50", "latency_ms_p95")})
                flat.update({key: (performance.get("token_usage") or {}).get(key) for key in ("prompt_tokens", "completion_tokens")})
                writer.writerow(flat)
        temporary.replace(self.output_dir / "results.csv")

    def abort(self, index: int, reason: str, elapsed: float | None) -> None:
        self.manifest.update(status="technical_abort", failure=reason, finished_at=utc_now())
        if index < len(self.rows) and self.rows[index]["status"] != "completed":
            row = self.rows[index]
            row.update(status="technical_abort", failure=reason, finished_at=utc_now(), elapsed_seconds=elapsed)
            if row["run_id"] and self.experiment_dir is not None:
                row["artifacts"] = self.artifacts(row["run_id"])
                try:
                    row["api_performance"] = api_performance(self.experiment_dir / "requests.jsonl", {row["run_id"]})
                except BatchError as exc:
                    row["api_performance"] = {"unavailable": str(exc)}
        self.save()


class BatchStateMachine:
    """Explicit menu/run phases prevent stale or duplicate START commands."""

    def __init__(self, suite: dict, report: BatchReport, expected_model: str,
                 clock: Callable[[], float] = time.monotonic):
        self.suite, self.report, self.expected_model, self.clock = suite, report, expected_model, clock
        self.phase = "initial"
        self.order = [i for i, row in enumerate(report.rows) if row.get("status") != "completed"]
        self.position = 0
        self.run_id: str | None = None
        self.previous_run_id: str | None = None
        self.started: float | None = None
        self.last_packet = clock()
        self.seen_run_ids: set[str] = set()

    @property
    def index(self) -> int:
        """Row index of the current pending seed (absolute in the full suite)."""

        if not self.order:
            raise BatchError("batch has no pending runs")
        return self.order[min(self.position, len(self.order) - 1)]

    @property
    def done(self) -> bool:
        return self.phase == "complete"

    def check_timeouts(self) -> None:
        now = self.clock()
        if self.started is not None and now - self.started >= self.suite["limits"]["run_timeout_seconds"]:
            raise BatchError("run timeout while waiting for terminal result and main menu")
        if now - self.last_packet >= self.suite["limits"]["idle_timeout_seconds"]:
            raise BatchError("idle timeout while waiting for controller progress")

    def read_timeout(self) -> float:
        self.check_timeouts()
        now = self.clock()
        remaining = self.suite["limits"]["idle_timeout_seconds"] - (now - self.last_packet)
        if self.started is not None:
            remaining = min(remaining, self.suite["limits"]["run_timeout_seconds"] - (now - self.started))
        return max(.001, remaining)

    def _metadata(self, packet: dict) -> dict:
        baseline = packet.get("baseline")
        if not isinstance(baseline, dict):
            raise BatchError("missing structured baseline metadata; update controller before evaluating")
        if type(baseline.get("protocol_version")) is not int or baseline["protocol_version"] != PROTOCOL_VERSION:
            raise BatchError("baseline protocol version mismatch")
        if baseline.get("mode") != "baseline_inference" or baseline.get("model") != self.expected_model:
            raise BatchError("controller is not running the configured remote baseline model")
        for key in ("can_start_run", "ready_for_command", "batch_attached"):
            if type(baseline.get(key)) is not bool:
                raise BatchError(f"invalid baseline boolean: {key}")
        if type(packet.get("in_game")) is not bool:
            raise BatchError("baseline packet must explicitly declare in_game")
        if not isinstance(baseline.get("experiment_dir"), str) or not Path(baseline["experiment_dir"]).is_absolute():
            raise BatchError("baseline experiment_dir must be absolute")
        if baseline.get("run_id") is not None and (not isinstance(baseline["run_id"], str) or not baseline["run_id"]):
            raise BatchError("invalid baseline run_id")
        summary = baseline.get("run_summary")
        if summary is not None and (not isinstance(summary, dict) or summary.get("run_id") != baseline.get("run_id")):
            raise BatchError("run summary does not match controller run_id")
        if self.report.experiment_dir is not None and Path(baseline["experiment_dir"]).resolve() != self.report.experiment_dir:
            raise BatchError("controller experiment directory changed during batch")
        if packet.get("error"):
            raise BatchError(f"controller error: {packet['error']}")
        if baseline.get("controller_error"):
            raise BatchError(f"controller error: {baseline['controller_error']}")
        decision = packet.get("decision")
        if isinstance(decision, dict) and (decision.get("error") or "暂停" in str(decision.get("assist_status", ""))):
            raise BatchError(f"controller paused: {decision.get('assist_reason') or decision.get('error') or decision.get('assist_status')}")
        return baseline

    def _summary(self, baseline: dict) -> dict | None:
        summary = baseline.get("run_summary")
        if summary is None:
            return None
        config = summary.get("run_config")
        if not isinstance(config, dict):
            raise BatchError("run summary missing run_config")
        if config.get("partial_run") is True or summary.get("human_actions", 0) != 0:
            raise BatchError("partial or human-intervened runs cannot enter a baseline batch")
        if config.get("character") != "IRONCLAD" or type(config.get("ascension")) is not int or config["ascension"] != 0:
            raise BatchError("run character/ascension does not match suite")
        expected = self.suite["seeds"][self.index]
        if config.get("seed") != expected or config.get("seed_source") != "requested":
            raise BatchError(f"run seed does not match requested seed {expected}")
        internal = config.get("internal_seed")
        if internal is not None:
            if isinstance(internal, bool) or not isinstance(internal, (int, str)):
                raise BatchError("invalid internal game seed")
            try:
                parsed = int(internal)
            except ValueError as exc:
                raise BatchError("invalid internal game seed") from exc
            expected_internal = seed_to_internal(expected)
            if parsed not in {expected_internal, expected_internal % (1 << 64)}:
                raise BatchError(f"internal game seed does not match requested seed {expected}")
        for key in ("accepted_actions", "model_actions", "environment_actions", "human_actions", "rejected_actions"):
            if type(summary.get(key)) is not int or summary[key] < 0:
                raise BatchError(f"invalid run action count: {key}")
        action_limit = self.suite["limits"]["max_actions_per_run"]
        if summary["accepted_actions"] > action_limit or (summary["accepted_actions"] == action_limit and not summary.get("terminal")):
            raise BatchError("maximum actions per run exceeded")
        if type(summary.get("terminal")) is not bool or not isinstance(summary.get("evaluation"), dict):
            raise BatchError("invalid terminal/evaluation metadata")
        if summary["terminal"]:
            evaluation = summary["evaluation"]
            score = evaluation.get("progress_score")
            if internal is None or evaluation.get("autonomous_eligible") is not True or evaluation.get("score_final") is not True:
                raise BatchError("terminal run is not a final autonomous evaluation")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 100:
                raise BatchError("invalid final progress score")
            if type(summary.get("won")) is not bool or type(summary.get("beat_heart")) is not bool:
                raise BatchError("terminal outcome is unknown")
            bosses = evaluation.get("boss_acts_defeated")
            if not isinstance(bosses, list) or any(type(act) is not int or act not in (1, 2, 3) for act in bosses) or bosses != sorted(set(bosses)):
                raise BatchError("invalid final boss completion metadata")
            if summary["won"] and (3 not in bosses or not math.isclose(score, 100)):
                raise BatchError("victory disagrees with final progress score")
            if summary["beat_heart"] and not summary["won"]:
                raise BatchError("heart victory disagrees with terminal outcome")
        return summary

    def _start(self, baseline: dict) -> list[dict]:
        if not baseline["can_start_run"] or not baseline["ready_for_command"] or not baseline["batch_attached"]:
            raise BatchError("main menu is not ready to start an attached batch run")
        self.previous_run_id = baseline["run_id"]
        self.run_id = None
        self.started = self.clock()
        self.phase = "await_run"
        self.report.rows[self.index]["attempted_at"] = utc_now()
        self.report.save()
        return [{"type": "input", "text": f"s IRONCLAD 0 {self.suite['seeds'][self.index]}"}]

    def consume(self, packet: dict) -> list[dict]:
        self.check_timeouts()
        if not isinstance(packet, dict) or not isinstance(packet.get("type"), str):
            raise BatchError("controller packet must be an object with a type")
        packet_type = packet["type"]
        if packet_type == "hello":
            if self.phase != "initial":
                raise BatchError("unexpected reconnect greeting")
            self.last_packet = self.clock()
            return []
        if packet_type not in {"view", "model_update"}:
            raise BatchError(f"unexpected controller packet: {packet_type}")
        baseline = self._metadata(packet)
        self.last_packet = self.clock()
        if packet_type == "view" and packet["in_game"]:
            raise BatchError("controller requested input during a run; batch paused")
        if self.phase == "initial":
            if packet_type != "view" or packet["in_game"] or not baseline["can_start_run"] or not baseline["ready_for_command"]:
                raise BatchError("batch must connect at a ready main menu; existing games are refused")
            if baseline["batch_attached"]:
                raise BatchError("controller already has a batch attached")
            self.report.freeze_experiment(baseline["experiment_dir"], baseline["model"])
            self.previous_run_id = baseline["run_id"]
            self.phase = "attaching"
            self.report.save()
            return [{"type": "baseline_batch_attach", "protocol_version": PROTOCOL_VERSION}]
        if self.phase == "attaching":
            if packet_type != "view" or packet["in_game"]:
                raise BatchError("attach acknowledgement must be a main-menu view")
            if not baseline["batch_attached"]:
                return []  # A duplicate pre-attach view cannot send another attach/START.
            return self._start(baseline)
        if self.phase == "detaching":
            if packet_type != "view" or packet["in_game"] or not baseline["can_start_run"]:
                raise BatchError("detach acknowledgement must be a main-menu view")
            if baseline["batch_attached"]:
                return []
            self.phase = "complete"
            self.report.manifest.update(status="completed", finished_at=utc_now())
            self.report.save()
            return []
        if self.phase == "complete":
            return []
        if not baseline["batch_attached"]:
            raise BatchError("controller batch supervision was detached unexpectedly")
        incoming_id = baseline["run_id"]
        if self.run_id is None:
            if incoming_id is None or incoming_id == self.previous_run_id:
                if packet_type == "view" and not packet["in_game"]:
                    return []  # Buffered/repeated menu from before this START.
                raise BatchError("in-game progress lacks the newly started run_id")
            if incoming_id in self.seen_run_ids:
                raise BatchError("controller reused a previously completed run_id")
            self.run_id = incoming_id
            self.seen_run_ids.add(incoming_id)
            self.report.rows[self.index]["run_id"] = incoming_id
            self.report.rows[self.index]["started_at"] = utc_now()
        elif incoming_id != self.run_id:
            raise BatchError("controller run_id changed before run completion")
        summary = self._summary(baseline)
        if summary is not None:
            row = self.report.rows[self.index]
            row["summary"] = copy.deepcopy(summary)
            row["internal_seed"] = summary["run_config"].get("internal_seed")
            self.report.save()
            if summary["terminal"]:
                self.phase = "await_menu"
        if packet_type == "model_update":
            return []  # This client never turns a model update into an action.
        if self.phase != "await_menu" or summary is None or not summary["terminal"]:
            raise BatchError("returned to menu without a matching eligible terminal summary")
        if not baseline["can_start_run"] or not baseline["ready_for_command"]:
            raise BatchError("return-to-menu view is not ready for a new run")
        self.report.freeze_experiment(baseline["experiment_dir"], baseline["model"])
        assert self.report.experiment_dir is not None and self.started is not None
        row = self.report.rows[self.index]
        row.update(status="completed", finished_at=utc_now(), elapsed_seconds=round(self.clock() - self.started, 3),
                   artifacts=self.report.artifacts(self.run_id),
                   api_performance=api_performance(self.report.experiment_dir / "requests.jsonl", {self.run_id}))
        self.position += 1
        self.started = None
        self.report.completed_performance = api_performance(
            self.report.experiment_dir / "requests.jsonl",
            {completed["run_id"] for completed in self.report.rows if completed["status"] == "completed"
             and not completed.get("reused_from")},
        )
        self.report.save()
        if self.position == len(self.order):
            self.phase = "detaching"
            return [{"type": "baseline_batch_detach", "protocol_version": PROTOCOL_VERSION}]
        return self._start(baseline)

    def abort(self, reason: str) -> None:
        elapsed = round(self.clock() - self.started, 3) if self.started is not None else None
        self.report.abort(self.index, reason, elapsed)
        self.phase = "aborted"


class JsonSocketClient:
    """A single bounded JSON-lines TCP connection, with no automatic reconnect."""

    def __init__(self, connection: socket.socket):
        self.connection = connection
        self.buffer = bytearray()

    def send(self, packet: dict) -> None:
        self.connection.sendall((json.dumps(packet, ensure_ascii=False) + "\n").encode("utf-8"))

    def receive(self, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("controller read timeout")
            self.connection.settimeout(remaining)
            chunk = self.connection.recv(65536)
            if not chunk:
                raise BatchError("controller disconnected before batch completion")
            self.buffer.extend(chunk)
            newline = self.buffer.find(b"\n")
            if newline > MAX_PACKET_BYTES or (newline < 0 and len(self.buffer) > MAX_PACKET_BYTES):
                raise BatchError("controller packet exceeds maximum size")
        line, _, rest = self.buffer.partition(b"\n")
        if len(line) > MAX_PACKET_BYTES:
            raise BatchError("controller packet exceeds maximum size")
        self.buffer = bytearray(rest)
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise BatchError(f"malformed controller JSON packet: {exc}") from exc
        if not isinstance(value, dict):
            raise BatchError("controller JSON packet must be an object")
        return value

    def close(self) -> None:
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.connection.close()


def run_batch(suite: dict, report: BatchReport, expected_model: str, host: str, port: int,
              connector: Callable[..., socket.socket] = socket.create_connection,
              clock: Callable[[], float] = time.monotonic,
              emit: Callable[[str], None] = print) -> int:
    state = BatchStateMachine(suite, report, expected_model, clock)
    client = None
    try:
        connection = connector((host, port), timeout=suite["limits"]["connection_timeout_seconds"])
        client = JsonSocketClient(connection)
        while not state.done:
            packet = client.receive(state.read_timeout())
            outgoing = state.consume(packet)
            if packet["type"] == "model_update":
                summary = (packet.get("baseline") or {}).get("run_summary") or {}
                emit(json.dumps({"run": f"{min(state.index + 1, len(suite['seeds']))}/{len(suite['seeds'])}",
                                 "phase": state.phase,
                                 "seed": suite["seeds"][min(state.index, len(suite["seeds"]) - 1)],
                                 "run_id": summary.get("run_id"), "accepted_actions": summary.get("accepted_actions"),
                                 "progress_score": (summary.get("evaluation") or {}).get("progress_score"),
                                 "terminal": summary.get("terminal", False)}, ensure_ascii=False))
            for command in outgoing:
                client.send(command)
        emit(f"Batch completed: {report.output_dir}")
        emit(json.dumps(aggregate_summaries({row["run_id"]: row["summary"] for row in report.rows
                                             if row["status"] == "completed"}), ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        # Fail closed before potentially expensive report/log processing.
        if client is not None:
            client.close()
            client = None
        state.abort("interrupted by user (Ctrl+C)")
        emit(f"Batch interrupted; partial report retained: {report.output_dir}")
        return 130
    except (BatchError, OSError, TimeoutError) as exc:
        if client is not None:
            client.close()
            client = None
        reason = str(exc)
        if isinstance(exc, TimeoutError):
            try:
                state.check_timeouts()
            except BatchError as timeout:
                reason = str(timeout)
        state.abort(reason)
        emit(f"Batch stopped: {reason}\nPartial report: {report.output_dir}")
        return 1
    finally:
        if client is not None:
            client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate fixed-seed IRONCLAD A0 baseline runs (one TCP client; close Overlay first)")
    parser.add_argument("--config", default="config/agent.json")
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dry-run", action="store_true", help="validate/print the complete plan without connecting or writing files")
    parser.add_argument("--no-reuse", action="store_true",
                        help="re-run seeds that already have a completed matching run instead of reusing them")
    args = parser.parse_args(argv)
    try:
        root = PROJECT_ROOT
        config_file = Path(args.config)
        config_file = config_file if config_file.is_absolute() else root / config_file
        suite_path = Path(args.suite)
        suite_path = suite_path if suite_path.is_absolute() else root / suite_path
        config = load_agent_config(root=root, config_file=config_file)
        if config["framework"] != MODE_FRAMEWORKS["baseline_inference"] or config["policy"]["runtime"] != "remote_api":
            raise BatchError("active config must select baseline_inference (autonomous remote_api, no learning)")
        suite = load_suite(suite_path)
        host = args.host if args.host is not None else config["server"]["host"]
        port = args.port if args.port is not None else config["server"]["port"]
        if not isinstance(host, str) or not host.strip() or type(port) is not int or not 1 <= port <= 65535:
            raise BatchError("host must be non-empty and port must be an integer from 1 to 65535")
        prompt_code_sha256 = hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "policy" / "local_qwen.py").read_bytes()).hexdigest()
        reused: dict[str, dict] = {}
        reuse_notes: list[dict] = []
        if not args.no_reuse:
            reused, reuse_notes = find_reusable_runs(
                (root / suite["output_dir"]).resolve(), suite,
                expected_fingerprint(config, prompt_code_sha256))
        missing = [seed for seed in suite["seeds"] if seed not in reused]
        plan = {"suite_id": suite["suite_id"], "character": "IRONCLAD", "ascension": 0,
                "seeds": suite["seeds"], "internal_seeds": [seed_to_internal(seed) for seed in suite["seeds"]],
                "limits": suite["limits"], "controller": {"host": host, "port": port},
                "model": config["policy"]["model"]["base"], "output_dir": str((root / suite["output_dir"]).resolve()),
                "reused": [{"seed": seed, "batch": reused[seed]["batch"], "run_id": reused[seed]["run_id"],
                            "progress_score": ((reused[seed]["summary"].get("evaluation") or {})
                                               .get("progress_score"))}
                           for seed in suite["seeds"] if seed in reused],
                "to_run": missing,
                "commands": [f"s IRONCLAD 0 {seed}" for seed in missing]}
        if reuse_notes:
            plan["ignored_batches"] = reuse_notes
        print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        if args.dry_run:
            return 0
        report = BatchReport(root, suite, suite_path, config_file, host, port,
                             expected_policy=config["policy"], reused=reused)
        if not missing:
            report.manifest.update(status="completed", finished_at=utc_now())
            report.save()
            print(f"Cohort already complete: {report.output_dir}")
            print(json.dumps(aggregate_summaries({row["run_id"]: row["summary"] for row in report.rows
                                                  if row["status"] == "completed"}), ensure_ascii=False))
            return 0
        print(f"Batch report: {report.output_dir}\nController owns all game decisions. Use this as the only TCP client.", flush=True)
        return run_batch(suite, report, config["policy"]["model"]["base"], host, port, emit=lambda message: print(message, flush=True))
    except (BatchError, ConfigError, OSError) as exc:
        print(f"Batch configuration/report error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
