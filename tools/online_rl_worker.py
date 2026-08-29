"""Persistent single-GPU rollout sampler and episode-level online RL updater.

The worker speaks one JSON object per stdin line and writes one JSON object per
stdout line. It is designed to run behind ``ssh -T``; no public TCP endpoint is
opened. ``sample`` starts/continues a frozen K-rollout group and ``episode``
submits a completed local game. Once K episodes arrive, the worker updates the
resident LoRA adapter before accepting the next group.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any

from gpu_selection import select_configured_gpu, select_gpu, validate_gpu_selector
from online_rl.protocol import ProtocolError, atomic_write_json, require_object, require_string
from online_rl.rewards import DEFAULT_REWARD_WEIGHTS, episode_reward
from online_rl.rollout_store import RolloutStore
from online_rl.update import TrainableAliasPolicy, UpdateConfig


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "model": "Qwen/Qwen3-0.6B",
    "initial_adapter": "",
    "initial_policy_version": "sft-v0",
    "output_root": "outputs/online_rl/default",
    "group_size": 4,
    "seed": 20260826,
    "sampling": {"temperature": 1.2},
    "reward": {
        "version": "v2_terminal_progress_heart",
        "weights": dict(DEFAULT_REWARD_WEIGHTS),
    },
    "gpu": {"device": "auto", "minimum_free_memory_mib": 12000},
    "attribution": {
        "enabled": False,
        "max_abs_advantage": 2.0,
    },
    "update": {
        "max_length": 4096,
        "learning_rate": 1e-5,
        "clip_range": 0.2,
        "update_epochs": 1,
        "max_grad_norm": 1.0,
        "load_in_4bit": True,
        "gradient_checkpointing": True,
        "empty_cache_between_steps": True,
    },
}


def configure_worker_gpu(gpu_config: dict[str, Any]) -> dict[str, int]:
    """Pin the worker to one physical GPU before importing Torch."""

    device = gpu_config["device"]
    minimum = int(gpu_config["minimum_free_memory_mib"])
    selected = select_configured_gpu(device, minimum)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(selected["index"])
    # This process performs variable-length 4k-token backward passes. Set the
    # allocator before Torch is imported so freed blocks can be reused without
    # splitting the GPU heap into unusable fragments.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return selected


def merge(default: dict[str, Any], supplied: dict[str, Any], location: str = "") -> dict[str, Any]:
    result = copy.deepcopy(default)
    for key, value in supplied.items():
        path = f"{location}.{key}" if location else key
        if key not in default:
            raise ValueError(f"unknown online RL configuration field: {path}")
        if isinstance(default[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{path} must be an object")
            result[key] = merge(default[key], value, path)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing online RL config: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid online RL JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("online RL config must be an object")
    config = merge(DEFAULT_CONFIG, raw)
    if config["schema_version"] != 1:
        raise ValueError("online RL schema_version must be 1")
    if not isinstance(config["group_size"], int) or config["group_size"] < 2:
        raise ValueError("group_size must be an integer >= 2")
    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")
    if not isinstance(config["sampling"]["temperature"], (int, float)) or config["sampling"]["temperature"] <= 0:
        raise ValueError("sampling.temperature must be positive")
    gpu = config["gpu"]
    validate_gpu_selector(gpu["device"], gpu["minimum_free_memory_mib"])
    for name, value in config["reward"]["weights"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"reward.weights.{name} must be a non-negative finite number")
    for key in ("max_length", "update_epochs"):
        if not isinstance(config["update"][key], int) or config["update"][key] < 1:
            raise ValueError(f"update.{key} must be an integer >= 1")
    for key in ("learning_rate", "clip_range", "max_grad_norm"):
        if not isinstance(config["update"][key], (int, float)) or config["update"][key] <= 0:
            raise ValueError(f"update.{key} must be positive")
    if not isinstance(config["update"]["load_in_4bit"], bool):
        raise ValueError("update.load_in_4bit must be true or false")
    for key in ("gradient_checkpointing", "empty_cache_between_steps"):
        if not isinstance(config["update"][key], bool):
            raise ValueError(f"update.{key} must be true or false")
    attribution = config["attribution"]
    if not isinstance(attribution["enabled"], bool):
        raise ValueError("attribution.enabled must be true or false")
    if not isinstance(attribution["max_abs_advantage"], (int, float)) or attribution["max_abs_advantage"] <= 0:
        raise ValueError("attribution.max_abs_advantage must be positive")
    return config


def normalize_scores(scores: list[dict[str, Any]], temperature: float) -> list[dict[str, Any]]:
    if not scores:
        raise ProtocolError("policy returned no legal action scores")
    values = [float(item["logprob"]) / temperature for item in scores]
    if not all(math.isfinite(value) for value in values):
        raise ProtocolError("policy returned a non-finite log probability")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    if not total or not math.isfinite(total):
        raise ProtocolError("policy action score normalization failed")
    return [
        {**item, "probability": weight / total, "normalized_logprob": math.log(weight / total)}
        for item, weight in zip(scores, weights)
    ]


class MockOnlinePolicy:
    """CPU-only protocol smoke implementation; never use it for an experiment."""

    def score(self, observation: dict[str, Any], legal_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        del observation
        result = []
        for index, action in enumerate(legal_actions):
            alias = str(action.get("id") or "")
            if alias:
                result.append({"action_id": alias, "label": str(action.get("label") or alias), "logprob": -float(index)})
        return result

    def update(
        self,
        episodes: list[dict[str, Any]],
        rewards: list[float],
        sampling_temperature: float = 1.0,
        *,
        attribution_enabled: bool = False,
        max_abs_advantage: float = 2.0,
    ) -> dict[str, Any]:
        del sampling_temperature, max_abs_advantage
        selected = sum(len(item.get("attributions", [])) for item in episodes) if attribution_enabled else sum(
            len(item["steps"]) for item in episodes
        )
        return {
            "algorithm": "mock_signed_masked_episode_grpo" if attribution_enabled else "mock",
            "episodes": len(episodes),
            "rewards": rewards,
            "steps": selected,
            "updated": bool(selected),
        }

    def save_adapter(self, path: Path) -> None:
        atomic_write_json(path / "mock-adapter.json", {"warning": "mock worker has no trainable adapter"})


class OnlineWorker:
    def __init__(self, config: dict[str, Any], mock: bool = False) -> None:
        self.config = config
        self.output_root = Path(str(config["output_root"])).resolve()
        self.store = RolloutStore(
            self.output_root,
            int(config["group_size"]),
            str(config["initial_policy_version"]),
        )
        current_version = self.store.current_policy_version
        initial_adapter = Path(str(config["initial_adapter"])).resolve()
        adapter = initial_adapter if current_version == str(config["initial_policy_version"]) else self.output_root / "versions" / current_version
        if mock:
            self.policy: MockOnlinePolicy | TrainableAliasPolicy = MockOnlinePolicy()
            self.model_id = "mock-online-policy"
            self.gpu: dict[str, int] | None = None
        else:
            self.gpu = configure_worker_gpu(config["gpu"])
            update = config["update"]
            self.policy = TrainableAliasPolicy(
                str(config["model"]),
                adapter,
                UpdateConfig(
                    max_length=int(update["max_length"]),
                    learning_rate=float(update["learning_rate"]),
                    clip_range=float(update["clip_range"]),
                    update_epochs=int(update["update_epochs"]),
                    max_grad_norm=float(update["max_grad_norm"]),
                    gradient_checkpointing=bool(update["gradient_checkpointing"]),
                    empty_cache_between_steps=bool(update["empty_cache_between_steps"]),
                ),
                load_in_4bit=bool(update["load_in_4bit"]),
            )
            self.model_id = str(config["model"])
        self.random = random.Random(int(config["seed"]))

    def sample(self, request: dict[str, Any]) -> dict[str, Any]:
        observation = require_object(request.get("observation"), "observation")
        actions = request.get("legal_actions")
        if not isinstance(actions, list) or not actions:
            raise ProtocolError("legal_actions must be a non-empty list")
        legal_actions = [item for item in actions if isinstance(item, dict) and str(item.get("id") or "")]
        if len(legal_actions) != len(actions):
            raise ProtocolError("legal_actions must contain only objects with ids")
        episode_id = require_string(request.get("episode_id"), "episode_id")
        run_config = require_object(request.get("run_config"), "run_config")
        expected_version = request.get("expected_policy_version")
        if expected_version is not None and str(expected_version) != self.store.current_policy_version:
            raise ProtocolError(f"policy version changed; expected {expected_version}, current {self.store.current_policy_version}")
        group = self.store.begin_episode(episode_id, run_config)
        episodes = group.get("episodes", {})
        if not isinstance(episodes, dict):
            raise ProtocolError("rollout group episodes must be an object")
        episode_ids = list(episodes)
        try:
            rollout_index = episode_ids.index(episode_id) + 1
        except ValueError as exc:  # Defensive: begin_episode must admit this ID.
            raise ProtocolError("episode is missing from its rollout group") from exc
        episodes_collected = sum(
            1 for entry in episodes.values() if isinstance(entry, dict) and entry.get("status") == "submitted"
        )
        scores = normalize_scores(self.policy.score(observation, legal_actions), float(self.config["sampling"]["temperature"]))
        draw = self.random.random()
        cumulative = 0.0
        selected = scores[-1]
        for item in scores:
            cumulative += float(item["probability"])
            if draw <= cumulative:
                selected = item
                break
        return {
            "type": "sampled_action",
            "request_id": request.get("request_id"),
            "model_id": self.model_id,
            "policy_version": self.store.current_policy_version,
            "iteration": self.store.completed_update_count,
            "group_id": group["group_id"],
            "group_size": int(self.config["group_size"]),
            "rollout_index": rollout_index,
            "episodes_collected": episodes_collected,
            "action_id": selected["action_id"],
            "selected_logprob": selected["normalized_logprob"],
            "sampling_draw": draw,
            "action_scores": scores,
        }

    def submit_episode(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = require_object(request.get("episode"), "episode")
        group, ready = self.store.submit_episode(payload)
        response: dict[str, Any] = {
            "type": "episode_queued",
            "request_id": request.get("request_id"),
            "group_id": group["group_id"],
            "policy_version": self.store.current_policy_version,
            "iteration": self.store.completed_update_count,
            "episodes_collected": len(group["episodes"]),
            "group_size": int(self.config["group_size"]),
        }
        if not ready:
            return response
        episodes = self.store.load_submitted_episodes(group)
        reward_details = [episode_reward(item["terminal_state"], self.config["reward"]["weights"]) for item in episodes]
        rewards = [float(item["total"]) for item in reward_details]
        attribution = self.config["attribution"]
        metrics = self.policy.update(
            episodes,
            rewards,
            sampling_temperature=float(self.config["sampling"]["temperature"]),
            attribution_enabled=bool(attribution["enabled"]),
            max_abs_advantage=float(attribution["max_abs_advantage"]),
        )
        metrics["reward_details"] = reward_details
        metrics["parent_policy_version"] = group["policy_version"]
        if metrics.get("updated", True):
            new_version = self.store.next_policy_version()
            adapter_path = self.output_root / "versions" / new_version
            self.policy.save_adapter(adapter_path)
            metrics["adapter_path"] = str(adapter_path)
            self.store.complete_group(str(group["group_id"]), new_version, metrics)
            response.update({"type": "policy_updated", "policy_version": new_version, "update": metrics})
        else:
            current_version = self.store.current_policy_version
            self.store.complete_group(
                str(group["group_id"]),
                current_version,
                metrics,
                policy_changed=False,
            )
            response.update({"type": "group_completed_no_update", "policy_version": current_version, "update": metrics})
        response["iteration"] = self.store.completed_update_count
        return response


def configure_stdio() -> None:
    for stream in (sys.stdin, sys.stdout):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="strict")


def serve(config: dict[str, Any], mock: bool, debug: bool) -> int:
    configure_stdio()
    worker = OnlineWorker(config, mock=mock)
    print(
        json.dumps(
            {
                "type": "ready",
                "mode": "remote_online_rl",
                "model_id": worker.model_id,
                "policy_version": worker.store.current_policy_version,
                "iteration": worker.store.completed_update_count,
                "group_size": config["group_size"],
                "gpu": worker.gpu,
                "attribution_enabled": bool(config["attribution"]["enabled"]),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        request: Any = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ProtocolError("request must be an object")
            if request.get("type") == "shutdown":
                break
            if request.get("type") == "sample":
                response = worker.sample(request)
            elif request.get("type") == "episode":
                response = worker.submit_episode(request)
            elif request.get("type") == "status":
                response = {
                    "type": "status",
                    "request_id": request.get("request_id"),
                    "policy_version": worker.store.current_policy_version,
                }
            else:
                raise ProtocolError("expected request type sample, episode, status, or shutdown")
            print(json.dumps(response, ensure_ascii=False), flush=True)
        except Exception as exc:
            print(
                json.dumps(
                    {"type": "error", "request_id": request.get("request_id") if isinstance(request, dict) else None, "error": str(exc)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if debug:
                traceback.print_exc(file=sys.stderr)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU StS online rollout and update worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mock", action="store_true", help="protocol-only worker without a model or update")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return serve(load_config(args.config), mock=args.mock, debug=args.debug)
    except (OSError, RuntimeError, ValueError, ProtocolError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
