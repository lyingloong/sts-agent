"""A small durable queue for same-policy, same-start-condition rollout groups."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .protocol import ProtocolError, atomic_write_json, start_key, validate_episode_payload


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class RolloutStore:
    def __init__(self, root: Path, group_size: int, initial_policy_version: str) -> None:
        if group_size < 2:
            raise ValueError("group_size must be at least 2")
        self.root = root
        self.groups_root = root / "groups"
        self.state_path = root / "state.json"
        self.group_size = group_size
        self.root.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state(initial_policy_version)

    def _load_state(self, initial_policy_version: str) -> dict[str, Any]:
        if self.state_path.is_file():
            import json

            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ProtocolError("online RL state.json must be an object")
            return value
        state = {
            "schema_version": "online-rl-0.1",
            "current_policy_version": initial_policy_version,
            "next_policy_index": 1,
            "next_group_index": 1,
            "active_group_id": None,
            "groups": {},
        }
        atomic_write_json(self.state_path, state)
        return state

    @property
    def current_policy_version(self) -> str:
        return str(self.state["current_policy_version"])

    @property
    def completed_update_count(self) -> int:
        """Return the number of successful policy updates in this experiment."""
        return max(0, int(self.state.get("next_policy_index", 1)) - 1)

    def _save(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def _group(self, group_id: str) -> dict[str, Any]:
        value = self.state["groups"].get(group_id)
        if not isinstance(value, dict):
            raise ProtocolError(f"unknown group_id: {group_id}")
        return value

    def begin_episode(self, episode_id: str, run_config: dict[str, Any]) -> dict[str, Any]:
        if not episode_id:
            raise ProtocolError("episode_id must be non-empty")
        key = start_key(run_config)
        for group in self.state["groups"].values():
            if not isinstance(group, dict):
                continue
            episode = group.get("episodes", {}).get(episode_id)
            if isinstance(episode, dict):
                return group
        active_id = self.state.get("active_group_id")
        if active_id is None:
            group_id = f"g-{int(self.state['next_group_index']):06d}"
            self.state["next_group_index"] = int(self.state["next_group_index"]) + 1
            group = {
                "group_id": group_id,
                "status": "collecting",
                "policy_version": self.current_policy_version,
                "start": key,
                "created_at_utc": now_utc(),
                "episodes": {},
            }
            self.state["groups"][group_id] = group
            self.state["active_group_id"] = group_id
        else:
            group = self._group(str(active_id))
            if group.get("status") != "collecting":
                raise ProtocolError("the active rollout group is not accepting episodes")
            if group.get("policy_version") != self.current_policy_version:
                raise ProtocolError("active group policy version is stale")
            if group.get("start") != key:
                raise ProtocolError("finish the active K-rollout group before changing character, ascension, or seed")
        episodes = group["episodes"]
        if len(episodes) >= self.group_size:
            raise ProtocolError("active rollout group already has K episodes")
        episodes[episode_id] = {"status": "started", "started_at_utc": now_utc()}
        self._save()
        self._write_group(group)
        return group

    def submit_episode(self, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        episode = validate_episode_payload(payload)
        group = self._group(episode["group_id"])
        if group.get("status") != "collecting":
            raise ProtocolError("rollout group is not collecting episodes")
        if group.get("policy_version") != episode["policy_version"]:
            raise ProtocolError("episode policy version does not match its group")
        if group.get("start") != episode["run_config"]:
            raise ProtocolError("episode initial condition does not match its group")
        entry = group.get("episodes", {}).get(episode["episode_id"])
        if not isinstance(entry, dict):
            raise ProtocolError("episode was not admitted by a sample request")
        if entry.get("status") == "submitted":
            return group, self._is_ready(group)
        if entry.get("status") != "started":
            raise ProtocolError("episode is in an invalid lifecycle state")
        episode_path = self.groups_root / str(group["group_id"]) / f"episode-{episode['episode_id']}.json"
        atomic_write_json(episode_path, episode)
        entry.update({"status": "submitted", "submitted_at_utc": now_utc(), "path": str(episode_path)})
        self._save()
        self._write_group(group)
        return group, self._is_ready(group)

    def load_submitted_episodes(self, group: dict[str, Any]) -> list[dict[str, Any]]:
        import json

        result: list[dict[str, Any]] = []
        for episode_id, entry in group.get("episodes", {}).items():
            if not isinstance(entry, dict) or entry.get("status") != "submitted":
                raise ProtocolError(f"episode {episode_id} has not been submitted")
            path = Path(str(entry.get("path") or ""))
            if not path.is_file():
                raise ProtocolError(f"episode payload is missing: {path}")
            result.append(validate_episode_payload(json.loads(path.read_text(encoding="utf-8"))))
        return result

    def complete_group(
        self,
        group_id: str,
        new_policy_version: str,
        result: dict[str, Any],
        *,
        policy_changed: bool = True,
    ) -> None:
        group = self._group(group_id)
        if not self._is_ready(group):
            raise ProtocolError("cannot complete an incomplete rollout group")
        group.update({"status": "completed", "completed_at_utc": now_utc(), "update": result})
        if policy_changed:
            self.state["current_policy_version"] = new_policy_version
            self.state["next_policy_index"] = int(self.state.get("next_policy_index", 1)) + 1
        self.state["active_group_id"] = None
        self._save()
        self._write_group(group)

    def next_policy_version(self) -> str:
        return f"v{int(self.state.get('next_policy_index', 1)):06d}"

    def _is_ready(self, group: dict[str, Any]) -> bool:
        episodes = group.get("episodes", {})
        return len(episodes) == self.group_size and all(
            isinstance(entry, dict) and entry.get("status") == "submitted" for entry in episodes.values()
        )

    def _write_group(self, group: dict[str, Any]) -> None:
        atomic_write_json(self.groups_root / str(group["group_id"]) / "group.json", group)
