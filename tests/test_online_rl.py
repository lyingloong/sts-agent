from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from online_rl.protocol import ProtocolError, validate_episode_payload  # noqa: E402
from online_rl.rewards import episode_reward  # noqa: E402
from online_rl.rollout_store import RolloutStore  # noqa: E402
from online_rl_worker import OnlineWorker, load_config, normalize_scores  # noqa: E402
import communication_controller as controller  # noqa: E402


def episode_payload(episode_id: str, group_id: str = "g-000001", version: str = "sft-v0") -> dict:
    return {
        "episode_id": episode_id,
        "group_id": group_id,
        "policy_version": version,
        "run_config": {"character": "IRONCLAD", "ascension": 0, "seed": "4QTKQ7VBMFX47"},
        "steps": [
            {
                "step_id": 0,
                "observation": {"screen_type": "NONE"},
                "legal_actions": [{"id": "A1", "label": "End"}],
                "selected_alias": "A1",
                "old_logprob": 0.0,
            }
        ],
        "terminal_state": {"screen_type": "GAME_OVER", "act": 1, "floor": 3, "current_hp": 0, "max_hp": 80},
    }


class OnlineRlTests(unittest.TestCase):
    def test_episode_requires_selected_alias_to_be_legal(self) -> None:
        payload = episode_payload("e-1")
        payload["steps"][0]["selected_alias"] = "A2"
        with self.assertRaises(ProtocolError):
            validate_episode_payload(payload)

    def test_reward_exposes_components(self) -> None:
        result = episode_reward({"screen_type": "VICTORY", "act": 3, "floor": 50, "current_hp": 20, "max_hp": 80})
        self.assertTrue(result["won"])
        self.assertEqual(result["victory"], 1000.0)
        self.assertGreater(float(result["total"]), 1300.0)

    def test_store_refuses_mixed_initial_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RolloutStore(Path(directory), group_size=4, initial_policy_version="sft-v0")
            group = store.begin_episode("e-1", {"character": "IRONCLAD", "ascension": 0, "seed": "AAA"})
            self.assertEqual(group["group_id"], "g-000001")
            with self.assertRaises(ProtocolError):
                store.begin_episode("e-2", {"character": "IRONCLAD", "ascension": 0, "seed": "BBB"})

    def test_mock_worker_updates_only_after_full_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "online.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model": "unused",
                        "initial_adapter": str(root / "adapter"),
                        "output_root": str(root / "output"),
                        "group_size": 2,
                    }
                ),
                encoding="utf-8",
            )
            worker = OnlineWorker(load_config(config_path), mock=True)
            request = {
                "request_id": 1,
                "episode_id": "e-1",
                "run_config": {"character": "IRONCLAD", "ascension": 0, "seed": "4QTKQ7VBMFX47"},
                "observation": {"screen_type": "NONE"},
                "legal_actions": [{"id": "A1", "label": "End"}, {"id": "A2", "label": "Play"}],
            }
            first = worker.sample(request)
            self.assertEqual(first["group_id"], "g-000001")
            one = episode_payload("e-1", group_id=first["group_id"], version=first["policy_version"])
            queued = worker.submit_episode({"episode": one})
            self.assertEqual(queued["type"], "episode_queued")
            second = worker.sample({**request, "episode_id": "e-2"})
            two = episode_payload("e-2", group_id=second["group_id"], version=second["policy_version"])
            updated = worker.submit_episode({"episode": two})
            self.assertEqual(updated["type"], "policy_updated")
            self.assertEqual(updated["policy_version"], "v000001")

    def test_temperature_normalization_tracks_sampling_policy(self) -> None:
        values = normalize_scores(
            [{"action_id": "A1", "logprob": -1.0}, {"action_id": "A2", "logprob": -2.0}],
            temperature=1.0,
        )
        self.assertAlmostEqual(sum(item["probability"] for item in values), 1.0)
        self.assertGreater(values[0]["probability"], values[1]["probability"])

    def test_trace_rebuilds_only_remote_sampled_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            record = {
                "step_id": 4,
                "accepted": True,
                "terminal": True,
                "pre_state": {"screen_type": "EVENT", "map": {"hidden": True}},
                "legal_actions": [{"id": "CHOOSE:index=0", "label": "Take"}],
                "decision": {
                    "mode": "remote_online_rl",
                    "source": "online_policy_sample",
                    "online_rl": {"group_id": "g-000001", "policy_version": "sft-v0", "selected_alias": "A1", "selected_logprob": 0.0},
                },
                "post_state": {"screen_type": "GAME_OVER", "act": 1, "floor": 2, "current_hp": 0, "max_hp": 80},
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            result = controller.online_episode_from_trace(
                path,
                "run-1",
                {"character": "IRONCLAD", "ascension": 0, "seed": "AAAA"},
            )
            self.assertEqual(result["group_id"], "g-000001")
            self.assertEqual(result["steps"][0]["selected_alias"], "A1")
            self.assertNotIn("map", result["steps"][0]["observation"])


if __name__ == "__main__":
    unittest.main()
