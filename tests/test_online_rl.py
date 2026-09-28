from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_agent.learning.online_rl.protocol import ProtocolError, validate_episode_payload
from sts_agent.learning.online_rl.rewards import episode_reward
from sts_agent.learning.online_rl.rollout_store import RolloutStore
from sts_agent.learning.online_rl.update import attribution_correction_loss
from sts_agent.learning.online_rl.worker import OnlineWorker, load_config, normalize_scores, select_gpu
from sts_agent.policy.local_qwen import configure_inference_gpu, parse_gpu_device
from sts_agent.runtime import controller


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


def load_mock_worker_config(path: Path) -> dict:
    return load_config(
        path,
        model="unused",
        initial_adapter="",
        initial_policy_version="sft-v0",
        mock=True,
    )


class OnlineRlTests(unittest.TestCase):
    def test_episode_requires_selected_alias_to_be_legal(self) -> None:
        payload = episode_payload("e-1")
        payload["steps"][0]["selected_alias"] = "A2"
        with self.assertRaises(ProtocolError):
            validate_episode_payload(payload)

    def test_episode_validates_attribution_against_sampled_steps(self) -> None:
        payload = episode_payload("e-1")
        payload["attributions"] = [{"step_id": 0, "direction": "weaken", "confidence": 0.9, "severity": 3}]
        parsed = validate_episode_payload(payload)
        self.assertEqual(parsed["attributions"][0]["direction"], "weaken")
        payload["attributions"][0]["step_id"] = 99
        with self.assertRaises(ProtocolError):
            validate_episode_payload(payload)

    def test_reward_exposes_components(self) -> None:
        result = episode_reward({"screen_type": "VICTORY", "act": 3, "floor": 50, "current_hp": 20, "max_hp": 80})
        self.assertTrue(result["won"])
        self.assertFalse(result["beat_heart"])
        self.assertEqual(result["victory"], 5000.0)
        self.assertEqual(result["act"], 1800.0)
        self.assertEqual(result["floor"], 1500.0)
        self.assertEqual(result["hp_fraction"], 0.5)
        self.assertEqual(result["total"], 8300.5)

    def test_heart_victory_receives_extra_terminal_reward(self) -> None:
        result = episode_reward({"screen_type": "VICTORY", "act": 4, "floor": 54, "current_hp": 20, "max_hp": 100})
        self.assertTrue(result["beat_heart"])
        self.assertEqual(result["heart"], 10000.0)
        self.assertGreater(float(result["total"]), 19000.0)

    def test_auto_gpu_prefers_low_utilization_then_more_free_memory(self) -> None:
        selected = select_gpu(
            [
                {"index": 0, "free_memory_mib": 23000, "utilization_percent": 9},
                {"index": 1, "free_memory_mib": 22000, "utilization_percent": 0},
                {"index": 2, "free_memory_mib": 8000, "utilization_percent": 0},
            ],
            12000,
        )
        self.assertEqual(selected["index"], 1)

    def test_auto_gpu_rejects_insufficient_free_memory(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "at least 12000 MiB"):
            select_gpu([{"index": 0, "free_memory_mib": 11999, "utilization_percent": 0}], 12000)

    def test_worker_model_identity_is_supplied_by_controller_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "online.json"
            config_path.write_text(json.dumps({"output_root": str(root / "output")}), encoding="utf-8")
            config = load_config(
                config_path,
                model="/models/Qwen3-0.6B",
                initial_adapter="/models/sft-v1",
                initial_policy_version="sft-v1",
            )
            self.assertEqual(config["model"], "/models/Qwen3-0.6B")
            self.assertEqual(config["initial_adapter"], "/models/sft-v1")
            self.assertEqual(config["initial_policy_version"], "sft-v1")

    def test_worker_rejects_duplicated_model_identity_in_remote_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "online.json"
            config_path.write_text(json.dumps({"model": "should-not-be-here"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown online RL configuration field: model"):
                load_mock_worker_config(config_path)

    def test_worker_rejects_tied_advantage_above_attribution_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "online.json"
            config_path.write_text(
                json.dumps(
                    {
                        "attribution": {
                            "max_abs_advantage": 0.1,
                            "min_tied_advantage": 0.2,
                        }
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must not exceed"):
                load_mock_worker_config(config_path)

    def test_worker_rejects_negative_correction_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "online.json"
            config_path.write_text(
                json.dumps({"update": {"weaken_unlikelihood_weight": -0.1}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must be non-negative"):
                load_mock_worker_config(config_path)

    def test_reinforce_correction_has_strong_gradient_for_low_probability_action(self) -> None:
        import torch

        scores = torch.tensor([-11.0, 0.0], requires_grad=True)
        loss, kind = attribution_correction_loss(
            torch,
            scores,
            0,
            "reinforce",
            weight=1.0,
            total_weight=1.0,
            reinforce_ce_weight=1.0,
            weaken_unlikelihood_weight=1.0,
        )
        self.assertEqual(kind, "reinforce_ce")
        self.assertIsNotNone(loss)
        assert loss is not None
        loss.backward()
        self.assertLess(float(scores.grad[0]), -0.99)

    def test_weaken_unlikelihood_has_strong_gradient_for_saturated_action(self) -> None:
        import torch

        scores = torch.tensor([11.0, 0.0], requires_grad=True)
        loss, kind = attribution_correction_loss(
            torch,
            scores,
            0,
            "weaken",
            weight=1.0,
            total_weight=1.0,
            reinforce_ce_weight=1.0,
            weaken_unlikelihood_weight=1.0,
        )
        self.assertEqual(kind, "weaken_unlikelihood")
        self.assertIsNotNone(loss)
        assert loss is not None
        loss.backward()
        self.assertGreater(float(scores.grad[0]), 0.99)

    def test_weaken_unlikelihood_skips_a_forced_action(self) -> None:
        import torch

        loss, kind = attribution_correction_loss(
            torch,
            torch.tensor([0.0], requires_grad=True),
            0,
            "weaken",
            weight=1.0,
            total_weight=1.0,
            reinforce_ce_weight=1.0,
            weaken_unlikelihood_weight=1.0,
        )
        self.assertIsNone(loss)
        self.assertIsNone(kind)

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
                        "output_root": str(root / "output"),
                        "group_size": 2,
                    }
                ),
                encoding="utf-8",
            )
            worker = OnlineWorker(load_mock_worker_config(config_path), mock=True)
            self.assertTrue(worker.config["update"]["gradient_checkpointing"])
            self.assertTrue(worker.config["update"]["empty_cache_between_steps"])
            self.assertEqual(worker.config["attribution"]["min_tied_advantage"], 0.2)
            self.assertEqual(worker.config["update"]["reinforce_ce_weight"], 0.25)
            self.assertEqual(worker.config["update"]["weaken_unlikelihood_weight"], 0.25)
            request = {
                "request_id": 1,
                "episode_id": "e-1",
                "run_config": {"character": "IRONCLAD", "ascension": 0, "seed": "4QTKQ7VBMFX47"},
                "observation": {"screen_type": "NONE"},
                "legal_actions": [{"id": "A1", "label": "End"}, {"id": "A2", "label": "Play"}],
            }
            first = worker.sample(request)
            self.assertEqual(first["group_id"], "g-000001")
            self.assertEqual(first["iteration"], 0)
            self.assertEqual(first["rollout_index"], 1)
            self.assertEqual(first["episodes_collected"], 0)
            self.assertEqual(first["group_size"], 2)
            one = episode_payload("e-1", group_id=first["group_id"], version=first["policy_version"])
            queued = worker.submit_episode({"episode": one})
            self.assertEqual(queued["type"], "episode_queued")
            self.assertEqual(queued["episodes_collected"], 1)
            second = worker.sample({**request, "episode_id": "e-2"})
            self.assertEqual(second["rollout_index"], 2)
            self.assertEqual(second["episodes_collected"], 1)
            two = episode_payload("e-2", group_id=second["group_id"], version=second["policy_version"])
            updated = worker.submit_episode({"episode": two})
            self.assertEqual(updated["type"], "policy_updated")
            self.assertEqual(updated["policy_version"], "v000001")
            self.assertEqual(updated["iteration"], 1)

    def test_attribution_worker_closes_empty_group_without_new_policy_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "online.json"
            config_path.write_text(
                json.dumps(
                    {
                        "output_root": str(root / "output"),
                        "group_size": 2,
                        "attribution": {"enabled": True},
                    }
                ),
                encoding="utf-8",
            )
            worker = OnlineWorker(load_mock_worker_config(config_path), mock=True)
            request = {
                "request_id": 1,
                "episode_id": "e-1",
                "run_config": {"character": "IRONCLAD", "ascension": 0, "seed": "4QTKQ7VBMFX47"},
                "observation": {"screen_type": "NONE"},
                "legal_actions": [{"id": "A1", "label": "End"}],
            }
            first = worker.sample(request)
            worker.submit_episode({"episode": episode_payload("e-1", first["group_id"], first["policy_version"])})
            second = worker.sample({**request, "episode_id": "e-2"})
            completed = worker.submit_episode({"episode": episode_payload("e-2", second["group_id"], second["policy_version"])})
            self.assertEqual(completed["type"], "group_completed_no_update")
            self.assertEqual(completed["policy_version"], "sft-v0")
            self.assertEqual(completed["iteration"], 0)

    def test_attribution_worker_saves_new_policy_when_a_step_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "online.json"
            config_path.write_text(
                json.dumps(
                    {
                        "output_root": str(root / "output"),
                        "group_size": 2,
                        "attribution": {"enabled": True},
                    }
                ),
                encoding="utf-8",
            )
            worker = OnlineWorker(load_mock_worker_config(config_path), mock=True)
            request = {
                "request_id": 1,
                "episode_id": "e-1",
                "run_config": {"character": "IRONCLAD", "ascension": 0, "seed": "4QTKQ7VBMFX47"},
                "observation": {"screen_type": "NONE"},
                "legal_actions": [{"id": "A1", "label": "End"}],
            }
            first = worker.sample(request)
            one = episode_payload("e-1", first["group_id"], first["policy_version"])
            one["attributions"] = [{"step_id": 0, "direction": "reinforce", "confidence": 0.9, "severity": 2}]
            worker.submit_episode({"episode": one})
            second = worker.sample({**request, "episode_id": "e-2"})
            two = episode_payload("e-2", second["group_id"], second["policy_version"])
            completed = worker.submit_episode({"episode": two})
            self.assertEqual(completed["type"], "policy_updated")
            self.assertEqual(completed["policy_version"], "v000001")

    def test_temperature_normalization_tracks_sampling_policy(self) -> None:
        values = normalize_scores(
            [{"action_id": "A1", "logprob": -1.0}, {"action_id": "A2", "logprob": -2.0}],
            temperature=1.0,
        )
        self.assertAlmostEqual(sum(item["probability"] for item in values), 1.0)
        self.assertGreater(values[0]["probability"], values[1]["probability"])

    def test_inference_gpu_argument_parser_rejects_invalid_indexes(self) -> None:
        self.assertEqual(parse_gpu_device("auto"), "auto")
        self.assertEqual(parse_gpu_device("visible"), "visible")
        self.assertEqual(parse_gpu_device("3"), 3)
        with self.assertRaises(Exception):
            parse_gpu_device("-1")

    def test_visible_inference_gpu_does_not_query_cuda(self) -> None:
        self.assertIsNone(configure_inference_gpu("visible", 1))

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
