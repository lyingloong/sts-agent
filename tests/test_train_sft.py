from __future__ import annotations

import sys
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import train_sft  # noqa: E402


class TrainSftTests(unittest.TestCase):
    def test_non_map_trace_observation_drops_map_like_live_inference(self) -> None:
        routed = train_sft.route_trace_observation(
            {
                "screen_type": "EVENT",
                "room_phase": "EVENT",
                "map": {"nodes": [{"x": 1, "y": 1}]},
                "choices": ["Take", "Leave"],
            }
        )
        self.assertNotIn("map", routed)
        self.assertEqual(routed["choices"], ["Take", "Leave"])

    def test_combat_trace_observation_uses_compact_live_schema(self) -> None:
        routed = train_sft.route_trace_observation(
            {
                "screen_type": "NONE",
                "room_phase": "COMBAT",
                "floor": 4,
                "combat": {"hand": [{"id": "Strike_R"}]},
                "map": {"nodes": [{"x": 1}]},
                "deck": [{"id": "Strike_R"}],
                "screen_state": {"unneeded": True},
            }
        )
        self.assertEqual(routed["combat"]["hand"][0]["id"], "Strike_R")
        self.assertNotIn("map", routed)
        self.assertNotIn("deck", routed)
        self.assertNotIn("screen_state", routed)

    def test_split_is_by_run_and_deterministic(self) -> None:
        samples = [
            {"run_id": f"run-{run}", "sample_id": f"run-{run}:{step}"}
            for run in range(4)
            for step in range(2)
        ]
        train, validation, validation_ids = train_sft.split_by_run(samples, validation_runs=1, seed=3)
        self.assertEqual(len(validation_ids), 1)
        self.assertEqual({item["run_id"] for item in validation}, set(validation_ids))
        self.assertTrue({item["run_id"] for item in train}.isdisjoint(set(validation_ids)))
        self.assertEqual(validation_ids, train_sft.split_by_run(samples, validation_runs=1, seed=3)[2])

    def test_sample_validation_rejects_target_outside_legal_aliases(self) -> None:
        sample = {
            "sample_id": "run-1:1",
            "run_id": "run-1",
            "observation": {},
            "legal_actions": [{"id": "A1", "label": "Take"}],
            "target": {"alias": "A2"},
        }
        with self.assertRaises(ValueError):
            train_sft.validate_sample(sample)


if __name__ == "__main__":
    unittest.main()
