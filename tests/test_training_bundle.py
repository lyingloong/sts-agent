from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import prepare_training_bundle  # noqa: E402


class TrainingBundleTests(unittest.TestCase):
    def test_builds_valid_dataset_manifest_and_trace_copy(self) -> None:
        record = {
            "accepted": True,
            "stable_after_action": True,
            "terminal": True,
            "run_id": "run-1",
            "step_id": 1,
            "screen_type": "EVENT",
            "pre_state": {"screen_type": "EVENT"},
            "post_state": {"screen_type": "GAME_OVER"},
            "legal_actions": [{"id": "CHOOSE:index=0", "label": "Take"}],
            "action": {"id": "CHOOSE:index=0"},
            "decision": {"source": "teacher_auto", "teacher": {"model_id": "teacher-v1"}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            traces = root / "traces"
            traces.mkdir()
            trace_path = traces / "action-trace-run-1.jsonl"
            trace_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

            output = root / "bundle"
            manifest = prepare_training_bundle.build_bundle(traces, output, include_traces=True)

            self.assertEqual(manifest["dataset"]["samples"], 1)
            self.assertEqual(manifest["source"]["terminal_outcome_counts"], {"game_over": 1})
            self.assertIn("dirty", manifest["repository"])
            self.assertTrue((output / "dataset" / "teacher-sft.jsonl").is_file())
            self.assertTrue((output / "traces" / trace_path.name).is_file())
            self.assertIn("dataset/teacher-sft.jsonl", (output / "SHA256SUMS").read_text(encoding="utf-8"))

            with self.assertRaises(ValueError):
                prepare_training_bundle.build_bundle(traces, output)


if __name__ == "__main__":
    unittest.main()
