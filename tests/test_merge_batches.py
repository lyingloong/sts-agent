from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_agent.commands.merge_batches import MergeError, merge_batches


def _summary(seed: str, score: float, *, won: bool = False, human: int = 0) -> dict:
    return {"run_id": f"run-{seed}", "accepted_actions": 1, "human_actions": human,
            "terminal": True, "won": won, "beat_heart": False,
            "run_config": {"seed": seed, "character": "IRONCLAD", "ascension": 0},
            "evaluation": {"autonomous_eligible": True, "progress_score": score,
                           "boss_acts_defeated": [1] if score >= 16.67 else [],
                           "highest_floor": 1, "highest_act": 1}}


def _write_batch(root: Path, name: str, rows: list[dict], model: str = "qwen3.8-chat") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "results.json").write_text(json.dumps({"schema_version": 1, "suite_id": name,
                                                        "status": "completed", "results": rows}), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps({"status": "completed",
                                                          "model_manifest": {"model": model}}), encoding="utf-8")
    return directory


def _row(seed: str, score: float, *, status: str = "completed", **summary_kwargs) -> dict:
    return {"seed": seed, "status": status, "run_id": f"run-{seed}",
            "summary": _summary(seed, score, **summary_kwargs) if status == "completed" else None}


class MergeBatchesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_merges_disjoint_batches_into_one_cohort(self) -> None:
        first = _write_batch(self.root, "first", [_row("A", 10.0), _row("B", 20.0)])
        second = _write_batch(self.root, "second", [_row("C", 30.0)])
        merged = merge_batches([first, second])
        evaluation = merged["evaluation"]
        self.assertEqual(evaluation["eligible_runs"], 3)
        self.assertAlmostEqual(evaluation["progress_score_mean"], 20.0)
        self.assertEqual([item["seed"] for item in merged["per_seed"]], ["A", "B", "C"])
        self.assertEqual(evaluation["model"], "qwen3.8-chat")

    def test_rejects_duplicate_completed_seeds(self) -> None:
        first = _write_batch(self.root, "first", [_row("A", 10.0)])
        second = _write_batch(self.root, "second", [_row("A", 20.0)])
        with self.assertRaises(MergeError):
            merge_batches([first, second])

    def test_rejects_mixed_models_unless_allowed(self) -> None:
        first = _write_batch(self.root, "first", [_row("A", 10.0)], model="qwen3.8-chat")
        second = _write_batch(self.root, "second", [_row("B", 20.0)], model="Qwen/Qwen3.8-27B")
        with self.assertRaises(MergeError):
            merge_batches([first, second])
        merged = merge_batches([first, second], allow_different_models=True)
        self.assertIsNone(merged["evaluation"]["model"])

    def test_non_completed_rows_are_reported_but_not_scored(self) -> None:
        first = _write_batch(self.root, "first", [_row("A", 10.0), _row("B", 0.0, status="technical_abort")])
        merged = merge_batches([first])
        self.assertEqual(merged["evaluation"]["eligible_runs"], 1)
        self.assertAlmostEqual(merged["evaluation"]["progress_score_mean"], 10.0)


if __name__ == "__main__":
    unittest.main()
