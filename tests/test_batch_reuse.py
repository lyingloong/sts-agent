from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_agent.commands import evaluate_baseline as batch
from sts_agent.evaluation.reuse import expected_fingerprint, find_reusable_runs, model_fingerprint

CONFIG = {"policy": {"model": {"base": "model-x", "revision": "rev-1", "initial_adapter": ""},
                     "api": {"provider": "openai_compatible", "base_url": "https://gateway.example/v1",
                             "temperature": 0, "max_output_tokens": 32,
                             "thinking": False, "prompt_format": "plain_text",
                             "prompt_template": "canonical", "extra_body": {}}}}
PROMPT_SHA = "a" * 64


def _manifest() -> dict:
    expected = expected_fingerprint(CONFIG, PROMPT_SHA)
    return {"model_manifest": {"model": expected["model"], "requested_revision": expected["requested_revision"],
                               "adapter": expected["adapter"], "prompt_builder": expected["prompt_builder"],
                               "prompt_code_sha256": expected["prompt_code_sha256"],
                               "evaluation_metric": expected["evaluation_metric"],
                               "settings": expected["settings"]}}


def _summary(seed: str, score: float) -> dict:
    return {"run_id": f"run-{seed}", "terminal": True, "won": False, "beat_heart": False, "human_actions": 0,
            "run_config": {"character": "IRONCLAD", "ascension": 0, "seed": seed, "internal_seed": 1},
            "evaluation": {"autonomous_eligible": True, "score_final": True, "progress_score": score,
                           "boss_acts_defeated": [], "highest_floor": 1, "highest_act": 1}}


def _row(seed: str, score: float) -> dict:
    return {"seed": seed, "status": "completed", "run_id": f"run-{seed}", "finished_at": "2026-01-01T00:00:00Z",
            "summary": _summary(seed, score), "api_performance": {"calls": 1}, "artifacts": {}}


def _write_batch(output_dir: Path, name: str, rows: list[dict], manifest: dict | None = None,
                 suite_id: str = "suite") -> Path:
    directory = output_dir / name
    directory.mkdir(parents=True)
    (directory / "results.json").write_text(json.dumps({"schema_version": 1, "suite_id": suite_id,
                                                        "status": "completed", "results": rows}), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(manifest or _manifest()), encoding="utf-8")
    return directory


def _suite(seeds: list[str]) -> dict:
    return {"schema_version": 1, "suite_id": "suite", "character": "IRONCLAD", "ascension": 0, "seeds": seeds,
            "limits": {"connection_timeout_seconds": 1, "idle_timeout_seconds": 1,
                       "run_timeout_seconds": 1, "max_actions_per_run": 10},
            "output_dir": "data/experiments/batches"}


class ReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "batches"
        self.output.mkdir()

    def test_discovers_matching_completed_runs_only(self) -> None:
        _write_batch(self.output, "one", [_row("A", 10.0), _row("B", 20.0)])
        other = _manifest()
        other["model_manifest"]["model"] = "different-model"
        _write_batch(self.output, "two", [_row("C", 30.0)], manifest=other)
        found, notes = find_reusable_runs(self.output, _suite(["A", "B", "C"]), expected_fingerprint(CONFIG, PROMPT_SHA))
        self.assertEqual(sorted(found), ["A", "B"])
        self.assertTrue(any(note["batch"].endswith("two") for note in notes))

    def test_prompt_hash_mismatch_is_not_reused(self) -> None:
        _write_batch(self.output, "one", [_row("A", 10.0)])
        found, notes = find_reusable_runs(self.output, _suite(["A"]), expected_fingerprint(CONFIG, "b" * 64))
        self.assertEqual(found, {})
        self.assertEqual(len(notes), 1)

    def test_newest_batch_wins_for_the_same_seed(self) -> None:
        older = _write_batch(self.output, "older", [_row("A", 10.0)])
        newer = _write_batch(self.output, "newer", [_row("A", 20.0)])
        older.touch()
        newer.touch()
        found, _ = find_reusable_runs(self.output, _suite(["A"]), expected_fingerprint(CONFIG, PROMPT_SHA))
        self.assertEqual(found["A"]["run_id"], "run-A")
        self.assertEqual(found["A"]["batch"], str(newer))
        self.assertEqual(found["A"]["summary"]["evaluation"]["progress_score"], 20.0)

    def test_alias_reuse_is_scoped_to_the_hosted_endpoint(self) -> None:
        previous = _manifest()
        previous["model_manifest"]["settings"]["base_url"] += "/"
        _write_batch(self.output, "same", [_row("A", 10)], manifest=previous)
        previous["model_manifest"]["settings"]["base_url"] = "https://other.example/v1"
        _write_batch(self.output, "different", [_row("B", 10)], manifest=previous)
        found, notes = find_reusable_runs(self.output, _suite(["A", "B"]), expected_fingerprint(CONFIG, PROMPT_SHA))
        self.assertEqual(list(found), ["A"])
        self.assertEqual(len(notes), 1)


class ReusedReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.suite = _suite(["A", "B", "C"])
        self.reused = {"A": {"seed": "A", "run_id": "run-A", "summary": _summary("A", 10.0),
                             "api_performance": {"calls": 1}, "artifacts": {}, "finished_at": "2026-01-01T00:00:00Z",
                             "batch": str(self.root / "old"), "suite_id": "old"}}

    def _report(self):
        return batch.BatchReport(self.root, self.suite, self.root / "suite.json", self.root / "agent.json",
                                 "127.0.0.1", 8765, clock=lambda: 1.0, reused=self.reused)

    def test_reused_rows_are_completed_and_counted(self) -> None:
        report = self._report()
        evaluation = json.loads((report.output_dir / "evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(report.rows[0]["status"], "completed")
        self.assertEqual(report.rows[0]["reused_from"]["batch"], str(self.root / "old"))
        self.assertEqual(report.rows[1]["status"], "pending")
        self.assertEqual(evaluation["eligible_runs"], 1)
        self.assertEqual(evaluation["reused_runs"], 1)
        self.assertEqual(evaluation["new_runs"], 0)

    def test_state_machine_skips_completed_rows(self) -> None:
        report = self._report()
        state = batch.BatchStateMachine(self.suite, report, "model-x", clock=lambda: 0.0)
        self.assertEqual(state.order, [1, 2])
        self.assertEqual(state.index, 1)


if __name__ == "__main__":
    unittest.main()
