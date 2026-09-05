from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import train_sft  # noqa: E402


class TrainSftTests(unittest.TestCase):
    def test_nccl_defaults_are_set_without_overriding_explicit_values(self) -> None:
        with patch.dict(train_sft.os.environ, {}, clear=True):
            train_sft.configure_nccl_for_consumer_rtx()
            self.assertEqual(train_sft.os.environ["NCCL_P2P_DISABLE"], "1")
            self.assertEqual(train_sft.os.environ["NCCL_IB_DISABLE"], "1")
        with patch.dict(
            train_sft.os.environ,
            {"NCCL_P2P_DISABLE": "0", "NCCL_IB_DISABLE": "0"},
            clear=True,
        ):
            train_sft.configure_nccl_for_consumer_rtx()
            self.assertEqual(train_sft.os.environ["NCCL_P2P_DISABLE"], "0")
            self.assertEqual(train_sft.os.environ["NCCL_IB_DISABLE"], "0")

    def test_default_training_arguments_fit_a_single_24gb_gpu(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["train_sft.py", "--dataset", "dataset.jsonl", "--output", "output"],
        ):
            args = train_sft.parse_args()
        self.assertEqual(args.per_device_train_batch_size, 1)
        self.assertEqual(args.gradient_accumulation_steps, 8)

        class FakeTrainingArguments:
            def __init__(self, eval_strategy: str, **kwargs: object) -> None:
                self.kwargs = {"eval_strategy": eval_strategy, **kwargs}

        built = train_sft.training_arguments(FakeTrainingArguments, args, bf16=True, has_evaluation=False)
        self.assertTrue(built.kwargs["group_by_length"])
        self.assertEqual(built.kwargs["eval_strategy"], "no")

    def test_initial_adapter_is_parsed_and_must_be_a_completed_adapter(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "train_sft.py",
                "--dataset",
                "dataset.jsonl",
                "--output",
                "output",
                "--initial-adapter",
                "previous-adapter",
            ],
        ):
            args = train_sft.parse_args()
        self.assertEqual(args.initial_adapter, Path("previous-adapter"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text("", encoding="utf-8")
            output = root / "output"
            adapter = root / "adapter"
            adapter.mkdir()
            args.dataset = dataset
            args.output = output
            args.initial_adapter = adapter
            with self.assertRaisesRegex(ValueError, "not a completed PEFT adapter"):
                train_sft.validate_args(args)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            train_sft.validate_args(args)

    def test_initial_adapter_cannot_be_combined_with_checkpoint_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset.jsonl"
            dataset.write_text("", encoding="utf-8")
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            args = train_sft.argparse.Namespace(
                dataset=dataset,
                manifest=None,
                output=root / "output",
                initial_adapter=adapter,
                resume_from_checkpoint=root / "checkpoint-1",
                max_length=4096,
                epochs=1.0,
                learning_rate=1e-4,
                validation_runs=0,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8,
                lora_r=16,
                lora_alpha=32,
                logging_steps=10,
                save_total_limit=3,
                dataloader_workers=2,
                lora_dropout=0.05,
            )
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                train_sft.validate_args(args)

    def test_epoch_loss_summary_groups_logging_windows_by_epoch(self) -> None:
        result = train_sft.summarize_epoch_losses(
            [
                {"epoch": 0.25, "loss": 0.8},
                {"epoch": 1.0, "loss": 0.4},
                {"epoch": 1.25, "loss": 0.2},
                {"epoch": 2.0, "loss": 0.1},
                {"epoch": "bad", "loss": 0.0},
                {"epoch": 2.5, "loss": float("nan")},
            ]
        )
        self.assertEqual(
            result,
            [
                {"epoch": 1, "logged_windows": 2, "mean_logged_loss": 0.6000000000000001, "last_logged_loss": 0.4},
                {"epoch": 2, "logged_windows": 2, "mean_logged_loss": 0.15000000000000002, "last_logged_loss": 0.1},
            ],
        )

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
        self.assertEqual(routed["deck"][0]["id"], "Strike_R")
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

    def test_zero_validation_runs_keeps_every_sample_for_training(self) -> None:
        samples = [
            {"run_id": f"run-{run}", "sample_id": f"run-{run}:{step}"}
            for run in range(2)
            for step in range(2)
        ]
        train, validation, validation_ids = train_sft.split_by_run(samples, validation_runs=0, seed=3)
        self.assertEqual(train, samples)
        self.assertEqual(validation, [])
        self.assertEqual(validation_ids, [])

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
