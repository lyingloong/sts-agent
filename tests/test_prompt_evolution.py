from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import agent_config  # noqa: E402
import prompt_evolution  # noqa: E402


class PromptEvolutionTests(unittest.TestCase):
    def make_manager(self) -> tuple[prompt_evolution.PromptEvolutionManager, Path]:
        root = Path(tempfile.mkdtemp())
        policy_root = root / "prompts" / "characters" / "IRONCLAD"
        (policy_root / "versions").mkdir(parents=True)
        (policy_root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": prompt_evolution.SCHEMA_VERSION,
                    "character": "IRONCLAD",
                    "base_prompt_version": "sts1-teacher-zh-v2",
                    "active_version": "p0001",
                    "next_version_index": 2,
                }
            ),
            encoding="utf-8",
        )
        (policy_root / "versions" / "p0001.json").write_text(
            json.dumps(
                {
                    "schema_version": prompt_evolution.SCHEMA_VERSION,
                    "version": "p0001",
                    "character": "IRONCLAD",
                    "parent_version": None,
                    "skills": {},
                }
            ),
            encoding="utf-8",
        )
        config = dict(agent_config.DEFAULT_CONFIG["prompt_evolution"])
        config["enabled"] = True
        config["policy_dir"] = "prompts/characters"
        teacher = dict(agent_config.DEFAULT_CONFIG["teacher"])
        manager = prompt_evolution.PromptEvolutionManager(root, config, teacher, "session-test")
        return manager, root

    @staticmethod
    def trace(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "step_id": 4,
                    "accepted": True,
                    "pre_state": {
                        "screen_type": "NONE",
                        "act": 1,
                        "floor": 2,
                        "current_hp": 50,
                        "max_hp": 80,
                        "combat": {"player": {"energy": 2}},
                        "recent_actions": [{"action": "结束回合"}],
                    },
                    "legal_actions": [
                        {"id": "A1", "label": "出牌 防御"},
                        {"id": "A2", "label": "结束回合"},
                    ],
                    "action": {"id": "A2", "label": "结束回合"},
                    "decision": {"source": "teacher_auto"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_candidate_is_not_active_before_acceptance(self) -> None:
        manager, root = self.make_manager()
        run_config = {"character": "IRONCLAD", "ascension": 0, "seed": "ABC123"}
        manager.begin_run(run_config)
        trace = root / "trace.jsonl"
        self.trace(trace)

        def complete(*_args, **_kwargs):
            return {
                "output": json.dumps(
                    {
                        "target_skill": "COMBAT",
                        "instruction": "在结束回合前先比较来袭伤害与可用防御，只有没有有价值的合法动作时才结束回合。",
                        "diagnosis": "可避免掉血",
                        "evidence_step_ids": [4],
                        "expected_effect": "减少空过",
                        "risk": "不要强行花光能量",
                    },
                    ensure_ascii=False,
                ),
                "model_id": "deepseek-v4-pro",
                "prompt_version": "test",
                "latency_ms": 1,
                "attempt": 1,
                "usage": {},
            }

        manager.remote.complete = complete  # type: ignore[method-assign]
        result = manager.complete_run(
            "run-1",
            run_config | {"prompt_evolution": {"role": "incumbent"}},
            trace,
            {"screen_type": "GAME_OVER", "act": 1, "floor": 2, "current_hp": 0, "max_hp": 80},
        )
        self.assertEqual(result["kind"], "start_candidate")
        self.assertEqual(manager.active_version, "p0001")
        self.assertEqual(manager.candidate["target_skill"], "COMBAT")

    def test_first_incumbent_may_start_without_explicit_seed(self) -> None:
        manager, _root = self.make_manager()
        metadata = manager.begin_run({"character": "IRONCLAD", "ascension": 0, "seed": ""})
        self.assertEqual(metadata["role"], "incumbent")

    def test_reject_requires_reason_and_accept_promotes_version(self) -> None:
        manager, _root = self.make_manager()
        manager.candidate = {
            "candidate_id": "c0001",
            "skills": {"COMBAT": "只在有威胁时先防御。"},
            "target_skill": "COMBAT",
            "instruction": "只在有威胁时先防御。",
        }
        manager.incumbent = {
            "run_config": {"character": "IRONCLAD", "ascension": 0, "seed": "ABC123"},
            "run_id": "incumbent",
            "policy": manager.active_policy,
            "metrics": {},
        }
        manager.candidate_result = {
            "run_config": {"character": "IRONCLAD", "ascension": 0, "seed": "ABC123"},
            "run_id": "candidate",
            "policy": manager.active_policy,
            "metrics": {},
        }
        self.assertEqual(manager.handle_review({"verdict": "reject", "reason": ""})["kind"], "error")
        # Disable continuation so acceptance does not require another fake API call.
        manager.auto_continue = False
        result = manager.handle_review({"verdict": "accept", "reason": "同一种子上更少空过。"})
        self.assertEqual(result["kind"], "accepted_paused")
        self.assertEqual(manager.active_version, "p0002")


if __name__ == "__main__":
    unittest.main()
