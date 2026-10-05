from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sts_agent import config as agent_config
from sts_agent.learning import prompt_evolution


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
                    "base_prompt_version": "sts1-player-zh-v3",
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
                    "role_skill": "",
                    "scene_skills": {},
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
                        {"id": "A1", "kind": "play", "card_index": 1, "label": "出牌 防御"},
                        {"id": "A2", "label": "结束回合"},
                    ],
                    "action": {"id": "A2", "label": "结束回合"},
                    "command": "END",
                    "post_state": {
                        "screen_type": "NONE",
                        "act": 1,
                        "floor": 2,
                        "current_hp": 45,
                        "max_hp": 80,
                        "description": "runtime effect description",
                    },
                    "decision": {"source": "teacher_auto", "model_action": "A2", "model_candidates": [{"rank": 1, "action_id": "A2", "probability": 0.8}]},
                    "terminal": False,
                    "stable_after_action": True,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def test_evolution_ledger_keeps_before_after_dynamics_and_runtime_descriptions(self) -> None:
        _manager, root = self.make_manager()
        trace = root / "trace.jsonl"
        self.trace(trace)
        ledger = prompt_evolution.trace_ledger(trace)
        self.assertEqual(len(ledger), 1)
        item = ledger[0]
        self.assertEqual(item["action_detail"]["id"], "A2")
        self.assertEqual(item["command"], "END")
        self.assertEqual(item["state_after"]["current_hp"], 45)
        self.assertEqual(item["decision"]["model_candidates"][0]["probability"], 0.8)
        self.assertEqual(item["state_after"]["description"], "runtime effect description")

    def test_public_action_retains_runtime_description_and_omits_flavor(self) -> None:
        action = {"id": "A1", "description": "Deal 6 damage.", "flavor": "cosmetic", "flavor_text": "cosmetic"}
        self.assertEqual(prompt_evolution._public_action(action), {"id": "A1", "description": "Deal 6 damage."})

    def test_evolution_ledger_deduplicates_adjacent_identical_states(self) -> None:
        manager, root = self.make_manager()
        trace = root / "dedup-trace.jsonl"
        state = {"screen_type": "MAP", "act": 1, "floor": 2, "map": {"nodes": [{"x": 1}]}}
        records = []
        for step_id in (1, 2):
            records.append(
                {
                    "step_id": step_id,
                    "accepted": True,
                    "pre_state": state,
                    "legal_actions": [{"id": "A1", "kind": "choose", "label": "选择"}],
                    "action": {"id": "A1", "kind": "choose", "label": "选择"},
                    "command": "CHOOSE 0",
                    "post_state": state,
                    "decision": {"source": "teacher_auto"},
                    "terminal": False,
                    "stable_after_action": True,
                }
            )
        trace.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
        ledger = prompt_evolution.trace_ledger(trace)
        self.assertNotIn("state_after", ledger[0])
        self.assertEqual(ledger[0]["state_after_ref"], 2)
        self.assertIn("state_after", ledger[1])

    def test_analysis_ledger_splits_by_act_and_compacts_automatic_steps(self) -> None:
        ledger = [
            {
                "step_id": 1,
                "source": "environment_auto",
                "screen_type": "COMBAT_REWARD",
                "screen_after": "COMBAT_REWARD",
                "action": "自动拾取金币",
                "state": {"act": 1, "floor": 3, "deck": [{"id": "Strike_R", "count": 5}]},
            },
            {
                "step_id": 2,
                "source": "teacher_auto",
                "screen_type": "COMBAT",
                "screen_after": "NONE",
                "action": "结束回合",
                "action_detail": {"id": "END"},
                "command": "END",
                "legal_actions": [{"id": "END", "label": "结束回合", "description": "static"}],
                "decision": {"source": "teacher_auto"},
                "state": {"act": 1, "floor": 3, "map": {"nodes": list(range(100))}, "deck": [{"id": "Strike_R", "count": 5}]},
                "state_after_ref": 3,
            },
            {
                "step_id": 3,
                "source": "teacher_auto",
                "screen_type": "COMBAT",
                "screen_after": "NONE",
                "action": "结束回合",
                "action_detail": {"id": "END"},
                "command": "END",
                "legal_actions": [],
                "decision": {"source": "teacher_auto"},
                "state": {"act": 2, "floor": 1, "deck": [{"id": "Strike_R", "count": 5}]},
            },
        ]
        segments = prompt_evolution.split_ledger_by_act(ledger)
        self.assertEqual([act for act, _items in segments], ["1", "2"])
        first = segments[0][1]
        self.assertTrue(first[0]["auto"])
        self.assertNotIn("legal_actions", first[0])
        self.assertNotIn("map", first[1]["state"])
        self.assertIn("deck", first[1]["state"])
        self.assertEqual(segments[1][1][0]["step_id"], 3)

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
        self.assertEqual(result["kind"], "candidate_review")
        self.assertEqual(result["review"]["phase"], "candidate")
        self.assertEqual(manager.active_version, "p0001")
        self.assertEqual(manager.candidate["target_skill"], "COMBAT")
        self.assertIn("role_skill", manager.candidate)
        self.assertIn("scene_skills", manager.candidate)
        self.assertNotIn("skills", manager.candidate)
        approved = manager.handle_review({"verdict": "accept", "reason": "候选修改范围清楚，先验证。"})
        self.assertEqual(approved["kind"], "start_candidate")

    def test_candidate_review_rejection_generates_another_review_without_running(self) -> None:
        manager, root = self.make_manager()
        run_config = {"character": "IRONCLAD", "ascension": 0, "seed": "ABC123"}
        manager.begin_run(run_config)
        trace = root / "trace.jsonl"
        self.trace(trace)

        responses = iter(
            [
                {"target_skill": "COMBAT", "skill": "候选一", "diagnosis": "d", "evidence_step_ids": [4], "expected_effect": "e", "risk": "r"},
                {"target_skill": "COMBAT", "skill": "候选二", "diagnosis": "d2", "evidence_step_ids": [4], "expected_effect": "e2", "risk": "r2"},
            ]
        )

        def complete(*_args, **_kwargs):
            value = next(responses)
            return {"output": json.dumps(value, ensure_ascii=False), "model_id": "teacher", "prompt_version": "test", "latency_ms": 1, "attempt": 1, "usage": {}}

        manager.remote.complete = complete  # type: ignore[method-assign]
        first = manager.complete_run("run-1", run_config | {"prompt_evolution": {"role": "incumbent"}}, trace, {"screen_type": "GAME_OVER", "act": 1, "floor": 2})
        self.assertEqual(first["kind"], "candidate_review")
        revised = manager.handle_review({"verdict": "reject", "reason": "证据不足，改成更具体的战斗条件。"})
        self.assertEqual(revised["kind"], "candidate_review")
        self.assertEqual(revised["review"]["phase"], "candidate")
        self.assertIsNone(manager.candidate_result)

    def test_first_incumbent_may_start_without_explicit_seed(self) -> None:
        manager, _root = self.make_manager()
        metadata = manager.begin_run({"character": "IRONCLAD", "ascension": 0, "seed": ""})
        self.assertEqual(metadata["role"], "incumbent")

    def test_analysis_budget_defaults_are_large_and_explicit(self) -> None:
        config = agent_config.DEFAULT_CONFIG["prompt_evolution"]
        self.assertEqual(config["max_analysis_input_chars"], 900000)
        self.assertEqual(config["max_analysis_tokens"], 8192)

    def test_skill_text_segments_keep_complete_skills_and_mark_changes(self) -> None:
        original, candidate = prompt_evolution.skill_text_segments(
            "先计算来袭伤害。能防住时再出牌。",
            "先计算来袭伤害。优先防住后再出牌。",
        )
        self.assertEqual("".join(text for text, _kind in original), "先计算来袭伤害。能防住时再出牌。")
        self.assertEqual("".join(text for text, _kind in candidate), "先计算来袭伤害。优先防住后再出牌。")
        self.assertIn(("能", "removed"), original)
        self.assertIn(("优先", "added"), candidate)

    def test_analysis_budget_truncation_keeps_request_valid_json(self) -> None:
        manager, _root = self.make_manager()
        manager.max_analysis_input_chars = 4096
        run = {
            "run_id": "run-1",
            "run_config": {"character": "IRONCLAD", "ascension": 0, "seed_source": "random"},
            "terminal_state": {"screen_type": "GAME_OVER", "act": 1, "floor": 2},
            "metrics": {"score": 0},
            "policy": manager.active_policy,
            "ledger": [
                {"step_id": index, "source": "teacher_auto", "screen_type": "COMBAT", "state": {"act": 1, "floor": 1, "recent_actions": ["x" * 500]}}
                for index in range(40)
            ],
        }
        messages = manager._candidate_messages(run, analysis_ledger=prompt_evolution.compact_analysis_ledger(run["ledger"]), analysis_act="1")
        parsed = json.loads(messages[1]["content"])
        self.assertTrue(manager.last_input_truncated)
        self.assertIn("analysis_warning", parsed)
        self.assertLessEqual(len(messages[1]["content"]), manager.max_analysis_input_chars)

    def test_action_prompt_uses_player_persona_and_player_base_version(self) -> None:
        manager, _root = self.make_manager()
        manager.begin_run({"character": "IRONCLAD", "ascension": 0, "seed": "ABC123"})
        prompt, version = manager.system_prompt({"screen_type": "MAP", "class": "IRONCLAD"})
        self.assertIn("正在亲自游玩", prompt)
        self.assertNotIn("远程教师", prompt)
        self.assertEqual(version, f"{prompt_evolution.PROMPT_VERSION}:p0001")

    def test_reject_requires_reason_and_accept_promotes_version(self) -> None:
        manager, _root = self.make_manager()
        manager.candidate = {
            "candidate_id": "c0001",
            "role_skill": "",
            "scene_skills": {"COMBAT": "只在有威胁时先防御。"},
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
