from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import experience_curator  # noqa: E402
import experience_knowledge  # noqa: E402
import communication_controller  # noqa: E402
import teacher_client  # noqa: E402
import teacher_prompts  # noqa: E402


class KnowledgeExperienceTests(unittest.TestCase):
    def test_nested_knowledge_request_is_parsed_without_action(self) -> None:
        action, request = teacher_client.parse_action_response(
            '{"knowledge_request":{"entity_type":"card","entity_id":"Corruption","question":"何时使用？"}}',
            {"A1", "A2"},
        )
        self.assertIsNone(action)
        self.assertEqual(request, {"entity_type": "card", "entity_id": "Corruption", "question": "何时使用？"})

    def test_action_and_knowledge_request_cannot_be_combined(self) -> None:
        with self.assertRaises(teacher_client.TeacherError):
            teacher_client.parse_action_response(
                '{"action_id":"A1","knowledge_request":{"entity_type":"card","entity_id":"X","question":"q"}}',
                {"A1"},
            )

    def test_curator_prompt_requires_full_replacement(self) -> None:
        entity = {"id": "TEST", "name": "Test", "description": "Deal damage.", "experience": ["旧经验"]}
        messages = experience_curator.build_curator_messages("card", entity, "问题", "回答")
        self.assertIn("完整替换 experience", messages[0]["content"])
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["existing_experience"], ["旧经验"])

    def test_curator_revision_includes_rejected_proposal_and_feedback(self) -> None:
        entity = {"id": "TEST", "name": "Test", "description": "Deal damage."}
        messages = experience_curator.build_curator_messages(
            "card",
            entity,
            "问题",
            "回答",
            previous_proposal=["第一条", "第二条"],
            reviewer_feedback="第二条不要绑定某个特定遗物。",
            revision=2,
        )
        self.assertIn("人工拒绝后的修订", messages[0]["content"])
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["revision"], 2)
        self.assertEqual(payload["previous_proposal"], ["第一条", "第二条"])
        self.assertEqual(payload["human_rejection_feedback"], "第二条不要绑定某个特定遗物。")

    def test_rejected_experience_is_recurated_with_feedback_until_accepted(self) -> None:
        class Bridge:
            def __init__(self) -> None:
                self.sent: list[dict] = []
                self.responses = iter(
                    [
                        {"type": "knowledge_experience_answer", "answer": "第一回合通常会造成伤害。"},
                        {"type": "knowledge_experience_review", "verdict": "reject", "reason": "第二条不要绑定当前遗物。"},
                        {"type": "knowledge_experience_review", "verdict": "accept"},
                    ]
                )

            def send(self, packet: dict) -> None:
                self.sent.append(packet)

            def receive(self) -> dict:
                return next(self.responses)

        class Teacher:
            model = "teacher-test"
            remote = object()

        bridge = Bridge()
        entity = {"id": "SnakePlant", "name": "蛇花", "description": "x"}
        proposals = [
            {"experience": ["第一版", "绑定贤者之石"], "teacher": {}},
            {"experience": ["第二版", "通用建议"], "teacher": {}},
        ]
        config = {"knowledge_experience": {"max_entries_per_entity": 3, "max_question_chars": 2000, "max_answer_chars": 6000}}
        with (
            patch.object(communication_controller, "visible_entity_references", return_value=[{"entity_type": "monster", "entity_id": "SnakePlant", "name": "蛇花"}]),
            patch.object(communication_controller, "find_entity", return_value=entity),
            patch.object(communication_controller, "curate_experience", side_effect=proposals) as curate,
            patch.object(communication_controller, "replace_experience", return_value=["第二版", "通用建议"]),
            patch.object(communication_controller, "append_audit"),
        ):
            result = communication_controller.request_knowledge_experience(
                bridge,
                Teacher(),
                {"in_game": True, "game_state": {"screen_type": "NONE", "room_phase": "COMBAT"}},
                {"entity_type": "monster", "entity_id": "SnakePlant", "question": "第一回合会做什么？"},
                [],
                config,
            )

        self.assertEqual(result["status"], "accepted")
        self.assertEqual(curate.call_count, 2)
        second = curate.call_args_list[1].kwargs
        self.assertEqual(second["previous_proposal"], ["第一版", "绑定贤者之石"])
        self.assertEqual(second["reviewer_feedback"], "第二条不要绑定当前遗物。")
        self.assertEqual(second["revision"], 2)
        reviews = [packet for packet in bridge.sent if packet.get("type") == "knowledge_experience_review_request"]
        self.assertEqual([packet["revision"] for packet in reviews], [1, 2])

    def test_knowledge_request_instruction_is_opt_in(self) -> None:
        ordinary = teacher_prompts.build_system_prompt({"screen_type": "COMBAT"})
        enabled = teacher_prompts.build_system_prompt({"screen_type": "COMBAT"}, allow_knowledge_request=True)
        normal = teacher_prompts.build_system_prompt(
            {"screen_type": "COMBAT"},
            allow_knowledge_request=True,
            knowledge_request_style="normal",
        )
        self.assertNotIn("knowledge_request", ordinary)
        self.assertIn("knowledge_request", enabled)
        self.assertIn("这是被鼓励使用的工具", enabled)
        self.assertIn("仍有不明白之处时可以提问", normal)
        self.assertIn("最终输出协议（优先级最高）", enabled)
        self.assertGreater(enabled.rfind("最终输出协议（优先级最高）"), enabled.rfind("【战斗决策】"))

    def test_knowledge_enabled_user_prompt_allows_either_response(self) -> None:
        prompt = teacher_client.build_prompt(
            {
                "screen_type": "EVENT",
                "screen_state": {
                    "event_id": "UPGRADE_SHRINE",
                    "event_name": "升级神龛",
                    "options": [{"label": "祈祷"}],
                },
            },
            [{"id": "A1", "label": "结束回合"}],
            allow_knowledge_request=True,
        )
        self.assertIn("either", prompt)
        self.assertIn('"knowledge_request"', prompt)
        self.assertIn("知识请求白名单", prompt)
        self.assertIn("event | UPGRADE_SHRINE", prompt)
        self.assertIn("不要把事件选项", prompt)
        self.assertNotIn('Return exactly one JSON object: {"action_id":"A1"}.', prompt)

    def test_event_static_knowledge_is_supported(self) -> None:
        event = experience_knowledge.find_entity("event", "WORLD_OF_GOOP")
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["name"], "World of Goop")
        self.assertIn("choices", event)

    def test_knowledge_allowlist_excludes_event_option_names(self) -> None:
        references = experience_knowledge.visible_entity_references(
            {
                "screen_type": "EVENT",
                "screen_state": {
                    "event_id": "UPGRADE_SHRINE",
                    "event_name": "升级神龛",
                    "options": [{"label": "祈祷"}, {"label": "离开"}],
                },
            }
        )
        self.assertEqual(references, [{"entity_type": "event", "entity_id": "UPGRADE_SHRINE", "name": "升级神龛"}])

    def test_replace_experience_is_bounded_and_removes_empty_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entities.json"
            path.write_text(json.dumps([{"id": "TEST", "name": "Test", "description": "x"}], ensure_ascii=False), encoding="utf-8")
            old = experience_knowledge.KNOWLEDGE_PATHS["card"]
            experience_knowledge.KNOWLEDGE_PATHS["card"] = path
            try:
                saved = experience_knowledge.replace_experience("card", "test", ["一", "二", "三"])
                self.assertEqual(saved, ["一", "二", "三"])
                self.assertEqual(json.loads(path.read_text(encoding="utf-8"))[0]["experience"], saved)
                with self.assertRaises(experience_knowledge.ExperienceError):
                    experience_knowledge.replace_experience("card", "test", ["1", "2", "3", "4"])
                experience_knowledge.replace_experience("card", "test", [])
                self.assertNotIn("experience", json.loads(path.read_text(encoding="utf-8"))[0])
            finally:
                experience_knowledge.KNOWLEDGE_PATHS["card"] = old


if __name__ == "__main__":
    unittest.main()
