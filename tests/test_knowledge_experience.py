from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

import experience_curator  # noqa: E402
import experience_knowledge  # noqa: E402
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
            {"screen_type": "COMBAT"},
            [{"id": "A1", "label": "结束回合"}],
            allow_knowledge_request=True,
        )
        self.assertIn("either", prompt)
        self.assertIn('"knowledge_request"', prompt)
        self.assertNotIn('Return exactly one JSON object: {"action_id":"A1"}.', prompt)

    def test_event_static_knowledge_is_supported(self) -> None:
        event = experience_knowledge.find_entity("event", "WORLD_OF_GOOP")
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["name"], "World of Goop")
        self.assertIn("choices", event)

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
