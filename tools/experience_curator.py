"""DeepSeek-backed human-audited experience curation."""

from __future__ import annotations

import json
from typing import Any, Callable

from experience_knowledge import ExperienceError, get_experience, public_entity, validate_experience
from teacher_client import RemoteTeacherClient, TeacherError, display_text


CURATOR_PROMPT_VERSION = "sts1-experience-curator-zh-v1"


def _json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ExperienceError("teacher returned no valid JSON object")


def build_curator_messages(
    entity_type: str,
    entity: dict[str, Any],
    question: str,
    human_answer: str,
    *,
    max_entries: int = 3,
    context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    existing = get_experience(entity)
    system = (
        "你是《杀戮尖塔1》的知识库经验整理员。根据实体静态资料、已有经验和人类回答，"
        f"重写该实体的完整替换 experience 列表，最多 {max_entries} 条。经验必须是可执行、场景相关、"
        "不与静态事实冲突的中文建议；合并重复内容，删除过时或空泛内容。只返回 JSON："
        '{"experience":["..."]}。不要输出 Markdown、解释或额外字段。'
    )
    user_payload = {
        "entity_type": entity_type,
        "entity": public_entity(entity),
        "existing_experience": existing,
        "human_question": question,
        "human_answer": human_answer,
        "current_context": context or {},
        "requirements": ["输出完整替换列表而不是增量", f"列表长度 0-{max_entries}", "每条中文经验应能指导具体决策"],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
    ]


def curate_experience(
    client: RemoteTeacherClient,
    entity_type: str,
    entity: dict[str, Any],
    question: str,
    human_answer: str,
    *,
    max_entries: int = 3,
    context: dict[str, Any] | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Ask the configured DeepSeek teacher for a full replacement list."""
    if not question.strip() or not human_answer.strip():
        raise ExperienceError("question and human_answer must not be empty")
    result = client.complete(
        build_curator_messages(entity_type, entity, question, human_answer, max_entries=max_entries, context=context),
        prompt_version=CURATOR_PROMPT_VERSION,
        thinking=False,
        json_mode=True,
        streaming=False,
        on_update=on_update,
    )
    payload = _json_object(str(result.get("output") or ""))
    values = validate_experience(payload.get("experience", []), max_entries)
    return {"experience": values, "teacher": {key: result.get(key) for key in ("model_id", "latency_ms", "attempt", "usage", "finish_reason")}, "output": display_text(result.get("output", ""), 2000)}
