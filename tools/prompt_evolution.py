"""Human-gated, single-seed evolution of role and scene skills.

The stable protocol prompt and game-action compiler remain controller-owned;
a generated candidate must pass a human pre-validation gate before it can
execute, and only a second human gate after the comparison run can promote it
to a versioned role/scene skill.
"""

from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from posthoc_feedback import POSTHOC_STATIC_FIELDS, _posthoc_state_snapshot, compact_text, terminal_summary
from teacher_client import RemoteTeacherClient, TeacherError
from teacher_prompts import PLAYER_PROMPT_VERSION as BASE_PROMPT_VERSION
from teacher_prompts import CHARACTER_KEYS, SCENARIO_PROMPTS, build_system_prompt, character_key, character_prompt, scenario_key


SCHEMA_VERSION = "sts-prompt-evolution-0.2"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"sts-prompt-evolution-0.1", SCHEMA_VERSION})
PROMPT_VERSION = "sts-prompt-evolution-v2"
SKILL_KEYS = frozenset({"ROLE", *SCENARIO_PROMPTS.keys()})
ACTION_SOURCES = frozenset({"teacher_auto", "model_auto", "online_policy_sample", "human_override"})


class PromptEvolutionError(RuntimeError):
    """A recoverable prompt-evolution error that must not end a game run."""


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PromptEvolutionError(f"cannot read {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PromptEvolutionError(f"invalid {label} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PromptEvolutionError(f"{label} must be a JSON object: {path}")
    return value


def _compact_skill(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise PromptEvolutionError("candidate instruction must not be empty")
    if len(text) > limit:
        raise PromptEvolutionError(f"candidate instruction exceeds configured {limit} characters")
    return text


def skill_text_segments(previous: Any, candidate: Any) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return complete original/candidate text split into change-aware spans.

    The first list is for the original skill and marks removed text as
    ``removed``. The second is for the candidate and marks added text as
    ``added``. Character-level matching keeps the surrounding sentence
    readable when a teacher revises only a phrase in a Chinese skill.
    """

    before = str(previous or "").strip()
    after = str(candidate or "").strip()

    def append(segments: list[tuple[str, str]], text: str, kind: str) -> None:
        if not text:
            return
        if segments and segments[-1][1] == kind:
            segments[-1] = (segments[-1][0] + text, kind)
        else:
            segments.append((text, kind))

    original: list[tuple[str, str]] = []
    revised: list[tuple[str, str]] = []
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    for opcode, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if opcode == "equal":
            append(original, before[old_start:old_end], "unchanged")
            append(revised, after[new_start:new_end], "unchanged")
        elif opcode == "delete":
            append(original, before[old_start:old_end], "removed")
        elif opcode == "insert":
            append(revised, after[new_start:new_end], "added")
        elif opcode == "replace":
            append(original, before[old_start:old_end], "removed")
            append(revised, after[new_start:new_end], "added")
    return original, revised


def _extract_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match is None:
            raise PromptEvolutionError("teacher candidate is not valid JSON") from None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise PromptEvolutionError("teacher candidate is not valid JSON") from exc
    if not isinstance(value, dict):
        raise PromptEvolutionError("teacher candidate must be a JSON object")
    return value


def _metric(terminal_state: dict[str, Any]) -> dict[str, Any]:
    def integer(name: str) -> int:
        try:
            return int(terminal_state.get(name, 0) or 0)
        except (TypeError, ValueError):
            return 0

    current_hp = integer("current_hp")
    max_hp = integer("max_hp")
    screen = str(terminal_state.get("screen_type") or "UNKNOWN").upper()
    won = screen == "VICTORY"
    act = integer("act")
    floor = integer("floor")
    hp_fraction = current_hp / max_hp if max_hp > 0 else 0.0
    score = (5000.0 if won else 0.0) + act * 600.0 + floor * 30.0 + hp_fraction * 2.0
    return {
        "screen_type": screen,
        "won": won,
        "act": act,
        "floor": floor,
        "current_hp": current_hp,
        "max_hp": max_hp,
        "hp_fraction": round(hp_fraction, 4),
        "gold": integer("gold"),
        "score": round(score, 4),
    }


def _same_start(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return all(left.get(key) == right.get(key) for key in ("character", "ascension", "seed"))


def _evidence_steps(run: dict[str, Any], step_ids: Any) -> list[dict[str, Any]]:
    wanted = {item for item in step_ids if isinstance(item, int)} if isinstance(step_ids, list) else set()
    result: list[dict[str, Any]] = []
    for item in run.get("ledger", []):
        if not isinstance(item, dict) or item.get("step_id") not in wanted:
            continue
        result.append(
            {
                "step_id": item.get("step_id"),
                "screen_type": item.get("screen_type"),
                "source": item.get("source"),
                "action": item.get("action"),
                "legal_actions": item.get("legal_actions", []),
            }
        )
    return result


def _public_action(value: Any) -> dict[str, Any]:
    """Keep action mechanics useful for analysis without static KB prose."""

    if not isinstance(value, dict):
        return {}
    result = {}
    # Semantic actions are deliberately scalar records. Retain every scalar
    # parameter (including future action fields) while excluding any nested
    # payload that could reintroduce static item knowledge.
    for key, item in value.items():
        if key in POSTHOC_STATIC_FIELDS:
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            result[key] = compact_text(item, 320) if isinstance(item, str) else item
    return result


def _public_decision(value: Any) -> dict[str, Any]:
    """Expose selection/probability metadata, never teacher chain-of-thought."""

    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("source", "mode", "model_action", "executed_action", "overridden", "confidence", "assist_status"):
        item = value.get(key)
        if isinstance(item, (str, int, float, bool)) or item is None:
            result[key] = item
    candidates = value.get("model_candidates")
    if isinstance(candidates, list):
        result["model_candidates"] = [
            {
                key: item.get(key)
                for key in ("rank", "action_id", "label", "probability", "score")
                if isinstance(item, dict) and (isinstance(item.get(key), (str, int, float, bool)) or item.get(key) is None)
            }
            for item in candidates[:8]
            if isinstance(item, dict)
        ]
    return result


def _analysis_state(value: Any, *, include_inventory: bool) -> dict[str, Any]:
    """Make a compact review snapshot while retaining decision-critical dynamics."""

    if not isinstance(value, dict):
        return {}
    result = dict(value)
    # The map is a large route graph and the static Boss preview is not useful
    # for post-run skill attribution. Both are already available to the live
    # action model and are intentionally excluded from this teacher request.
    result.pop("map", None)
    result.pop("act_boss_profile", None)
    if not include_inventory:
        for key in ("deck", "relics", "potions"):
            result.pop(key, None)
    combat = result.get("combat")
    if isinstance(combat, dict):
        combat = dict(combat)
        monsters = combat.get("monsters")
        if isinstance(monsters, list):
            compact_monsters = []
            for monster in monsters:
                if not isinstance(monster, dict):
                    continue
                compact_monster = dict(monster)
                # Static move tables are not needed to judge an executed step;
                # live HP, intent, damage, and Powers remain in the snapshot.
                compact_monster.pop("knowledge", None)
                compact_monsters.append(compact_monster)
            combat["monsters"] = compact_monsters
        result["combat"] = combat
    return result


def compact_analysis_ledger(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reduce one Act's ledger to the public facts needed for skill review.

    The full ledger remains on disk. This representation is only the payload
    sent to the remote reviewer, so environment/controller actions are a
    one-line boundary marker and unchanged inventory is not repeated at every
    step.
    """

    result: list[dict[str, Any]] = []
    previous_inventory: str | None = None
    for item in ledger:
        if not isinstance(item, dict):
            continue
        state = item.get("state") if isinstance(item.get("state"), dict) else {}
        inventory = {key: state.get(key) for key in ("deck", "relics", "potions")}
        inventory_key = json.dumps(inventory, ensure_ascii=False, sort_keys=True, default=str)
        source = str(item.get("source") or "environment_auto")
        compact: dict[str, Any] = {
            "step_id": item.get("step_id"),
            "act": state.get("act"),
            "floor": state.get("floor"),
            "screen_type": item.get("screen_type"),
            "screen_after": item.get("screen_after"),
            "source": source,
        }
        if source not in ACTION_SOURCES:
            compact["action"] = compact_text(item.get("action") or "自动推进", 120)
            compact["auto"] = True
            result.append(compact)
            continue
        compact.update(
            {
                "action": compact_text(item.get("action"), 180),
                "action_detail": item.get("action_detail", {}),
                "command": compact_text(item.get("command"), 160),
                "legal_actions": [
                    {
                        key: value
                        for key, value in _public_action(action).items()
                        if key in {"id", "kind", "label", "command", "target", "card_index", "slot", "index"}
                    }
                    for action in item.get("legal_actions", [])
                    if isinstance(action, dict)
                ],
                "decision": _public_decision(item.get("decision")),
                "state": _analysis_state(state, include_inventory=inventory_key != previous_inventory),
            }
        )
        state_after = item.get("state_after")
        if isinstance(state_after, dict) and (item.get("terminal") or item.get("screen_after") != item.get("screen_type")):
            compact["state_after"] = _analysis_state(state_after, include_inventory=True)
        elif item.get("state_after_ref") is not None:
            compact["state_after_ref"] = item.get("state_after_ref")
        if item.get("terminal"):
            compact["terminal"] = True
        result.append(compact)
        previous_inventory = inventory_key
    return result


def split_ledger_by_act(ledger: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group an audit ledger into Act-sized teacher analysis segments."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for item in ledger:
        if not isinstance(item, dict):
            continue
        state = item.get("state") if isinstance(item.get("state"), dict) else {}
        raw_act = state.get("act")
        if raw_act is None and isinstance(item.get("state_after"), dict):
            raw_act = item["state_after"].get("act")
        act = str(raw_act if raw_act is not None else "unknown")
        if act not in grouped:
            grouped[act] = []
            order.append(act)
        grouped[act].append(item)
    order.sort(key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
    return [(act, compact_analysis_ledger(grouped[act])) for act in order]


def trace_ledger(trace_path: Path) -> list[dict[str, Any]]:
    """Return a detailed, bounded public ledger for evolution analysis."""

    result: list[dict[str, Any]] = []
    map_ids: dict[str, str] = {}
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PromptEvolutionError(f"cannot read run trace: {trace_path}") from exc
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("accepted") is not True:
            continue
        action = record.get("action") if isinstance(record.get("action"), dict) else {}
        decision = record.get("decision") if isinstance(record.get("decision"), dict) else {}
        pre_state = record.get("pre_state") if isinstance(record.get("pre_state"), dict) else {}
        legal_actions = record.get("legal_actions")
        legal_actions = legal_actions if isinstance(legal_actions, list) else []
        source = str(decision.get("source") or "environment_auto")
        post_state = record.get("post_state") if isinstance(record.get("post_state"), dict) else {}
        result.append(
            {
                "step_id": record.get("step_id"),
                "screen_type": pre_state.get("screen_type"),
                "screen_after": post_state.get("screen_type"),
                "source": source,
                "policy_action": source in ACTION_SOURCES,
                "action": compact_text(action.get("label") or action.get("command") or action.get("id"), 300),
                "action_detail": _public_action(action),
                "command": compact_text(record.get("command"), 320),
                "legal_actions": [_public_action(item) for item in legal_actions if isinstance(item, dict)],
                "decision": _public_decision(decision),
                "state": _posthoc_state_snapshot(pre_state, map_ids),
                "state_after": _posthoc_state_snapshot(post_state, map_ids),
                "terminal": bool(record.get("terminal")),
                "stable_after_action": record.get("stable_after_action"),
            }
        )
    # An acknowledgement's post-state is normally the next decision's
    # pre-state. Keep an explicit reference in that common case instead of
    # sending the same large observation twice. Retain the full post-state
    # for transitions, unstable animations, and the terminal action.
    for index, item in enumerate(result[:-1]):
        next_item = result[index + 1]
        if (
            item.get("state_after") == next_item.get("state")
            and item.get("stable_after_action") is not False
            and not item.get("terminal")
        ):
            item["state_after_ref"] = next_item.get("step_id")
            item.pop("state_after", None)
    return result


class PromptEvolutionManager:
    """Own prompt versions, ephemeral candidates, and one-controller run flow."""

    def __init__(self, project_root: Path, config: dict[str, Any], teacher_config: dict[str, Any], session_id: str) -> None:
        self.project_root = project_root
        self.config = config
        self.enabled = bool(config["enabled"])
        self.session_id = session_id
        self.library_root = project_root / str(config["policy_dir"])
        self.policy_root: Path | None = None
        self.version_root: Path | None = None
        self.manifest_path: Path | None = None
        self.character: str | None = None
        self.output_root = project_root / str(config["output_dir"])
        self.session_root = self.output_root / "sessions" / session_id
        self.max_skill_chars = int(config["max_skill_chars"])
        self.max_analysis_input_chars = int(config["max_analysis_input_chars"])
        self.max_analysis_tokens = int(config["max_analysis_tokens"])
        self.max_revisions_per_cycle = int(config["max_revisions_per_cycle"])
        self.auto_continue = bool(config["auto_continue"])
        self.remote = RemoteTeacherClient(project_root, teacher_config)
        self.last_error: str | None = None
        self.manifest: dict[str, Any] = {}
        self.active_policy: dict[str, Any] = {}
        self.current_policy: dict[str, Any] = {}
        self.incumbent: dict[str, Any] | None = None
        self.candidate: dict[str, Any] | None = None
        self.candidate_result: dict[str, Any] | None = None
        self.revision_count = 0
        self.candidate_index = 0
        self.stage = "等待开始游戏"
        self.current_role = "-"
        self.last_input_chars = 0
        self.last_input_original_chars = 0
        self.last_input_truncated = False
        # Review one Act at a time so a long run never becomes one oversized
        # teacher request. The full ledger remains available on disk.
        self.pending_analysis_segments: list[tuple[str, list[dict[str, Any]]]] = []
        self.current_analysis_act: str | None = None
        self.current_analysis_ledger: list[dict[str, Any]] | None = None
        # The run's selected character is not known until START, so its
        # isolated prompt library is loaded by begin_run instead of __init__.

    def _load_layout(self) -> None:
        if self.manifest_path is None:
            raise PromptEvolutionError("prompt policy character library is not selected")
        self.manifest = _read_object(self.manifest_path, "prompt policy manifest")
        if self.manifest.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
            raise PromptEvolutionError("prompt policy manifest has an unsupported schema_version")
        if self.manifest.get("character") != self.character:
            raise PromptEvolutionError("prompt policy manifest character does not match the active run")
        active_version = str(self.manifest.get("active_version") or "")
        if not re.fullmatch(r"p\d{4}", active_version):
            raise PromptEvolutionError("prompt policy manifest active_version must be pNNNN")
        self.active_policy = self._load_version(active_version)
        self.current_policy = self.active_policy

    def _load_version(self, version: str) -> dict[str, Any]:
        if self.version_root is None:
            raise PromptEvolutionError("prompt policy character library is not selected")
        policy = _read_object(self.version_root / f"{version}.json", "prompt policy version")
        if policy.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS or policy.get("version") != version:
            raise PromptEvolutionError(f"prompt policy version is invalid: {version}")
        if policy.get("character") != self.character:
            raise PromptEvolutionError(f"prompt policy {version} character does not match the active run")
        role_skill = policy.get("role_skill")
        if role_skill is not None and not isinstance(role_skill, str):
            raise PromptEvolutionError(f"prompt policy {version} role_skill must be a string")
        scene_skills = policy.get("scene_skills")
        if scene_skills is not None:
            if not isinstance(scene_skills, dict):
                raise PromptEvolutionError(f"prompt policy {version} scene_skills must be an object")
            for key, value in scene_skills.items():
                if key not in SCENARIO_PROMPTS or not isinstance(value, str):
                    raise PromptEvolutionError(f"prompt policy {version} contains invalid scene skill {key!r}")
        # Versions written before 0.2 used one mixed dictionary.  Read it so
        # existing character libraries remain usable, but all new versions
        # are written with the explicit role/scene structure below.
        legacy = policy.get("skills")
        if legacy is not None:
            if not isinstance(legacy, dict):
                raise PromptEvolutionError(f"prompt policy {version} legacy skills must be an object")
            for key, value in legacy.items():
                if key not in {"GLOBAL", *SKILL_KEYS} or not isinstance(value, str):
                    raise PromptEvolutionError(f"prompt policy {version} contains invalid legacy skill {key!r}")
        return policy

    def _effective_skills(self, policy: dict[str, Any]) -> tuple[str, dict[str, str]]:
        """Resolve a complete role/scene skill set for one policy version."""

        role_skill = str(policy.get("role_skill") or "").strip()
        scene_skills = {
            key: value
            for key, value in SCENARIO_PROMPTS.items()
            if isinstance(value, str) and value.strip()
        }
        raw_scenes = policy.get("scene_skills")
        if isinstance(raw_scenes, dict):
            scene_skills.update(
                {key: str(value).strip() for key, value in raw_scenes.items() if key in SCENARIO_PROMPTS and str(value).strip()}
            )
        legacy = policy.get("skills")
        if isinstance(legacy, dict):
            if not role_skill:
                role_skill = str(legacy.get("ROLE") or legacy.get("GLOBAL") or "").strip()
            scene_skills.update(
                {key: str(legacy[key]).strip() for key in SCENARIO_PROMPTS if str(legacy.get(key) or "").strip()}
            )
        if not role_skill and self.character:
            role_skill = character_prompt(self.character).strip()
        return role_skill, scene_skills

    @property
    def active_version(self) -> str:
        return str(self.active_policy.get("version") or "-")

    @property
    def current_policy_id(self) -> str:
        return str(self.current_policy.get("candidate_id") or self.current_policy.get("version") or "-")

    def preflight(self) -> str | None:
        if not self.enabled:
            return None
        error = self.remote.preflight()
        self.last_error = error
        return error

    def _select_character_library(self, value: Any) -> str:
        character = character_key(value)
        if character is None or character not in CHARACTER_KEYS:
            raise PromptEvolutionError("prompt evolution requires IRONCLAD, THE_SILENT, DEFECT, or WATCHER")
        if self.character == character and self.manifest_path is not None:
            return character
        if self.candidate is not None or self.candidate_result is not None:
            raise PromptEvolutionError("cannot switch character while a prompt candidate is pending review")
        self.character = character
        self.policy_root = self.library_root / character
        self.version_root = self.policy_root / "versions"
        self.manifest_path = self.policy_root / "manifest.json"
        self._load_layout()
        return character

    def begin_run(self, run_config: dict[str, Any]) -> dict[str, Any]:
        """Choose incumbent or prepared candidate for the next fixed-seed run."""

        if not self.enabled:
            return {}
        character = self._select_character_library(run_config.get("character"))
        # The first incumbent may start from a random game seed. The
        # controller records the seed exposed by the game before proposing a
        # candidate; candidate/revision runs are then restarted on that seed.
        seed = str(run_config.get("seed") or "").strip()
        if not seed and self.candidate is not None:
            raise PromptEvolutionError("candidate prompt runs require the incumbent game seed")
        if self.candidate is not None and self.candidate_result is None:
            self.current_policy = self.candidate
            role = "candidate" if self.revision_count == 0 else "revision"
            self.current_role = role
            self.stage = f"正在运行 {role}：{self.current_policy_id}"
        else:
            self.current_policy = self.active_policy
            role = "incumbent"
            self.current_role = role
            self.incumbent = None
            self.candidate_result = None
            self.revision_count = 0
            self.stage = f"正在运行 incumbent：{self.current_policy_id}"
        metadata = {
            "role": role,
            "character_library": character,
            "policy_id": self.current_policy_id,
            "base_version": self.active_version,
            "session_id": self.session_id,
        }
        self._write_session_state({"status": "running", "run": metadata, "run_config": run_config})
        return metadata

    def status(self) -> dict[str, Any]:
        """Return a small UI-safe snapshot of the current evolution stage."""

        return {
            "stage": self.stage,
            "character": self.character or "-",
            "active_version": self.active_version,
            "policy_id": self.current_policy_id,
            "role": self.current_role,
            "candidate_id": (self.candidate or {}).get("candidate_id") if isinstance(self.candidate, dict) else None,
            "target_skill": (self.candidate or {}).get("target_skill") if isinstance(self.candidate, dict) else None,
            "revision": self.revision_count,
            "analysis_act": self.current_analysis_act or "-",
            "pending_analysis_acts": [act for act, _ledger in self.pending_analysis_segments],
            "analysis_input_chars": self.last_input_chars,
            "analysis_input_original_chars": self.last_input_original_chars,
            "analysis_input_limit_chars": self.max_analysis_input_chars,
            "analysis_input_truncated": self.last_input_truncated,
            "error": self.last_error,
        }

    def system_prompt(
        self,
        observation: dict[str, Any],
        *,
        allow_knowledge_request: bool = False,
        knowledge_request_style: str = "encouraged",
    ) -> tuple[str, str]:
        """Compose the player prompt from the active role and scene skills."""

        if not self.enabled:
            return (
                build_system_prompt(
                    observation,
                    character=self.character,
                    persona="player",
                    allow_knowledge_request=allow_knowledge_request,
                    knowledge_request_style=knowledge_request_style,
                ),
                BASE_PROMPT_VERSION,
            )
        role_skill, scene_skills = self._effective_skills(self.current_policy)
        return (
            build_system_prompt(
                observation,
                character=self.character,
                persona="player",
                role_skill=role_skill,
                scene_skill=scene_skills.get(scenario_key(observation), ""),
                allow_knowledge_request=allow_knowledge_request,
                knowledge_request_style=knowledge_request_style,
            ),
            f"{PROMPT_VERSION}:{self.current_policy_id}",
        )

    def _candidate_messages(
        self,
        run: dict[str, Any],
        *,
        analysis_ledger: list[dict[str, Any]] | None = None,
        analysis_act: str | None = None,
        rejected_candidate: dict[str, Any] | None = None,
        rejection_reason: str = "",
    ) -> list[dict[str, str]]:
        keys = ", ".join(sorted(SKILL_KEYS))
        system = (
            "你是《杀戮尖塔》Prompt 演化器。根据一个 Act 的公开动作账本与终局结果，"
            "提出一个可审核的、范围明确的 skill 修改。你不是行动模型，也不能写游戏命令、代码、"
            "硬合法性限制或完整重写通用协议。只修改一个 role skill 或 scene skill；输出该 skill 的完整新文本。"
            "如果人工拒绝理由明确指出另一个场景，target_skill 可以切换到该场景；如果理由涉及多个问题，"
            "只选择证据最充分的一个，本轮仍只改一个 skill。"
            "不能提及当前种子、具体楼层号、具体卡牌/敌人名称，除非其是通用机制示例。"
            "不要把一次终局结果平摊归因给所有动作；只引用账本中有清楚证据的 1-8 个步骤。"
            "账本中若出现 state_after_ref，表示动作后的状态与所指向下一步的 state 完全相同；只有状态变化或终局才会内联 state_after。"
            "输出严格 JSON，不要 Markdown 或推理："
            '{"target_skill":"COMBAT","skill":"...","diagnosis":"...",'
            '"evidence_step_ids":[4,8],"expected_effect":"...","risk":"..."}。'
            f"当前角色为 {self.character}。target_skill 必须是以下之一：{keys}。"
            "ROLE 是角色专属 skill；其他键是场景 skill；skill 将直接替换目标 skill 的当前版本。"
        )
        terminal_state = run.get("terminal_state") if isinstance(run.get("terminal_state"), dict) else {}
        ledger = analysis_ledger if analysis_ledger is not None else run.get("analysis_ledger", run.get("ledger", []))
        ledger = ledger if isinstance(ledger, list) else []
        observed_scenes = {
            scenario_key(item.get("state", {}))
            for item in ledger
            if isinstance(item, dict) and isinstance(item.get("state"), dict)
        }
        role_skill, scene_skills = self._effective_skills(run["policy"])
        source_counts: dict[str, int] = {}
        for item in ledger:
            if isinstance(item, dict):
                source = str(item.get("source") or "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
        first_state = next(
            (item.get("state") for item in ledger if isinstance(item, dict) and isinstance(item.get("state"), dict)),
            {},
        )
        user = {
            "base_prompt_version": BASE_PROMPT_VERSION,
            "run_id": run.get("run_id"),
            "analysis_scope": {
                "act": analysis_act,
                "accepted_steps": len(ledger),
                "first_step_id": ledger[0].get("step_id") if ledger else None,
                "last_step_id": ledger[-1].get("step_id") if ledger else None,
            },
            "incumbent_policy": {
                "version": run["policy"].get("version"),
                "role_skill": role_skill,
                "scene_skills": {key: scene_skills[key] for key in sorted(observed_scenes) if key in scene_skills},
                "available_skill_keys": sorted(SKILL_KEYS),
            },
            "run_config": {
                key: run["run_config"].get(key)
                for key in ("character", "ascension", "seed_source", "collection_mode")
                if key in run["run_config"]
            },
            "terminal": terminal_summary(terminal_state),
            "metrics": run["metrics"],
            "trace_summary": {
                "accepted_steps": len(ledger),
                "action_sources": source_counts,
                "first_step_id": ledger[0].get("step_id") if ledger else None,
                "last_step_id": ledger[-1].get("step_id") if ledger else None,
            },
            "act_context": {
                "initial_deck": first_state.get("deck") if isinstance(first_state, dict) else [],
                "initial_relics": first_state.get("relics") if isinstance(first_state, dict) else [],
                "initial_potions": first_state.get("potions") if isinstance(first_state, dict) else [],
            },
            "action_ledger": ledger,
        }
        if rejected_candidate is not None:
            user["rejected_candidate"] = {
                key: rejected_candidate.get(key)
                for key in ("candidate_id", "target_skill", "skill", "instruction", "diagnosis", "expected_effect", "risk")
            }
            user["human_rejection_reason"] = rejection_reason
            user["revision_requirement"] = "保留可取之处，但必须直接解决人工拒绝理由；必要时切换 target_skill，仍只改一个 skill。"
        serialized = json.dumps(user, ensure_ascii=False, separators=(",", ":"), default=str)
        self.last_input_original_chars = len(serialized)
        self.last_input_chars = len(serialized)
        self.last_input_truncated = self.max_analysis_input_chars > 0 and len(serialized) > self.max_analysis_input_chars
        if self.last_input_truncated:
            # Keep the fallback valid JSON. Prefix-cutting a JSON document can
            # make the teacher request unparsable and is worse than explicitly
            # omitting the least important tail of the Act ledger.
            original_count = len(ledger)
            shortened = list(ledger)
            user["analysis_warning"] = "analysis payload exceeded the configured character limit; tail steps were omitted"
            while shortened and len(serialized) > self.max_analysis_input_chars:
                next_length = len(shortened) // 2
                if next_length == 0:
                    break
                shortened = shortened[:next_length]
                user["action_ledger"] = shortened
                user["analysis_warning"] = (
                    "analysis payload exceeded the configured character limit; "
                    f"{original_count - len(shortened)} tail steps omitted"
                )
                serialized = json.dumps(user, ensure_ascii=False, separators=(",", ":"), default=str)
            # A pathological single step or oversized skill should still
            # produce a valid, auditable request rather than a malformed one.
            if len(serialized) > self.max_analysis_input_chars:
                minimal = {
                    "analysis_scope": user.get("analysis_scope"),
                    "terminal": user.get("terminal"),
                    "metrics": user.get("metrics"),
                    "incumbent_policy": {"version": user["incumbent_policy"].get("version"), "available_skill_keys": sorted(SKILL_KEYS)},
                    "action_ledger": [],
                    "analysis_warning": "analysis payload exceeded the configured character limit; ledger omitted",
                }
                serialized = json.dumps(minimal, ensure_ascii=False, separators=(",", ":"), default=str)
        self.last_input_chars = len(serialized)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": serialized},
        ]

    def _next_candidate_id(self) -> str:
        self.candidate_index += 1
        return f"c{self.candidate_index:04d}"

    def _propose_candidate(
        self,
        run: dict[str, Any],
        *,
        analysis_ledger: list[dict[str, Any]] | None = None,
        analysis_act: str | None = None,
        rejected_candidate: dict[str, Any] | None = None,
        rejection_reason: str = "",
    ) -> dict[str, Any]:
        messages = self._candidate_messages(
            run,
            analysis_ledger=analysis_ledger,
            analysis_act=analysis_act,
            rejected_candidate=rejected_candidate,
            rejection_reason=rejection_reason,
        )
        try:
            response = self.remote.complete(
                messages,
                prompt_version=PROMPT_VERSION,
                thinking=False,
                json_mode=True,
                streaming=False,
                temperature=0.0,
                max_tokens=self.max_analysis_tokens,
            )
        except TeacherError as exc:
            self.last_error = str(exc)
            raise PromptEvolutionError(str(exc)) from exc
        value = _extract_object(str(response["output"]))
        target = str(value.get("target_skill") or "").strip().upper()
        if target == "GLOBAL":
            target = "ROLE"
        if target not in SKILL_KEYS:
            raise PromptEvolutionError("teacher candidate has an unsupported target_skill")
        evidence = value.get("evidence_step_ids")
        if not isinstance(evidence, list) or not evidence:
            raise PromptEvolutionError("teacher candidate must cite at least one evidence_step_id")
        evidence_ledger = analysis_ledger if analysis_ledger is not None else run.get("analysis_ledger", run["ledger"])
        valid_steps = {
            item.get("step_id")
            for item in evidence_ledger
            if isinstance(item, dict) and isinstance(item.get("step_id"), int)
        }
        evidence_ids = [item for item in evidence if isinstance(item, int) and item in valid_steps]
        if not evidence_ids:
            raise PromptEvolutionError("teacher candidate evidence_step_ids do not reference this run")
        current_role, current_scenes = self._effective_skills(run["policy"])
        instruction = _compact_skill(value.get("skill") or value.get("instruction"), self.max_skill_chars)
        role_skill = instruction if target == "ROLE" else current_role
        scene_skills = dict(current_scenes)
        if target != "ROLE":
            scene_skills[target] = instruction
        candidate = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": self._next_candidate_id(),
            "character": self.character,
            "parent_version": self.active_version,
            "target_skill": target,
            "previous_skill": current_role if target == "ROLE" else str(current_scenes.get(target) or ""),
            "skill": instruction,
            "role_skill": role_skill,
            "scene_skills": scene_skills,
            "diagnosis": compact_text(value.get("diagnosis"), 1000),
            "evidence_step_ids": sorted(set(evidence_ids))[:8],
            "expected_effect": compact_text(value.get("expected_effect"), 1000),
            "risk": compact_text(value.get("risk"), 1000),
            "analysis_input_chars": self.last_input_chars,
            "analysis_input_original_chars": self.last_input_original_chars,
            "analysis_input_limit_chars": self.max_analysis_input_chars,
            "analysis_input_truncated": self.last_input_truncated,
            "analysis_act": analysis_act,
            "teacher": {
                "model_id": response["model_id"],
                "prompt_version": response["prompt_version"],
                "latency_ms": response["latency_ms"],
                "attempt": response["attempt"],
                "usage": response["usage"],
            },
            "created_at_utc": now_utc(),
        }
        self.last_error = None
        write_json(self.session_root / "candidates" / f"{candidate['candidate_id']}.json", candidate)
        return candidate

    def _propose_next_segment_candidate(self, run: dict[str, Any]) -> dict[str, Any] | None:
        """Pop the next Act segment and ask the teacher for one candidate."""

        if not self.pending_analysis_segments:
            self.current_analysis_act = None
            self.current_analysis_ledger = None
            return None
        act, fallback_ledger = self.pending_analysis_segments.pop(0)
        # After accepting an earlier Act candidate, use the latest validation
        # run as the evidence source when that Act exists there. This keeps
        # step ids and state transitions aligned with the incumbent being
        # compared; the original segment is only a fallback for a run that
        # ended before reaching the requested Act.
        latest_segments = dict(split_ledger_by_act(run.get("ledger", [])))
        ledger = latest_segments.get(act, fallback_ledger)
        self.current_analysis_act = act
        self.current_analysis_ledger = ledger
        return self._propose_candidate(run, analysis_ledger=ledger, analysis_act=act)

    def complete_run(
        self,
        run_id: str,
        run_config: dict[str, Any],
        trace_path: Path,
        terminal_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Record a terminal run and return whether a candidate/review is ready."""

        if not self.enabled:
            return {"kind": "disabled"}
        evolution = run_config.get("prompt_evolution")
        evolution = evolution if isinstance(evolution, dict) else {}
        role = str(evolution.get("role") or "")
        if role not in {"incumbent", "candidate", "revision"}:
            return {"kind": "ignored", "reason": "run has no prompt-evolution metadata"}
        policy = self.current_policy
        run = {
            "run_id": run_id,
            "run_config": dict(run_config),
            "trace_path": str(trace_path),
            "terminal_state": dict(terminal_state),
            "metrics": _metric(terminal_state),
            "policy": policy,
            "ledger": trace_ledger(trace_path),
            "completed_at_utc": now_utc(),
        }
        run["analysis_acts"] = [act for act, _ledger in split_ledger_by_act(run["ledger"])]
        write_json(self.session_root / "runs" / f"{run_id}.json", run)
        if role == "incumbent":
            self.incumbent = run
            self.candidate_result = None
            self.revision_count = 0
            self.stage = "教师正在分析 incumbent 轨迹并生成候选 skill"
            if not str(run_config.get("seed") or "").strip():
                message = "game did not expose a display seed; cannot anchor prompt evolution reruns"
                self._write_session_state({"status": "seed_unavailable", "error": message, "incumbent": run})
                return {"kind": "proposal_error", "error": message}
            try:
                self.pending_analysis_segments = split_ledger_by_act(run["ledger"])
                self.candidate = self._propose_next_segment_candidate(run)
                if self.candidate is None:
                    raise PromptEvolutionError("incumbent trace has no accepted actions to analyze")
            except PromptEvolutionError as exc:
                self._write_session_state({"status": "proposal_error", "error": str(exc), "incumbent": run})
                self.stage = "候选 skill 生成失败"
                return {"kind": "proposal_error", "error": str(exc)}
            self.stage = f"候选 skill 已生成：{self.candidate.get('target_skill', '?')}，等待人工批准验证"
            review = self._candidate_review()
            self._write_session_state({"status": "candidate_ready_review", "incumbent": run, "candidate": self.candidate, "review": review})
            return {"kind": "candidate_review", "review": review}
        if self.incumbent is None:
            return {"kind": "proposal_error", "error": "candidate run has no incumbent in this controller session"}
        if not _same_start(self.incumbent["run_config"], run_config):
            return {"kind": "proposal_error", "error": "candidate run did not use the incumbent character, ascension, and seed"}
        self.candidate_result = run
        self.stage = "候选轨迹已完成，等待人工审核"
        comparison = self._comparison(self.incumbent, run, self.candidate or {})
        review = {"schema_version": SCHEMA_VERSION, "comparison": comparison, "candidate": self.candidate, "revision": self.revision_count}
        review["phase"] = "validation"
        self._write_session_state({"status": "awaiting_review", "incumbent": self.incumbent, "candidate_result": run, "candidate": self.candidate, "review": review})
        return {"kind": "result_review", "review": review}

    def _candidate_review(self) -> dict[str, Any]:
        """Build the pre-validation review without pretending a run happened."""

        if self.incumbent is None or self.candidate is None:
            raise PromptEvolutionError("candidate review requires incumbent and candidate")
        metrics = self.incumbent.get("metrics") if isinstance(self.incumbent.get("metrics"), dict) else {}
        change = self.candidate
        return {
            "schema_version": SCHEMA_VERSION,
            "phase": "candidate",
            "comparison": {
                "seed": self.incumbent["run_config"].get("seed"),
                "incumbent": {
                    "run_id": self.incumbent["run_id"],
                    "policy_id": self.incumbent["policy"].get("candidate_id") or self.incumbent["policy"].get("version"),
                    "metrics": metrics,
                },
                "candidate": {
                    "run_id": None,
                    "policy_id": change.get("candidate_id"),
                    "metrics": None,
                },
                "score_delta": None,
                "automatically_favorable": None,
                "change_summary": {
                    key: change.get(key, change.get({"previous_skill": "previous_instruction", "skill": "instruction"}.get(key, key), ""))
                    for key in ("target_skill", "previous_skill", "skill", "diagnosis", "evidence_step_ids", "expected_effect", "risk")
                },
                "evidence": {
                    "incumbent": _evidence_steps(self.incumbent, change.get("evidence_step_ids", [])),
                    "candidate": [],
                },
            },
            "candidate": self.candidate,
            "revision": self.revision_count,
        }

    @staticmethod
    def _comparison(incumbent: dict[str, Any], candidate: dict[str, Any], change: dict[str, Any]) -> dict[str, Any]:
        left = incumbent["metrics"]
        right = candidate["metrics"]
        return {
            "seed": incumbent["run_config"].get("seed"),
            "incumbent": {"run_id": incumbent["run_id"], "policy_id": incumbent["policy"].get("candidate_id") or incumbent["policy"].get("version"), "metrics": left},
            "candidate": {"run_id": candidate["run_id"], "policy_id": candidate["policy"].get("candidate_id") or candidate["policy"].get("version"), "metrics": right},
            "score_delta": round(float(right["score"]) - float(left["score"]), 4),
            "automatically_favorable": float(right["score"]) > float(left["score"]),
            "change_summary": {
                key: change.get(key, change.get({"previous_skill": "previous_instruction", "skill": "instruction"}.get(key, key), ""))
                for key in ("target_skill", "previous_skill", "skill", "diagnosis", "evidence_step_ids", "expected_effect", "risk")
            },
            "evidence": {
                "incumbent": _evidence_steps(incumbent, change.get("evidence_step_ids", [])),
                "candidate": _evidence_steps(candidate, change.get("evidence_step_ids", [])),
            },
        }

    def handle_review(self, response: dict[str, Any]) -> dict[str, Any]:
        """Apply a human review. Accepted versions alone update the manifest."""

        if self.candidate is None or self.incumbent is None:
            return {"kind": "error", "error": "no prompt-evolution candidate is awaiting review"}
        verdict = str(response.get("verdict") or "").strip().lower()
        reason = compact_text(response.get("reason"), 4000)
        pre_validation = self.candidate_result is None
        if verdict == "abandon":
            self.stage = "本轮演化已放弃"
            self._write_session_state({"status": "abandoned", "reason": reason, "candidate": self.candidate})
            self.candidate = None
            self.candidate_result = None
            return {"kind": "abandoned"}
        if pre_validation:
            if verdict == "accept":
                self.stage = f"候选 {self.current_policy_id} 已批准，准备验证同一种子"
                self._write_session_state({"status": "candidate_approved", "incumbent": self.incumbent, "candidate": self.candidate})
                return {"kind": "start_candidate", "candidate": self.candidate, "run_config": self.incumbent["run_config"]}
            if verdict != "reject":
                return {"kind": "error", "error": "review verdict must be accept, reject, or abandon"}
            if not reason:
                return {"kind": "error", "error": "rejecting a candidate requires a reason"}
            if self.revision_count >= self.max_revisions_per_cycle:
                self.stage = "达到本轮修订上限"
                self._write_session_state({"status": "revision_limit", "reason": reason, "candidate": self.candidate})
                return {"kind": "abandoned", "reason": "reached configured revision limit"}
            rejected = self.candidate
            self.revision_count += 1
            try:
                self.candidate = self._propose_candidate(
                    self.incumbent,
                    analysis_ledger=self.current_analysis_ledger,
                    analysis_act=self.current_analysis_act,
                    rejected_candidate=rejected,
                    rejection_reason=reason,
                )
            except PromptEvolutionError as exc:
                self.stage = "候选 skill 修订失败"
                self._write_session_state({"status": "revision_error", "error": str(exc), "candidate": rejected})
                return {"kind": "error", "error": str(exc)}
            self.stage = f"修订候选 skill 已生成：{self.candidate.get('target_skill', '?')}，等待人工批准验证"
            review = self._candidate_review()
            self._write_session_state({"status": "candidate_ready_review", "incumbent": self.incumbent, "candidate": self.candidate, "review": review, "revision": self.revision_count})
            return {"kind": "candidate_review", "review": review, "revision": self.revision_count}
        if self.candidate_result is None:
            return {"kind": "error", "error": "candidate validation result is missing"}
        if verdict == "accept":
            promoted = self._promote_candidate(reason)
            self.incumbent = self.candidate_result
            # The accepted candidate becomes the immutable incumbent policy
            # for the next comparison; retain the promoted version metadata so
            # the next teacher proposal is based on pNNNN, not on cNNNN.
            self.incumbent["policy"] = promoted
            self.active_policy = promoted
            self.current_policy = promoted
            self.candidate = None
            self.candidate_result = None
            self.revision_count = 0
            if not self.auto_continue:
                self.stage = f"已接受 {self.active_version}，等待下一轮"
                self._write_session_state({"status": "accepted_paused", "active_version": self.active_version})
                return {"kind": "accepted_paused", "version": self.active_version}
            try:
                self.candidate = self._propose_next_segment_candidate(self.incumbent)
                if self.candidate is None:
                    self.stage = f"已接受 {self.active_version}，本局各 Act 已分析完毕"
                    self._write_session_state({"status": "accepted_paused", "active_version": self.active_version})
                    return {"kind": "accepted_paused", "version": self.active_version}
            except PromptEvolutionError as exc:
                self.stage = "已接受，但下一候选生成失败"
                self._write_session_state({"status": "accepted_proposal_error", "error": str(exc), "active_version": self.active_version})
                return {"kind": "accepted_paused", "version": self.active_version, "error": str(exc)}
            self.stage = f"新候选 skill 已生成：{self.candidate.get('target_skill', '?')}，等待人工批准验证"
            review = self._candidate_review()
            self._write_session_state({"status": "candidate_ready_review", "incumbent": self.incumbent, "candidate": self.candidate, "review": review})
            return {"kind": "candidate_review", "review": review, "accepted_version": self.active_version}
        if verdict != "reject":
            return {"kind": "error", "error": "review verdict must be accept, reject, or abandon"}
        if not reason:
            return {"kind": "error", "error": "rejecting a candidate requires a reason"}
        if self.revision_count >= self.max_revisions_per_cycle:
            self.stage = "达到本轮修订上限"
            self._write_session_state({"status": "revision_limit", "reason": reason, "candidate": self.candidate})
            return {"kind": "abandoned", "reason": "reached configured revision limit"}
        rejected = self.candidate
        self.revision_count += 1
        self.candidate_result = None
        try:
            self.candidate = self._propose_candidate(
                self.incumbent,
                analysis_ledger=self.current_analysis_ledger,
                analysis_act=self.current_analysis_act,
                rejected_candidate=rejected,
                rejection_reason=reason,
            )
        except PromptEvolutionError as exc:
            self.stage = "候选 skill 修订失败"
            self._write_session_state({"status": "revision_error", "error": str(exc), "candidate": rejected})
            return {"kind": "error", "error": str(exc)}
        self.stage = f"修订候选 skill 已生成：{self.candidate.get('target_skill', '?')}，等待人工批准验证"
        review = self._candidate_review()
        self._write_session_state({"status": "candidate_ready_review", "incumbent": self.incumbent, "candidate": self.candidate, "review": review, "revision": self.revision_count})
        return {"kind": "candidate_review", "review": review, "revision": self.revision_count}

    def _promote_candidate(self, human_note: str) -> dict[str, Any]:
        assert self.candidate is not None
        if self.manifest_path is None:
            run_config = self.incumbent.get("run_config", {}) if isinstance(self.incumbent, dict) else {}
            character = character_key(run_config.get("character")) if isinstance(run_config, dict) else None
            if character is None:
                raise PromptEvolutionError("cannot promote a prompt candidate without a character")
            self.character = character
            self.policy_root = self.library_root / character
            self.version_root = self.policy_root / "versions"
            self.manifest_path = self.policy_root / "manifest.json"
            self._load_layout()
        try:
            next_index = int(self.manifest.get("next_version_index"))
        except (TypeError, ValueError) as exc:
            raise PromptEvolutionError("prompt policy manifest next_version_index must be an integer") from exc
        version = f"p{next_index:04d}"
        role_skill, scene_skills = self._effective_skills(self.candidate)
        policy = {
            "schema_version": SCHEMA_VERSION,
            "version": version,
            "character": self.character,
            "parent_version": self.active_version,
            "role_skill": role_skill,
            "scene_skills": scene_skills,
            "change": {
                key: self.candidate.get(key)
                for key in ("candidate_id", "target_skill", "previous_skill", "skill", "diagnosis", "evidence_step_ids", "expected_effect", "risk", "teacher")
            },
            "human_acceptance_note": human_note,
            "accepted_at_utc": now_utc(),
        }
        write_json(self.version_root / f"{version}.json", policy)
        self.manifest["active_version"] = version
        self.manifest["next_version_index"] = next_index + 1
        self.manifest["updated_at_utc"] = now_utc()
        write_json(self.manifest_path, self.manifest)
        return policy

    def _write_session_state(self, value: dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = {"schema_version": SCHEMA_VERSION, "session_id": self.session_id, "updated_at_utc": now_utc(), **value}
        write_json(self.session_root / "state.json", payload)
