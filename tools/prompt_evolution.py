"""Human-gated, single-seed prompt evolution for the remote StS teacher.

The module deliberately evolves only small scene-skill supplements.  The
stable protocol prompt and game-action compiler remain controller-owned; a
candidate cannot execute commands or become active until a human accepts it.
"""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from posthoc_feedback import _posthoc_state_snapshot, compact_text, terminal_summary
from teacher_client import RemoteTeacherClient, TeacherError, compact_json
from teacher_prompts import PROMPT_VERSION as BASE_PROMPT_VERSION
from teacher_prompts import CHARACTER_KEYS, SCENARIO_PROMPTS, build_system_prompt, character_key, scenario_key


SCHEMA_VERSION = "sts-prompt-evolution-0.1"
PROMPT_VERSION = "sts-prompt-evolution-v1"
SKILL_KEYS = frozenset({"GLOBAL", "ROLE", *SCENARIO_PROMPTS.keys()})
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
        result.append(
            {
                "step_id": record.get("step_id"),
                "screen_type": pre_state.get("screen_type"),
                "source": source,
                "policy_action": source in ACTION_SOURCES,
                "action": compact_text(action.get("label") or action.get("command") or action.get("id"), 300),
                "legal_actions": [
                    compact_text(item.get("label") or item.get("command") or item.get("id"), 240)
                    for item in legal_actions
                    if isinstance(item, dict)
                ],
                "state": _posthoc_state_snapshot(pre_state, map_ids),
            }
        )
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
        # The run's selected character is not known until START, so its
        # isolated prompt library is loaded by begin_run instead of __init__.

    def _load_layout(self) -> None:
        if self.manifest_path is None:
            raise PromptEvolutionError("prompt policy character library is not selected")
        self.manifest = _read_object(self.manifest_path, "prompt policy manifest")
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
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
        if policy.get("schema_version") != SCHEMA_VERSION or policy.get("version") != version:
            raise PromptEvolutionError(f"prompt policy version is invalid: {version}")
        if policy.get("character") != self.character:
            raise PromptEvolutionError(f"prompt policy {version} character does not match the active run")
        skills = policy.get("skills")
        if not isinstance(skills, dict):
            raise PromptEvolutionError(f"prompt policy {version} skills must be an object")
        for key, value in skills.items():
            if key not in SKILL_KEYS or not isinstance(value, str):
                raise PromptEvolutionError(f"prompt policy {version} contains invalid skill {key!r}")
        return policy

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
        else:
            self.current_policy = self.active_policy
            role = "incumbent"
            self.incumbent = None
            self.candidate_result = None
            self.revision_count = 0
        metadata = {
            "role": role,
            "character_library": character,
            "policy_id": self.current_policy_id,
            "base_version": self.active_version,
            "session_id": self.session_id,
        }
        self._write_session_state({"status": "running", "run": metadata, "run_config": run_config})
        return metadata

    def system_prompt(self, observation: dict[str, Any]) -> tuple[str, str]:
        """Compose the stable teacher prompt with accepted/candidate scene skills."""

        base = build_system_prompt(observation, character=self.character)
        if not self.enabled:
            return base, BASE_PROMPT_VERSION
        skills = self.current_policy.get("skills")
        skills = skills if isinstance(skills, dict) else {}
        selected = [
            str(skills[key]).strip()
            for key in ("GLOBAL", "ROLE", scenario_key(observation))
            if str(skills.get(key) or "").strip()
        ]
        if not selected:
            return base, f"{PROMPT_VERSION}:{self.current_policy_id}"
        appendix = "\n".join(f"- {item}" for item in selected)
        return (
            base
            + "\n\n当前已审核策略补充（仅在当前公开状态支持时适用；不得取代合法动作表或虚构信息）：\n"
            + appendix,
            f"{PROMPT_VERSION}:{self.current_policy_id}",
        )

    def _candidate_messages(
        self,
        run: dict[str, Any],
        *,
        rejected_candidate: dict[str, Any] | None = None,
        rejection_reason: str = "",
    ) -> list[dict[str, str]]:
        keys = ", ".join(sorted(SKILL_KEYS))
        system = (
            "你是《杀戮尖塔》单种子 Prompt 演化器。根据完整公开动作账本与终局结果，"
            "提出一个可审核的、很小的策略补充修改。你不是行动模型，也不能写游戏命令、代码、"
            "硬合法性限制或完整重写提示词。只修改一个 scene skill；它必须是可复用的条件性策略，"
            "不能提及当前种子、具体楼层号、具体卡牌/敌人名称，除非其是通用机制示例。"
            "不要把一次终局结果平摊归因给所有动作；只引用账本中有清楚证据的 1-8 个步骤。"
            "输出严格 JSON，不要 Markdown 或推理："
            '{"target_skill":"COMBAT","instruction":"...","diagnosis":"...",'
            '"evidence_step_ids":[4,8],"expected_effect":"...","risk":"..."}。'
            f"当前角色为 {self.character}。target_skill 必须是以下之一：{keys}。"
            "ROLE 仅描述该角色的通用机制与构筑原则；instruction 将替换该 skill 先前的补充文本。"
        )
        user = {
            "base_prompt_version": BASE_PROMPT_VERSION,
            "incumbent_policy": {
                "version": run["policy"].get("version"),
                "skills": run["policy"].get("skills", {}),
            },
            "run_config": {key: run["run_config"].get(key) for key in ("character", "ascension", "seed")},
            "terminal": terminal_summary(run["terminal_state"]),
            "metrics": run["metrics"],
            "action_ledger": run["ledger"],
        }
        if rejected_candidate is not None:
            user["rejected_candidate"] = {
                key: rejected_candidate.get(key)
                for key in ("candidate_id", "target_skill", "instruction", "diagnosis", "expected_effect", "risk")
            }
            user["human_rejection_reason"] = rejection_reason
            user["revision_requirement"] = "保留可取之处，但必须直接解决人工拒绝理由，仍只改一个 skill。"
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": compact_json(user, limit=self.max_analysis_input_chars)},
        ]

    def _next_candidate_id(self) -> str:
        self.candidate_index += 1
        return f"c{self.candidate_index:04d}"

    def _propose_candidate(
        self,
        run: dict[str, Any],
        *,
        rejected_candidate: dict[str, Any] | None = None,
        rejection_reason: str = "",
    ) -> dict[str, Any]:
        messages = self._candidate_messages(
            run,
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
        if target not in SKILL_KEYS:
            raise PromptEvolutionError("teacher candidate has an unsupported target_skill")
        evidence = value.get("evidence_step_ids")
        if not isinstance(evidence, list) or not evidence:
            raise PromptEvolutionError("teacher candidate must cite at least one evidence_step_id")
        valid_steps = {item.get("step_id") for item in run["ledger"] if isinstance(item.get("step_id"), int)}
        evidence_ids = [item for item in evidence if isinstance(item, int) and item in valid_steps]
        if not evidence_ids:
            raise PromptEvolutionError("teacher candidate evidence_step_ids do not reference this run")
        current_skills = run["policy"].get("skills")
        current_skills = current_skills if isinstance(current_skills, dict) else {}
        skills = {key: str(item) for key, item in current_skills.items() if key in SKILL_KEYS and isinstance(item, str)}
        instruction = _compact_skill(value.get("instruction"), self.max_skill_chars)
        skills[target] = instruction
        candidate = {
            "schema_version": SCHEMA_VERSION,
            "candidate_id": self._next_candidate_id(),
            "character": self.character,
            "parent_version": self.active_version,
            "target_skill": target,
            "previous_instruction": str(current_skills.get(target) or ""),
            "instruction": instruction,
            "skills": skills,
            "diagnosis": compact_text(value.get("diagnosis"), 1000),
            "evidence_step_ids": sorted(set(evidence_ids))[:8],
            "expected_effect": compact_text(value.get("expected_effect"), 1000),
            "risk": compact_text(value.get("risk"), 1000),
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
        write_json(self.session_root / "runs" / f"{run_id}.json", run)
        if role == "incumbent":
            self.incumbent = run
            self.candidate_result = None
            self.revision_count = 0
            if not str(run_config.get("seed") or "").strip():
                message = "game did not expose a display seed; cannot anchor prompt evolution reruns"
                self._write_session_state({"status": "seed_unavailable", "error": message, "incumbent": run})
                return {"kind": "proposal_error", "error": message}
            try:
                self.candidate = self._propose_candidate(run)
            except PromptEvolutionError as exc:
                self._write_session_state({"status": "proposal_error", "error": str(exc), "incumbent": run})
                return {"kind": "proposal_error", "error": str(exc)}
            self._write_session_state({"status": "candidate_ready", "incumbent": run, "candidate": self.candidate})
            return {"kind": "start_candidate", "candidate": self.candidate, "run_config": run_config}
        if self.incumbent is None:
            return {"kind": "proposal_error", "error": "candidate run has no incumbent in this controller session"}
        if not _same_start(self.incumbent["run_config"], run_config):
            return {"kind": "proposal_error", "error": "candidate run did not use the incumbent character, ascension, and seed"}
        self.candidate_result = run
        comparison = self._comparison(self.incumbent, run, self.candidate or {})
        review = {"schema_version": SCHEMA_VERSION, "comparison": comparison, "candidate": self.candidate, "revision": self.revision_count}
        self._write_session_state({"status": "awaiting_review", "incumbent": self.incumbent, "candidate_result": run, "candidate": self.candidate, "review": review})
        return {"kind": "review", "review": review}

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
                key: change.get(key, "")
                for key in ("target_skill", "previous_instruction", "instruction", "diagnosis", "evidence_step_ids", "expected_effect", "risk")
            },
            "evidence": {
                "incumbent": _evidence_steps(incumbent, change.get("evidence_step_ids", [])),
                "candidate": _evidence_steps(candidate, change.get("evidence_step_ids", [])),
            },
        }

    def handle_review(self, response: dict[str, Any]) -> dict[str, Any]:
        """Apply a human review. Accepted versions alone update the manifest."""

        if self.candidate is None or self.candidate_result is None or self.incumbent is None:
            return {"kind": "error", "error": "no prompt-evolution candidate is awaiting review"}
        verdict = str(response.get("verdict") or "").strip().lower()
        reason = compact_text(response.get("reason"), 4000)
        if verdict == "abandon":
            self._write_session_state({"status": "abandoned", "reason": reason, "candidate": self.candidate})
            self.candidate = None
            self.candidate_result = None
            return {"kind": "abandoned"}
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
                self._write_session_state({"status": "accepted_paused", "active_version": self.active_version})
                return {"kind": "accepted_paused", "version": self.active_version}
            try:
                self.candidate = self._propose_candidate(self.incumbent)
            except PromptEvolutionError as exc:
                self._write_session_state({"status": "accepted_proposal_error", "error": str(exc), "active_version": self.active_version})
                return {"kind": "accepted_paused", "version": self.active_version, "error": str(exc)}
            self._write_session_state({"status": "candidate_ready", "incumbent": self.incumbent, "candidate": self.candidate})
            return {"kind": "start_candidate", "candidate": self.candidate, "run_config": self.incumbent["run_config"], "accepted_version": self.active_version}
        if verdict != "reject":
            return {"kind": "error", "error": "review verdict must be accept, reject, or abandon"}
        if not reason:
            return {"kind": "error", "error": "rejecting a candidate requires a reason"}
        if self.revision_count >= self.max_revisions_per_cycle:
            self._write_session_state({"status": "revision_limit", "reason": reason, "candidate": self.candidate})
            return {"kind": "abandoned", "reason": "reached configured revision limit"}
        rejected = self.candidate
        self.revision_count += 1
        self.candidate_result = None
        try:
            self.candidate = self._propose_candidate(
                self.incumbent,
                rejected_candidate=rejected,
                rejection_reason=reason,
            )
        except PromptEvolutionError as exc:
            self._write_session_state({"status": "revision_error", "error": str(exc), "candidate": rejected})
            return {"kind": "error", "error": str(exc)}
        self._write_session_state({"status": "candidate_ready", "incumbent": self.incumbent, "candidate": self.candidate, "revision": self.revision_count})
        return {"kind": "start_candidate", "candidate": self.candidate, "run_config": self.incumbent["run_config"], "revision": self.revision_count}

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
        policy = {
            "schema_version": SCHEMA_VERSION,
            "version": version,
            "character": self.character,
            "parent_version": self.active_version,
            "skills": self.candidate["skills"],
            "change": {
                key: self.candidate.get(key)
                for key in ("candidate_id", "target_skill", "previous_instruction", "instruction", "diagnosis", "evidence_step_ids", "expected_effect", "risk", "teacher")
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
