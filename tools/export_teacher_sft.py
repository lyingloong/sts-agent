"""Export accepted remote-teacher trace steps into alias-based SFT JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if isinstance(value, dict):
                yield value


def action_id(action: dict[str, Any]) -> str:
    return str(action.get("id") or action.get("action_id") or "")


def action_label(action: dict[str, Any]) -> str:
    return str(action.get("label") or action.get("kind") or action.get("command") or "")


def alias_actions(legal_actions: list[Any]) -> tuple[list[dict[str, str]], dict[str, str]]:
    aliases: list[dict[str, str]] = []
    action_to_alias: dict[str, str] = {}
    for index, item in enumerate(legal_actions, start=1):
        if not isinstance(item, dict):
            continue
        canonical_id = action_id(item)
        if not canonical_id or canonical_id in action_to_alias:
            continue
        alias = f"A{len(aliases) + 1}"
        action_to_alias[canonical_id] = alias
        aliases.append({"id": alias, "label": action_label(item), "action_id": canonical_id})
    return aliases, action_to_alias


def export_record(record: dict[str, Any], run_outcome: str | None = None) -> dict[str, Any] | None:
    if not record.get("accepted") or not record.get("stable_after_action"):
        return None
    decision = record.get("decision")
    if not isinstance(decision, dict) or decision.get("source") != "teacher_auto":
        return None
    pre_state = record.get("pre_state")
    action = record.get("action")
    legal_actions = record.get("legal_actions")
    if not isinstance(pre_state, dict) or not isinstance(action, dict) or not isinstance(legal_actions, list):
        return None
    target_action_id = action_id(action)
    aliases, action_to_alias = alias_actions(legal_actions)
    target_alias = action_to_alias.get(target_action_id)
    if target_alias is None:
        return None
    teacher = decision.get("teacher")
    sample = {
        "schema_version": "sts-teacher-sft-0.1",
        "sample_id": f"{record.get('run_id', 'unknown')}:{record.get('step_id', 'unknown')}",
        "run_id": record.get("run_id"),
        "run_config": record.get("run_config"),
        "screen_type": record.get("screen_type") or pre_state.get("screen_type"),
        "observation": pre_state,
        "legal_actions": aliases,
        "target": {"alias": target_alias, "action_id": target_action_id},
        "source": "teacher_auto",
        "teacher": teacher if isinstance(teacher, dict) else {"model_id": decision.get("model_id")},
    }
    if run_outcome is not None:
        sample["run_outcome"] = run_outcome
    return sample


def terminal_outcome(records: Iterable[dict[str, Any]]) -> str | None:
    """Return a trusted terminal result for one per-run trace file."""
    for record in records:
        if not record.get("accepted") or not record.get("stable_after_action") or not record.get("terminal"):
            continue
        post_state = record.get("post_state")
        screen_type = post_state.get("screen_type") if isinstance(post_state, dict) else None
        if screen_type == "VICTORY":
            return "victory"
        if screen_type == "GAME_OVER":
            return "game_over"
    return None


def trace_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("action-trace-*.jsonl"))
    raise ValueError(f"input path does not exist: {input_path}")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export accepted remote-teacher StS traces as SFT JSONL")
    parser.add_argument("--input", type=Path, default=root / "data" / "trajectories", help="trace file or directory")
    parser.add_argument("--output", type=Path, default=root / "data" / "datasets" / "teacher-sft.jsonl", help="output JSONL path")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="include traces without a GAME_OVER/VICTORY terminal record; use only for debugging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = trace_paths(args.input)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as output:
        for path in paths:
            records = list(read_jsonl(path))
            outcome = terminal_outcome(records)
            if outcome is None and not args.include_incomplete:
                skipped += len(records)
                continue
            for record in records:
                sample = export_record(record, outcome or "incomplete")
                if sample is None:
                    skipped += 1
                    continue
                output.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1
    print(f"exported {written} teacher SFT samples from {len(paths)} trace files; skipped {skipped} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
