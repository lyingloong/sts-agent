"""Build response-only SFT samples from an ActionRecorder dataset bundle.

Run selection belongs to ActionRecorder. This module owns policy action mapping,
runtime state projection, history, training-step filtering and the exact prompt.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

from sts_agent.learning.export_teacher_sft import alias_actions, action_id
from sts_agent.learning.sft import (git_metadata, read_jsonl, render_chat_prefix,
                                    sha256_file, validate_sample)
from sts_agent.paths import PROJECT_ROOT
from sts_agent.policy.local_qwen import PROMPT_TEMPLATES, build_scoring_prompt
from sts_agent.recording.communication_bridge import CommunicationBridgeRecorder


DEFAULT_MAX_LENGTH = 8192


class ExportLengthError(ValueError):
    """A full sample cannot fit; never truncate or silently exclude it."""

    def __init__(self, sample_id: str, lengths: dict[str, int], limit: int):
        self.sample_id, self.lengths, self.limit = sample_id, lengths, limit
        detail = ", ".join(f"{name}={length}" for name, length in lengths.items())
        super().__init__(
            f"sample {sample_id} exceeds --max-length {limit}: {detail} tokens "
            "(chat template + input + answer + EOS). Export aborted; no truncation "
            "or length-based sample exclusion. Increase --max-length and retry."
        )


def checked_path(root: Path, item: dict) -> Path:
    path = (root / item["path"]).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"invalid dataset member: {item['path']}")
    if sha256_file(path) != item.get("sha256"):
        raise ValueError(f"dataset member hash mismatch: {path}")
    return path


def export_record(row: dict) -> tuple[dict | None, str]:
    """Export pre-action supervision; post-state/animation stability is irrelevant.

    ``usable_for_sft`` still requires exact pre-state/action matching and,
    for queued combat actions, proof that the action actually executed.
    """
    if not row.get("usable_for_sft"):
        return None, row.get("training_exclusion_reason") or "unusable_step"
    observation, action, legal = (row.get(k) for k in
        ("training_policy_observation", "training_action", "training_legal_actions"))
    if not isinstance(observation, dict) or not isinstance(action, dict) or not isinstance(legal, list):
        return None, "missing_training_fields"
    aliases, mapping = alias_actions(legal)
    target_id = action_id(action)
    if target_id not in mapping:
        return None, "target_not_legal"
    sample = {
        "schema_version": "sts-human-sft-1", "source": "human_actionrecorder",
        "sample_id": f"{row['run_id']}:{row['step_id']}", "run_id": row["run_id"],
        "run_config": row.get("run_config"), "screen_type": observation.get("screen_type"),
        "observation": observation, "legal_actions": aliases,
        "target": {"alias": mapping[target_id], "action_id": target_id},
        "provenance": {"transaction_id": row["step_id"], "segment_id": row.get("segment_id"),
                       "observation_source": row.get("training_observation_source"),
                       "submission_state_id": (row.get("raw_message_before") or {}).get("recorder_state_id"),
                       "execution_state_id": (row.get("execution_raw_message_before") or {}).get("recorder_state_id")},
        "prompt": build_scoring_prompt(observation, aliases, "canonical"),
        "prompt_variants": {
            name: build_scoring_prompt(observation, aliases, name)
            for name in PROMPT_TEMPLATES
        },
    }
    validate_sample(sample)
    return sample, "kept"


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_dataset(source: Path, output: Path, *, tokenizer=None, max_length: int | None = None,
                  tokenizer_name: str | None = None) -> dict:
    """Publish a complete bundle only on success; failures leave no partial output."""
    if output.exists() or output.is_symlink():
        raise ValueError(f"output must be a new directory: {output}")
    if max_length is not None and (tokenizer is None or max_length < 1):
        raise ValueError("a positive --max-length requires --tokenizer")
    if tokenizer is not None and max_length is None:
        max_length = DEFAULT_MAX_LENGTH
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        result = _build_dataset(source, staging, tokenizer=tokenizer,
                                max_length=max_length, tokenizer_name=tokenizer_name)
        if output.exists() or output.is_symlink():
            raise ValueError(f"output appeared during export; refusing to overwrite: {output}")
        staging.rename(output)
    return result


def _build_dataset(source: Path, output: Path, *, tokenizer, max_length: int | None,
                   tokenizer_name: str | None) -> dict:
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "actionrecorder-dataset-1":
        raise ValueError("expected an ActionRecorder dataset (actionrecorder-dataset-1)")
    inputs, run_ids = [], set()
    for run in manifest["runs"]:
        if run["run_id"] in run_ids:
            raise ValueError(f"duplicate run: {run['run_id']}")
        run_ids.add(run["run_id"])
        inputs.append((run, checked_path(source, run["events"]), checked_path(source, run["trajectory"])))
    output.mkdir(parents=True)
    (output / "dataset").mkdir()
    (output / "previews").mkdir()
    dataset_path = output / "dataset" / "human-sft.jsonl"
    report = {"runs": [], "steps": Counter(), "screens": Counter(), "max_prompt_chars": 0,
              "token_lengths": [], "variant_token_lengths": {name: [] for name in PROMPT_TEMPLATES},
              "tokenizer": tokenizer_name, "max_length": max_length,
              "prompt_templates": list(PROMPT_TEMPLATES)}
    previews, identities = set(), set()
    with dataset_path.open("w", encoding="utf-8") as destination, (output / "excluded-steps.jsonl").open("w", encoding="utf-8") as rejected:
        for index, (run, events_path, steps_path) in enumerate(inputs):
            derived = output / "trajectories" / f"run-{index:04d}"
            recorder = CommunicationBridgeRecorder(derived)
            recorder.start()
            try:
                for event in read_jsonl(events_path):
                    if str(event.get("run_id")) != run["run_id"]:
                        raise ValueError(f"unexpected run ID in {events_path}")
                    recorder.ingest_marker(event)
            finally:
                recorder.close()
            counts, seen_transactions = Counter(), set()
            for path in sorted(derived.glob("human-*.jsonl")):
                for row in read_jsonl(path):
                    if not row.get("step_id"):
                        continue
                    seen_transactions.add(row["step_id"])
                    sample, reason = export_record(row)
                    if sample is not None and tokenizer is not None:
                        target_length = len(tokenizer(sample["target"]["alias"] + "\n", add_special_tokens=False, truncation=False)["input_ids"])
                        target_length += int(tokenizer.eos_token_id is not None)
                        lengths = {}
                        for template_name, prompt in sample["prompt_variants"].items():
                            prefix = render_chat_prefix(tokenizer, prompt)
                            length = len(tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"]) + target_length
                            lengths[template_name] = length
                            report["variant_token_lengths"][template_name].append(length)
                        length = lengths["canonical"]
                        sample["token_length"] = length
                        sample["variant_token_lengths"] = lengths
                        report["token_lengths"].append(length)
                        overlong = [name for name, count in lengths.items() if max_length is not None and count > max_length]
                        if overlong:
                            raise ExportLengthError(sample["sample_id"], lengths, max_length)
                    counts[reason] += 1
                    if sample is None:
                        rejected.write(json.dumps({"run_id": run["run_id"], "transaction_id": row["step_id"], "reason": reason}) + "\n")
                        continue
                    if sample["sample_id"] in identities:
                        raise ValueError(f"duplicate sample: {sample['sample_id']}")
                    identities.add(sample["sample_id"])
                    sample["run_outcome"] = run["metadata"]["outcome"]  # metadata, never part of prompt
                    destination.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    report["max_prompt_chars"] = max(report["max_prompt_chars"], len(sample["prompt"]))
                    screen = str(sample["screen_type"])
                    report["screens"][screen] += 1
                    if screen not in previews:
                        previews.add(screen)
                        filename = re.sub(r"[^A-Za-z0-9_-]", "_", screen) + ".md"
                        (output / "previews" / filename).write_text(
                            f"样本：{sample['sample_id']}\n\n## 输入\n\n{sample['prompt']}\n\n## 监督答案\n\n{sample['target']['alias']}\n",
                            encoding="utf-8")
                        for template_name, prompt in sample["prompt_variants"].items():
                            variant_filename = re.sub(r"[^A-Za-z0-9_-]", "_", f"{screen}--{template_name}") + ".md"
                            (output / "previews" / variant_filename).write_text(
                                f"样本：{sample['sample_id']}\n\n模板：{template_name}\n\n## 输入\n\n{prompt}\n\n## 监督答案\n\n{sample['target']['alias']}\n",
                                encoding="utf-8")
            # Explain raw attempts which never became accepted policy steps.
            for step in read_jsonl(steps_path):
                if step["transaction_id"] not in seen_transactions:
                    reason = "raw_" + step["status"] + "_without_policy_step"
                    counts[reason] += 1
                    rejected.write(json.dumps({"run_id": run["run_id"], "transaction_id": step["transaction_id"], "reason": reason}) + "\n")
            report["steps"].update(counts)
            report["runs"].append({"run_id": run["run_id"], "counts": dict(counts)})
    write_json(output / "quality_report.json", report)
    provenance_files = sorted((PROJECT_ROOT / "sts_agent").rglob("*.py"))
    result = {"schema_version": "sts-human-sft-bundle-1",
              "dataset": {"path": "dataset/human-sft.jsonl", "sha256": sha256_file(dataset_path), "samples": len(identities)},
              "source_manifest_sha256": sha256_file(manifest_path), "git": git_metadata(PROJECT_ROOT),
              "files": {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in provenance_files},
              "action_granularity": "fine", "prompt_templates": list(PROMPT_TEMPLATES),
              "tokenizer": tokenizer_name, "max_length": max_length,
              "length_checked": tokenizer is not None, "length_policy": "error_on_overflow_no_truncation",
              "notes": "Only response tokens are supervised. Runtime entity descriptions are supplied by the game bridge; no local entity snapshot or character truncation is used."}
    write_json(output / "manifest.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="ActionRecorder dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="new SFT bundle directory")
    parser.add_argument("--tokenizer", help="optional locally cached tokenizer name/path; never downloads")
    parser.add_argument("--max-length", type=int,
                        help=f"abort on any overlong sample; default {DEFAULT_MAX_LENGTH} with --tokenizer; never truncate")
    args = parser.parse_args()
    try:
        tokenizer = None
        if args.tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        else:
            print("warning: no --tokenizer; token length is not checked. Full prompts are preserved without truncation.", file=sys.stderr)
        result = build_dataset(args.input, args.output, tokenizer=tokenizer,
                               max_length=args.max_length, tokenizer_name=args.tokenizer)
        print(json.dumps(result["dataset"], ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
