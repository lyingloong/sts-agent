"""Merge completed baseline batches into one cohort evaluation.

Batches that ran the same model under the same conditions can be combined after
the fact, so a 10-seed historical batch and a 20-seed continuation batch form a
single 30-seed cohort without re-running the first ten games.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sts_agent.evaluation.progress import aggregate_summaries


class MergeError(ValueError):
    """A batch cannot be merged because its files or identities disagree."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise MergeError(f"missing {path}") from exc
    except json.JSONDecodeError as exc:
        raise MergeError(f"invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise MergeError(f"{path} must contain one JSON object")
    return value


def _model_name(manifest: dict[str, Any]) -> str | None:
    for key in ("model_manifest", "preflight"):
        value = manifest.get(key)
        if isinstance(value, dict):
            nested = value.get("observed_model_manifest") if key == "preflight" else value
            if isinstance(nested, dict) and nested.get("model"):
                return str(nested["model"])
    return None


def load_batch(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    results = _read_json(directory / "results.json")
    manifest = _read_json(directory / "manifest.json")
    rows = results.get("results")
    if not isinstance(rows, list):
        raise MergeError(f"{directory / 'results.json'} has no results list")
    return {"directory": directory, "suite_id": results.get("suite_id"),
            "status": results.get("status"), "model": _model_name(manifest), "rows": rows}


def merge_batches(directories: list[Path], *, allow_different_models: bool = False) -> dict[str, Any]:
    if not directories:
        raise MergeError("at least one batch directory is required")
    batches = [load_batch(path) for path in directories]
    models = {batch["model"] for batch in batches if batch["model"]}
    if len(models) > 1 and not allow_different_models:
        raise MergeError("batches use different models: " + ", ".join(sorted(models)))

    summaries: dict[str, dict[str, Any]] = {}
    per_seed: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    duplicates: list[str] = []
    for batch in batches:
        for row in batch["rows"]:
            seed = str(row.get("seed") or "")
            if seed in seen and row.get("status") == "completed":
                duplicates.append(seed)
                continue
            if row.get("status") == "completed" and isinstance(row.get("summary"), dict):
                seen[seed] = batch["directory"]
                run_id = str(row.get("run_id") or seed)
                key = run_id if run_id not in summaries else f"{batch['suite_id']}:{run_id}"
                summaries[key] = row["summary"]
                evaluation = row["summary"].get("evaluation") or {}
                per_seed.append({"seed": seed, "suite_id": batch["suite_id"], "run_id": run_id,
                                 "progress_score": evaluation.get("progress_score"),
                                 "highest_floor": evaluation.get("highest_floor"),
                                 "highest_act": evaluation.get("highest_act"),
                                 "won": row["summary"].get("won")})
    if duplicates:
        raise MergeError("duplicate completed seeds: " + ", ".join(sorted(duplicates)))

    evaluation = aggregate_summaries(summaries)
    evaluation.update(
        batches=[{"suite_id": batch["suite_id"], "status": batch["status"],
                  "directory": str(batch["directory"]), "runs": len(batch["rows"])} for batch in batches],
        model=next(iter(models)) if len(models) == 1 else None,
    )
    return {"schema_version": 1, "evaluation": evaluation, "per_seed": per_seed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge completed baseline batches into one cohort")
    parser.add_argument("batches", nargs="+", type=Path, help="batch directories containing results.json")
    parser.add_argument("--output", type=Path, help="output JSON path (default: <first batch>/merged-evaluation.json)")
    parser.add_argument("--allow-different-models", action="store_true",
                        help="merge anyway; recorded in the output but results are not comparable")
    args = parser.parse_args(argv)
    try:
        merged = merge_batches(args.batches, allow_different_models=args.allow_different_models)
    except MergeError as exc:
        parser.exit(2, f"merge failed: {exc}\n")
    output = args.output or (args.batches[0].resolve() / "merged-evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = merged["evaluation"]
    print(json.dumps({"model": summary.get("model"), "eligible_runs": summary.get("eligible_runs"),
                      "progress_score_mean": summary.get("progress_score_mean"),
                      "progress_score_std": summary.get("progress_score_std"),
                      "act_clear_rates": summary.get("act_clear_rates"),
                      "win_rate": summary.get("win_rate"), "output": str(output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
