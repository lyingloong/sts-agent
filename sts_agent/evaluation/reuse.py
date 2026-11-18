"""Reuse completed baseline runs when a cohort is extended.

A cohort is defined by its seeds plus the model and prompt configuration that
produced them. When the same configuration already has terminal, eligible runs
for some seeds, a later batch should keep those rows and only start the missing
seeds instead of replaying the whole suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sts_agent.evaluation.progress import METRIC_VERSION

# Request-shaping settings that change model behaviour. Operational settings
# such as rpm_limit, timeouts and retries are deliberately excluded: they
# affect throughput, not the recorded decisions.
SEMANTIC_SETTINGS = ("provider", "temperature", "max_output_tokens", "thinking",
                     "prompt_format", "prompt_template", "extra_body")


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _semantic_settings(settings: dict) -> dict:
    result = {key: settings.get(key) for key in SEMANTIC_SETTINGS}
    if settings.get("provider") == "openai_compatible":
        # Identical model aliases can resolve to different weights at different endpoints.
        result["base_url"] = str(settings.get("base_url") or "").rstrip("/")
    return result


def model_fingerprint(manifest: dict) -> dict:
    """The part of a controller manifest that makes two runs comparable."""

    settings = manifest.get("settings")
    settings = settings if isinstance(settings, dict) else {}
    return {"model": manifest.get("model"),
            "requested_revision": manifest.get("requested_revision"),
            "adapter": manifest.get("adapter"),
            "prompt_builder": manifest.get("prompt_builder"),
            "prompt_code_sha256": manifest.get("prompt_code_sha256"),
            "evaluation_metric": manifest.get("evaluation_metric"),
            "settings": _semantic_settings(settings)}


def expected_fingerprint(config: dict, prompt_code_sha256: str) -> dict:
    """Build the fingerprint the current configuration would produce."""

    policy = config.get("policy") or {}
    model = policy.get("model") or {}
    api = policy.get("api") or {}
    return {"model": model.get("base"),
            "requested_revision": model.get("revision"),
            "adapter": model.get("initial_adapter") or None,
            "prompt_builder": "build_scoring_prompt",
            "prompt_code_sha256": prompt_code_sha256,
            "evaluation_metric": METRIC_VERSION,
            "settings": _semantic_settings(api)}


def _eligible(row: dict, suite: dict) -> bool:
    summary = row.get("summary")
    if row.get("status") != "completed" or not isinstance(summary, dict):
        return False
    if summary.get("terminal") is not True:
        return False
    config = summary.get("run_config")
    if not isinstance(config, dict):
        return False
    if config.get("character") != suite["character"] or config.get("ascension") != suite["ascension"]:
        return False
    if config.get("partial_run") is True or summary.get("human_actions", 0) != 0:
        return False
    evaluation = summary.get("evaluation")
    return isinstance(evaluation, dict) and evaluation.get("autonomous_eligible") is True


def find_reusable_runs(output_dir: Path, suite: dict, expected: dict) -> tuple[dict[str, dict], list[dict]]:
    """Return completed runs keyed by seed, plus notes about ignored batches.

    The newest matching batch for a seed wins. Batches whose model/prompt
    fingerprint differs are ignored rather than merged silently.
    """

    seeds = set(suite["seeds"])
    found: dict[str, dict] = {}
    notes: list[dict] = []
    if not output_dir.is_dir():
        return found, notes
    for results_path in sorted(output_dir.glob("*/results.json"), key=lambda path: path.stat().st_mtime, reverse=True):
        batch_dir = results_path.parent
        results = _read_json(results_path)
        manifest = _read_json(batch_dir / "manifest.json")
        if results is None or manifest is None:
            notes.append({"batch": str(batch_dir), "skipped": "unreadable"})
            continue
        model_manifest = manifest.get("model_manifest")
        if not isinstance(model_manifest, dict):
            notes.append({"batch": str(batch_dir), "skipped": "no model manifest"})
            continue
        if model_fingerprint(model_manifest) != expected:
            notes.append({"batch": str(batch_dir), "skipped": "different model/prompt configuration"})
            continue
        for row in results.get("results") or []:
            seed = row.get("seed")
            if seed not in seeds or seed in found or not _eligible(row, suite):
                continue
            found[seed] = {"seed": seed, "run_id": row.get("run_id"), "summary": row["summary"],
                           "api_performance": row.get("api_performance"), "artifacts": row.get("artifacts"),
                           "finished_at": row.get("finished_at"), "batch": str(batch_dir),
                           "suite_id": results.get("suite_id")}
    return found, notes
