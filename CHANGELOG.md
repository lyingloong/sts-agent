# Changelog

All notable changes to `sts-agent` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses semantic versioning.

No git tag has been cut yet. `0.1.0` matches `pyproject.toml` and describes the current `master`
milestone; entries below it summarise earlier work by date range so the history stays readable.
When the first tag is created, move the `[Unreleased]` section under that version.

## [Unreleased]

### Added

- `AGENTS.md`, a short agent guide covering commands, invariants and the changelog/commit/push
  workflow, and this changelog.
- `doc/PAPER_PLAN.md`, the research plan.
- Hosted OpenAI-compatible baseline (`baseline_api`): `ApiPolicyClient` gains a `provider` switch,
  bearer auth, `extra_body` passthrough, RPM throttling, `Retry-After`-aware retries and tolerant
  alias parsing; the USTC profile ships as `config/modes/baseline_api.json`.
- Single-seed smoke suite (`config/evaluation/a0_ironclad_smoke.json`) for validating the hosted
  path before a full batch.
- Fixed 30-seed suite (`config/evaluation/a0_ironclad_30.json`); its first ten seeds are exactly
  the fixed-10 suite, and the remaining twenty continue the same deterministic generator.

### Changed

- Config validation is provider-aware: hosted profiles no longer need local ports or serving
  settings, and `policy.api.max_retries` accepts up to 10 attempts.
- Hosted runs pace at 18 RPM to stay inside USTC's fixed 60-second window of 20 requests, and the
  fixed suites allow two hours per run so a slow hosted game is not aborted.
- The paper plan targets a full-game benchmark plus an inference-time study instead of human
  trajectory collection and SFT, which are no longer on the critical path.

### Removed

- Stale machine-local `CODEX_HANDOFF.md` (untracked); repository guidance now lives in
  `AGENTS.md` and `CONTRIBUTING.md`.

## [0.1.0] - 2026-10-20

First coherent milestone: a human/AI data pipeline, a remote zero-shot baseline, a fixed-seed
evaluation harness, and an open-source-ready repository.

### Added

- Batch baseline evaluation: `sts_agent/commands/evaluate_baseline.py`, a controller-side batch
  client, fixed 10-seed A0 Ironclad suite (`config/evaluation/a0_ironclad_10.json`), per-run
  progress scoring and an aggregate `evaluation.json` (mean/SD progress, act clear rates, win
  rate, latency and token usage).
- Baseline I/O export (`commands/export_baseline_io.py`) for reviewing the exact request/response
  pairs a run produced.
- Policy presentation layer and prompt audit: `policy/presentation.py`, `policy/formats.py`,
  `core/action_aliases.py`, `policy/reward_text.py`, `game/sts1/campfire.py`, plus
  `doc/POLICY_PROMPT_AUDIT.md` and `doc/POLICY_PROMPT_PREVIEW_MARKDOWN.md`.
- Controller recovery and command channel (`runtime/recovery.py`, `runtime/command_channel.py`)
  so state changes during a submission are retried without stalling the run.
- Repository hygiene: GitHub Actions offline test matrix (3.10/3.12, Ubuntu and Windows),
  `CONTRIBUTING.md`, MIT `LICENSE`, and a sanitised `config/agent.example.json`.
- Codex MCP mode for collecting AI-played A0 Ironclad trajectories separately from human data.
- Human SFT export from runtime observations (`learning/export_human_sft.py`) and a transport
  bundle builder with checksums and seed-family splits.
- Remote BF16 baseline: pinned Qwen3.8-27B snapshot, Transformers serving entry point, SSH tunnel
  client, and the `baseline_inference` mode.
- Progress evaluation module (`evaluation/progress.py`) implementing the Orak-style score
  (cleared floors plus 50/3 per act boss, 100 for a three-act clear).
- Room-local policy history with a unified 30-step window across all room types.

### Changed

- The model-facing prompt was reworked: duplicate "current choice" blocks, reward hashes and
  numeric fields already present in card text were removed; deck composition, potion slots and
  enemy status descriptions were added.
- Controller lifetime now owns the SSH tunnel and child processes, and shuts them down on exit.
- The checked-in prompt preview is generated from the live renderer instead of being maintained by
  hand.

### Fixed

- Python 3.10 compatibility in paths the package claims to support, and tests that implicitly
  depended on a machine-local `active_mode`.
- Recorder snapshots are synchronised with the published CommunicationMod state instead of relying
  on arrival order.

### Removed

- The static `data/knowledge/` snapshot of cards, relics, powers and events. Runtime game
  descriptions are now the only entity source.

## Earlier history

### 2026-10-05 to 2026-10-09

- Human SFT construction from runtime game observations; removal of the offline knowledge base.
- Unified policy scenes with retained card-selection context.
- Remote BF16 baseline integration and weight pinning; `baseline_inference` configuration.
- Recorder snapshot synchronisation and baseline progress scoring.
- 30-step room history, overlay/terminal alignment, controller shutdown fixes.

### 2026-09-27 to 2026-09-30

- Configuration reworked into a single `active_mode` plus `config/modes/*.json`.
- Codex MCP entry point for AI-played data collection; A0 Ironclad play protocol document.
- Human collection bridge fed by ActionRecorder events, with action/state transaction matching.
- Overlay and terminal clients consolidated on one controller socket.

### 2026-08-21 to 2026-09-07

- Initial framework: CommunicationMod connection, observation/action contracts, per-run traces,
  QLoRA SFT trainer, teacher client and prompt evolution scaffolding.
