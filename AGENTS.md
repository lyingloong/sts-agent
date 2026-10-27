# AGENTS.md

Instructions for coding agents (Codex, Claude Code, Cursor, DSH and similar) working in this
repository. Humans should read [CONTRIBUTING.md](CONTRIBUTING.md); the rules overlap by design,
but this file adds the map and the traps.

## What this repository is

`sts-agent` is a CommunicationMod-based agent framework for Slay the Spire 1. It owns the game
connection, canonical observations and legal actions, policy backends (local Qwen, remote teacher,
remote baseline service, Codex MCP), trajectory storage, and the export/training pipeline.

The companion in-game recording Mod is a separate repository,
[sts-action-recorder](https://github.com/lyingloong/sts-action-recorder). This repository consumes
its events over TCP but does not vendor it. Bugs in what the game recorded belong there; bugs in
how a recording is interpreted belong here.

## Environment

- Python 3.10+ is the supported floor; CI covers 3.10 and 3.12 on Ubuntu and Windows.
- The offline suite needs no game, GPU, model weights, SSH host or API key.
- `python -m pip install -e .` installs the core package. Optional stacks live in
  `requirements/` and in `pyproject.toml` extras (`codex`, `serving`, `serving-transformers`).
- Tk is required to import the UI modules; CI installs `python3-tk` on Linux.
- Live game control is Windows-only in practice (Steam + ModTheSpire + BaseMod +
  CommunicationMod). Remote policy and training hosts are Linux.

## Commands

| Task | Command |
| --- | --- |
| Compile check | `python -m compileall -q sts_agent tests` |
| Full offline suite | `python -m unittest discover -s tests -v` |
| Whitespace gate | `git diff --check` |
| Regenerate the checked-in prompt preview | `python -m sts_agent.commands.preview_policy_prompt` |
| Batch baseline run | `python -m sts_agent.commands.evaluate_baseline --dry-run`, then without `--dry-run` |

Do not start the game, open an SSH tunnel, download weights, or call a real model/teacher endpoint
unless the task explicitly asks for a live run. Tests must not either: they use synthetic state,
loopback sockets and temporary directories.

## Layout

| Path | Contents |
| --- | --- |
| `sts_agent/core/` | Contracts, mode composition, action aliases. No game, model or UI imports. |
| `sts_agent/runtime/` | CommunicationMod protocol, controller, lifecycle, batch client, recovery. |
| `sts_agent/game/sts1/` | State projection, legal actions, per-screen adapters. |
| `sts_agent/policy/` | Prompt building, history, scenes, Qwen/remote/teacher backends. |
| `sts_agent/learning/` | SFT export, training, Prompt Evolution, post-hoc feedback, online RL. |
| `sts_agent/recording/` | ActionRecorder bridge and trajectory assembly. |
| `sts_agent/ui/` | Overlay and terminal clients. |
| `sts_agent/commands/` | One-shot download, preview, probe and evaluation entry points. |
| `config/` | `agent.json`, mode parameters under `modes/`, fixed evaluation suites. |
| `doc/` | Architecture, protocol, prompt, dataset and experiment documentation. |
| `tests/` | Offline regression suite. |

## Invariants

These are the rules a plausible-looking change is most likely to break.

1. **Package boundaries.** `core` must not import Transformers, Tk, HTTP clients, or the game
   protocol implementation. Dependencies point `core <- game/policy`, then `learning`, then
   `runtime`; `ui` consumes runtime views only.
2. **Observations are evidence.** Keep the original observation, raw message and identifiers in
   audit records. Model-facing cleanup must operate on a display copy, never mutate the record.
3. **No silent truncation.** Exported samples or prompts that exceed a limit must fail loudly.
   Do not clip, skip by length, or publish a half-written dataset directory.
4. **No inferred labels.** A training label comes from an accepted action matched to the state the
   game published before it. Do not infer an executed action from a changed screen, do not use the
   post-state as ground truth for effects, and do not treat a technical stop as a game loss.
5. **Train/serve parity.** Live inference, SFT export and action scoring share the same prompt
   builder. Changing one without the others is a bug, even if tests pass locally.
6. **Mode composition.** `config/agent.json` selects one `active_mode`; mode parameters live in
   `config/modes/`. Do not reintroduce ad-hoc flags that fork behaviour outside the mode system.
7. **Docs and tests move with contracts.** Protocol, configuration, output-schema and prompt
   changes update their documentation and tests in the same commit. Regenerate
   `doc/POLICY_PROMPT_PREVIEW.md` when the renderer output changes; a test compares it.
8. **Offline tests stay offline.** No real SSH, model service, teacher API or game process.

## Data and secrets

Ignored and never committed: `data/trajectories/`, `data/experiments/`, `data/reviews/`,
`data/transfer/`, `models/`, `checkpoints/`, `outputs/`, `*_api.txt`, `config/online_rl.json`,
`.venv*/`, `CODEX_HANDOFF.md`.

`config/agent.json` is tracked and doubles as a deployment example. Keep it free of secrets and
private account details; put real host names and keys in untracked files.

Recorded trajectories are player data. Do not paste a full run into an issue or commit; use a
short redacted excerpt or a synthetic fixture, and refer to decisions by request/step ID.

## Live-game validation

Unit tests and offline replay do not prove a live fix. When a change touches recording, action
execution or screen parsing, state which game/Mod versions, mode, exact steps and observed result
you validated. If you could not run the game, say so explicitly and list what remains unverified
instead of implying it was tested.

## Changelog

Add a line to `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md) for every user-visible change
(behaviour, configuration, schema, commands). Group under Added / Changed / Fixed / Removed /
Docs. Pure refactors and test-only changes can be omitted.

## Where to read more

- [doc/ARCHITECTURE.md](doc/ARCHITECTURE.md) — runtime chain and module boundaries
- [doc/MODES.md](doc/MODES.md) — what each `active_mode` composes
- [doc/PROTOCOL.md](doc/PROTOCOL.md) — CommunicationMod and ActionRecorder protocol
- [doc/MODEL_OBSERVATION_AND_ACTIONS.md](doc/MODEL_OBSERVATION_AND_ACTIONS.md) — what the model sees
- [doc/HUMAN_SFT_DATASET.md](doc/HUMAN_SFT_DATASET.md) — recorder → dataset → SFT export
- [doc/BASELINE_EVALUATION.md](doc/BASELINE_EVALUATION.md) — progress score and batch protocol
- [doc/PAPER_PLAN.md](doc/PAPER_PLAN.md) — research plan and remaining experiments
