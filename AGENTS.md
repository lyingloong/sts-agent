# AGENTS.md

Short guide for coding agents. Humans: [CONTRIBUTING.md](CONTRIBUTING.md). Details: [doc/](doc/).

## Repo

CommunicationMod-based Slay the Spire 1 agent: state projection, legal actions, policy backends,
trajectories, SFT export and baseline evaluation. The in-game recorder is a separate repo
(`sts-action-recorder`); consume its events, do not vendor it.

## Commands

```sh
python -m compileall -q sts_agent tests
python -m unittest discover -s tests -v
python -m sts_agent.commands.preview_policy_prompt   # rewrites doc/POLICY_PROMPT_PREVIEW.md
git diff --check
```

Python 3.10+, no game, GPU or API key needed for tests. Unless the task asks for a live run, do not
start the game, an SSH tunnel, a model download or a real model/teacher endpoint; tests stay offline.

## Rules

- `core` imports no game, model or UI code. Keep the documented package direction.
- Preserve raw observations and ids. Model-facing cleanup works on a display copy.
- Never truncate silently, infer an executed action from a changed screen, or count a technical
  stop as a game loss.
- Live inference, SFT export and action scoring share one prompt builder. Change all or none.
- Contract changes (protocol, schema, prompt, config) update docs and tests in the same commit.
- A live-game fix needs a live-game check; state what you actually ran, not what you assume.

## Data

Never commit `data/trajectories/`, `data/experiments/`, `data/reviews/`, `data/transfer/`, models,
checkpoints, `*_api.txt`, `config/agent.json` or `config/online_rl.json`. Keep shared defaults in
`config/agent.example.json`. Use portable paths and placeholder host names in tracked examples.

## Workflow

Update [CHANGELOG.md](CHANGELOG.md) under `[Unreleased]` for every user-visible change, then commit
and push when the change is verified and the tree is clean. Prefer small, focused commits; do not
leave finished work sitting only on a local branch.
