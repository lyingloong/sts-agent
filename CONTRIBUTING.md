# Contributing

## Development setup

Clone the repository, create a Python 3.10+ virtual environment and install the core package from the checkout with `python -m pip install -e .`.
Source checkouts supply the runtime profiles and prompts. On Linux, the full test suite imports Tkinter for UI tests. Use a Python distribution with Tk support or install `python3-tk` for the matching Ubuntu system Python.
Core tests run offline with synthetic data and loopback services.
Tests that exercise optional Torch tensors skip explicitly when Torch is absent. Install the appropriate optional dependencies to run those checks.
Install optional model/training dependencies only for the workflow being changed; see the README and `requirements/`.

From the repository root:

```sh
python -m compileall -q sts_agent tests
python -m unittest discover -s tests -v
git diff --check
```

When changing prompt construction, regenerate the developer preview using
`python -m sts_agent.commands.preview_policy_prompt` and update `doc/POLICY_PROMPT_PREVIEW.md` with its complete output.
Tests verify that the checked-in preview matches the live renderer.

## Submitting changes

Create a topic branch from the current `master` branch in your fork. Include a regression test
for a bug fix and update the relevant usage example for a behavior change. Open a pull request
against `master`, linking the issue if one exists. Describe the user-visible change and the
checks you ran. Mention any game, mod or model combination that still needs live verification.

## Changes and validation

- Keep game adaptation, controller transport, policy display and learning code in their existing package boundaries.
- Preserve original observations and identities in audit records; model-facing cleanup must operate on a display copy.
- Never silently truncate exported inputs, infer an executed action from a changed screen, or treat a technical stop as a game loss.
- Protocol, configuration and output changes must update their documentation and tests together.
- Automated tests use synthetic inputs, local loopback services and temporary files. Do not make them contact real SSH/model/API endpoints.

Game/mod interoperability still needs manual validation. State the game/mod versions, selected mode, exact steps and observed result;
a passing unit test or offline replay is not proof of a live-game fix.

## Reports and data hygiene

Report bugs using [GitHub Issues](https://github.com/lyingloong/sts-agent/issues) with the relevant versions, mode name, error message and a minimal reproduction.
Prefer a small redacted excerpt or synthetic fixture rather than a full trajectory. Label decisions by request/step ID when available.

Do not commit credentials, private keys, local handoff files, raw trajectories, review exports, game JARs, model weights or checkpoints.
Check staged files with `git diff --cached --stat` and `git diff --cached` before submitting.
Remove secrets and private host/account details from logs before sharing. For suspected credential leaks or security-sensitive reports, follow the [private reporting instructions](SECURITY.md).

Game assets, third-party Mods, model weights and recorded datasets have their own rights and distribution requirements.
Code contribution does not grant rights to redistribute those materials.
The repository code is available under the [MIT License](LICENSE).
