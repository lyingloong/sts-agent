# StS Agent

**English** · [简体中文](README-zh.md)

[![Offline tests](https://github.com/lyingloong/sts-agent/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/lyingloong/sts-agent/actions/workflows/tests.yml)

Record a *Slay the Spire 1* run, get suggestions while you play, or let a model play through the same interface. You can evaluate models on fixed seeds and export gameplay decisions as training data.

CommunicationMod reports the game state. The companion ActionRecorder mod captures the actions you take. StS Agent matches the two by their state and action IDs and writes one JSONL trace per run. In model-controlled modes, StS Agent sends the state and legal actions to the model and executes its choice through CommunicationMod.

## Record your first run

The steps below use Windows PowerShell. The default `collect_human` mode records as you play. It runs without a GPU, model or API key.

### Install StS Agent

You need Git, Python 3.10 or later, and a copy of *Slay the Spire 1*. Clone the repository into a path without spaces. Install it in editable mode so the game process can use the configuration, prompts and output directories in your checkout.

```powershell
git clone https://github.com/lyingloong/sts-agent.git
cd sts-agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\sts-agent-config.exe
```

The last command shows the configuration path and active mode. On a fresh checkout, the mode should be `collect_human`.

<details>
<summary>Install the offline tools on Linux or macOS</summary>

Run these commands from the root of the cloned repository. Game integration and the overlay are tested primarily on Windows. Model serving and training are tested primarily on Linux.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/sts-agent-config
```

</details>

### Start the game

1. Install ModTheSpire, BaseMod, CommunicationMod and ActionRecorder. Enable the last three in ModTheSpire, launch the game once, then exit. [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod) · [ActionRecorder installation guide](https://github.com/lyingloong/sts-action-recorder)
2. Open `%LOCALAPPDATA%\ModTheSpire\CommunicationMod\config.properties` and set the three values below. Replace `C\:/path/to/sts-agent` with the absolute path to your checkout, using forward slashes as shown. CommunicationMod splits the command on whitespace, so the path to `pythonw.exe` must be free of spaces.

   ```properties
   command=C\:/path/to/sts-agent/.venv/Scripts/pythonw.exe -m sts_agent.runtime.controller
   runAtGameStart=true
   maxInitializationTimeout=300
   ```

3. Relaunch through ModTheSpire, start a run and play a card. CommunicationMod starts the StS Agent controller, which records the run in `collect_human` mode. You can leave the overlay closed.

### Find your recording

From the repository root, list the most recent run files by modification time.

```powershell
Get-ChildItem .\data\trajectories\human\human-*.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 3 Name,Length
```

Each `human-*.jsonl` file records the state before an action, the legal choices and the action the player took. ActionRecorder also writes its raw events to `data/actionrecorder/` under the **game’s working directory**. If your checkout contains only `communication-bridge-*.jsonl` files, check that ActionRecorder is enabled and that its destination matches the recorder listener (default `127.0.0.1:8766`). [Trace fields](doc/DATA_AND_TRACE_SCHEMA.md) · [How states and actions are matched](doc/PROTOCOL.md) (in Chinese)

## Choose a mode

To customize the configuration, copy `config/agent.example.json` to `config/agent.json`. Git ignores this local file. Set `active_mode` to the mode you want, and put changes specific to that mode under `mode_overrides`. Run `sts-agent-config --list` to see the available modes or `sts-agent-config --mode baseline_api` to check a mode offline before switching. Restart the game and any connected client after making changes. [Configuration template](config/agent.example.json) · [Configuration examples and precedence (in Chinese)](config/README.md)

The detailed guides in the table are in Chinese.

| Task | Mode or tool | Guide |
| --- | --- | --- |
| Get suggestions or let a local or remote model play | Use `candidate_assist_local` for suggestions or `local_inference` for automatic play. Both have remote variants and work with the overlay or terminal. | [Modes](doc/MODES.md) · [Commands](doc/COMMANDS.md) |
| Compare models on the same seeds | Use `baseline_api` for a hosted API or `baseline_inference` for a self-hosted model. Preview the batch with `sts-agent-evaluate-baseline --dry-run`. | [Configure a backend](doc/BASELINE_INFERENCE.md) · [Run and score an evaluation](doc/BASELINE_EVALUATION.md) |
| Let Codex play an Ascension 0 Ironclad run | `collect_codex` connects Codex to the controller through a local MCP server. | [Play with Codex](doc/CODEX_PLAY_A0_IRONCLAD.md) |
| Build a training dataset or update a policy online | Export human demonstrations from an ActionRecorder dataset, train on teacher trajectories, or run online reinforcement learning on a remote machine. | [Human demonstrations for SFT](doc/HUMAN_SFT_DATASET.md) · [Teacher data and training](doc/TRAINING_ON_4090.md) · [Online reinforcement learning](doc/REMOTE_ONLINE_RL.md) |

To use the short command names above, activate the virtual environment or run the corresponding executables from `.venv\Scripts\`. The controller accepts one client at a time, so disconnect the interactive client before starting a batch evaluation. [Command reference (in Chinese)](doc/COMMANDS.md)

## Development and support

[Documentation index (in Chinese)](doc/README.md) · [Architecture](doc/ARCHITECTURE.md) · [Example policy prompt](doc/POLICY_PROMPT_PREVIEW.md) · [Changelog](CHANGELOG.md)

Maintenance continues on game integration, recording and evaluation tools. Work on the original paper and its experiments has stopped. [Research plan and historical results (in Chinese)](doc/PAPER_PLAN.md)

Use GitHub Issues for questions and bug reports. Please report security issues privately. The contributing guide covers offline tests and the pull request process. [Open an issue](https://github.com/lyingloong/sts-agent/issues) · [Report a security issue](SECURITY.md) · [Contribute](CONTRIBUTING.md)

StS Agent is licensed under MIT. The game, third-party mods, model weights and datasets have their own licenses. [MIT License](LICENSE)
