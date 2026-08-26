# Remote Policy and Online RL

The game and CommunicationMod remain on the Windows machine. A persistent
`ssh -T` JSONL child process sends routed public observations and legal action
aliases to one GPU on the training machine. No public inference port is opened.

## Modes

`policy.runtime` in `config/agent.json` selects one mode:

| Runtime | Remote command | Behavior |
| --- | --- | --- |
| `remote_inference` | `tools/policy_server.py` | Loads one fixed SFT LoRA adapter. Existing Candidate Assist or confidence-auto behavior is unchanged. |
| `remote_online_rl` | `tools/online_rl_worker.py` | Samples every action from a frozen adapter, collects `K=4` complete local episodes, then updates the resident LoRA adapter on the same remote GPU. |

`local` remains the existing on-machine Candidate Assist mode.

## Preparation

Commit and push the code, then on the training machine pull it and create the
machine-local online config:

```bash
cd /data/wanghao/sts-agent
cp config/online_rl.example.json config/online_rl.json
```

Edit only the absolute paths in `config/online_rl.json`. It is ignored by Git.
`initial_adapter` must point at a completed SFT adapter directory, not a
`checkpoint-*` directory. The first run uses `initial_policy_version`; each
completed group creates `output_root/versions/v000001/`, then `v000002/`, etc.

The example uses `gpu.device: "auto"`. At worker startup it queries
`nvidia-smi`, filters GPUs with at least `minimum_free_memory_mib` free (12 GiB
by default), then selects the lowest-utilization eligible GPU, breaking ties by
more free memory. The worker pins itself to that GPU for its full lifetime;
it does not migrate the resident model between groups.

The remote Python environment needs the training dependencies:

```bash
.venv/bin/pip install -r tools/requirements-train.txt
```

On Windows, set `policy.enabled` to `true`, disable `teacher.enabled`, and set
the relevant mode:

```json
"policy": {
  "enabled": true,
  "runtime": "remote_inference",
  "model": "/data/wanghao/sts-agent/models/base/Qwen3-0.6B",
  "adapter": "/data/wanghao/sts-agent/outputs/sft/qwen3-0.6b/sft-v0-r001",
  "remote": {
    "ssh_host": "linke84090",
    "project_root": "/data/wanghao/sts-agent",
    "python": ".venv/bin/python",
    "online_config": "config/online_rl.json",
    "connect_timeout_seconds": 30.0,
    "update_timeout_seconds": 1800.0
  }
}
```

The controller launches the SSH worker during its pre-game handshake. It must
therefore be possible to run `ssh linke84090` without an interactive password
prompt. SSH agent/key setup belongs to the local machine; no private key path
is stored in this repository.

## Remote inference

Set `runtime` to `remote_inference`. This uses the existing `rank` or `score`
request types, so setting `confidence_auto.enabled` keeps its normal meaning.
The remote process only loads the base model and specified adapter; it never
writes model weights and it cannot update the policy.

## Online RL execution

Set `runtime` to `remote_online_rl`. This mode always samples and auto-executes
the remote action; human overrides invalidate the episode for online update.
After a valid terminal upload, the controller automatically confirms the end
screen, returns to the main menu, and starts the next run with the same
character, ascension and seed. It pauses rather than changing seed when the
remote worker rejects an episode or becomes unavailable.

Consequently, enter `s <character> <ascension> <seed>` only once for a normal
online run. The controller keeps cycling through groups until the game,
controller, or remote worker is stopped. Do not manually intervene in a
rollout: a human-overridden action invalidates that episode for the update.

Use a fixed character, ascension and display seed. The worker admits the first
episode to an active group and requires every next episode in that group to
share those three values and the same frozen policy version. It refuses a
different seed until the group is complete.

```text
adapter sft-v0-r001
  -> local game episode 1  -- sampled actions --> group g-000001
  -> local game episode 2  -- sampled actions --> group g-000001
  -> local game episode 3  -- sampled actions --> group g-000001
  -> local game episode 4  -- sampled actions --> update on remote GPU
  -> saved adapter v000001
  -> next four local episodes use v000001
```

Each action trace is flushed locally before terminal upload. The local trace
records `group_id`, `policy_version`, dynamic alias, the behavior-policy
logprob, and all legal aliases. At terminal `GAME_OVER` or `VICTORY`, the
controller reconstructs a routed observation from that trace and submits one
complete episode. The remote worker persists it under
`output_root/groups/<group-id>/`; a repeated completed upload is idempotent.

The current update is action-level episode-group-relative PPO: it computes a
scalar reward for each complete episode, z-scores those four group rewards,
and applies clipped policy-gradient loss to every sampled action. This is not
TRL's one-prompt/multiple-text-completions GRPO abstraction, because game state
changes after every action. Reward components are versioned and written to each
group's `group.json`: victory, Act, floor and terminal HP fraction.

The current reward version is `v2_terminal_progress_heart`:

```text
5000 * victory
+ 10000 * beat_heart
+ 600 * terminal_act
+ 30 * terminal_floor
+ 2 * terminal_hp_fraction
```

`beat_heart` is true only for a `VICTORY` terminal state in Act 4. The floor
term is deliberately comparable to the Act term over a normal run, while HP
is only a tie-breaker. Sampling uses temperature `T=1.2`: legal-action scores
are divided by `T` before softmax, creating more rollout diversity than
`T=1.0`. The same temperature is used when recomputing PPO ratios during the
update. If a completed K=4 group has near-zero reward standard deviation,
raise it to `1.4`; if rollouts become uniformly destructive, lower it toward
`1.0`.

The fourth upload can take longer because it performs the update. The
controller waits up to `update_timeout_seconds` (default 30 minutes) before
allowing another policy request. It holds one remote GPU for both sampling and
updating, so adapter changes are synchronous and no cross-GPU reload is needed.

## Protocol smoke test

Before using a real model, test the remote command and JSONL protocol without
allocating a GPU:

```bash
cd /data/wanghao/sts-agent
cp config/online_rl.example.json /tmp/sts-online-mock.json
.venv/bin/python tools/online_rl_worker.py --config /tmp/sts-online-mock.json --mock
```

The process prints a `ready` object. Send a `sample` JSON request on one line;
the process returns `sampled_action`. The Windows controller starts this same
process automatically over SSH in real use.

## Safety and limits

- Only `remote_online_rl` auto-executes every action. Start it with an isolated
  fixed seed and watch the first group.
- A worker restart during a group retains its queue and policy version. Restart
  before the next action, then replay/re-upload only a completed local trace.
- A worker restart during the actual update is not yet transactionally
  resumable; discard that incomplete group directory and restart a fresh group.
- The update has no KL reference policy in this first implementation. Keep the
  learning rate small (`1e-5`), one update epoch, and select later versions by
  held-out fixed-seed evaluation rather than reward on the training seed alone.
