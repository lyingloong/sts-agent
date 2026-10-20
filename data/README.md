# Data layout

`data/` contains machine-local recording, evaluation and learning artifacts. Only this guide is tracked.
Entity descriptions come from current game objects; this repository no longer ships a static game knowledge database.

| Path | Purpose |
| --- | --- |
| `trajectories/` | Per-run normalized human, teacher or Codex action traces. |
| `raw/`, `traces/` | Raw messages and protocol/debugging records. |
| `experiments/` | Baseline request/response logs, action traces, summaries and batch reports. |
| `reviews/` | Locally exported complete model I/O and representative review cases. |
| `datasets/` | Rebuildable SFT and other training datasets. |
| `transfer/` | Checksummed transfer bundles for training machines. |
| `reflections/`, `attributions/`, `prompt_evolution/` | Experiment feedback and prompt evolution artifacts. |
| `runtime/` | UI state and local model deployment metadata. |

All paths above are ignored by Git. Do not commit generated logs just because they use Markdown or JSON:
they can contain machine paths, model outputs and private game/session data.
Small synthetic test inputs belong under `tests/`; publishing a dataset requires a separate review.

Human SFT export starts with an ActionRecorder dataset: see [Human SFT datasets](../doc/HUMAN_SFT_DATASET.md).
For checked teacher data transfer bundles use `python -m sts_agent.learning.training_bundle --help`.
Model weights, adapters and checkpoints belong in the ignored `models/`, `outputs/` or `checkpoints/` directories,
and API keys must remain in ignored credential files.
