# Data Layout

`data/` separates reproducible game knowledge from machine-local experimental data.

| Path | Git | Purpose |
| --- | --- | --- |
| `knowledge/` | tracked | Reviewed StS1 static knowledge snapshot and explicit local corrections. |
| `trajectories/` | ignored | Per-run normalized action traces; the source for dataset export. |
| `raw/` | ignored | Session-scoped CommunicationMod messages for protocol debugging only. |
| `datasets/` | ignored | Rebuildable SFT, preference, calibration, or trajectory datasets. |
| `transfer/` | ignored | Immutable data bundles copied to a training machine. |
| `runtime/` | ignored | Overlay geometry and other local UI state. |

Do not put API keys, Hugging Face caches, base-model weights, or checkpoints under `data/`. Build a transfer bundle with `tools/prepare_training_bundle.py`; it records source hashes and exports only valid teacher SFT samples. `data/raw/` is deliberately excluded because it is not a training input.
