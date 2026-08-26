# 8x RTX 4090 Training And Migration

本指南假定训练机是可通过 SSH 访问的 Linux 主机。它只承担模型训练、离线评测和 checkpoint 存储；StS1、CommunicationMod、controller 和人工接管继续在本地 Windows 机器运行。

## 先明确机器分工

```text
本地 Windows / RTX 3050                 远程 Linux / 8x RTX 4090
StS1 + CommunicationMod                  Qwen SFT / 评测 / 后续 GRPO 更新
controller + Overlay                     基础模型下载与 LoRA checkpoint 存储
action-trace 采集                        数据校验、训练日志和 run manifest
```

当前 controller 只会在本机启动 `policy_server.py`，尚未实现远程推理服务。因此训练后需要把 LoRA adapter 拉回本机，或后续单独实现远程 policy endpoint。8 张 4090 不能自动产生 8 路游戏 rollout；每一路 rollout 仍需要一个真实、可控制的 StS 游戏进程。

截至 2026-08-26，第一批数据为 30 条完整终局轨迹、6,417 条可导出教师 SFT 样本。它们全部是 Ironclad A0 且均以死亡结束，适合验证 SFT 管线和接口学习，不适合作为最终策略质量结论。

## 迁移什么

| 类别 | 迁移方式 | 原因 |
| --- | --- | --- |
| 源码与 `data/knowledge/` | Git | 小、版本化、可审查。 |
| `data/trajectories/` 与导出的 SFT | training bundle | 保留 dataset hash、来源 trace hash 和导出规则。 |
| `data/raw/` | 默认不迁移 | 仅协议排错，不是训练输入。 |
| Qwen 基础模型 | 在训练机重新下载 | 不复制 Hugging Face cache、`.venv` 或 `.uv-cache`。 |
| LoRA checkpoint 与评测结果 | 从训练机拉回 | 只同步 adapter、tokenizer、run manifest 和评测文件。 |
| `*_api.txt`、SSH 私钥 | 永不迁移或提交 | 凭据应由目标机器自己的密钥管理提供。 |

## 1. 在本机生成数据 release

先在项目根目录执行：

```powershell
python tools\prepare_training_bundle.py `
  --output data\transfer\sft-v0-<YYYYMMDD>-r1 `
  --include-traces
```

bundle 包含：

- `dataset/teacher-sft.jsonl`：训练输入；
- `traces/`：可选的、可重新导出的 action trace；
- `manifest.json`：样本数、screen 分布、教师模型、来源文件 hash、终局统计，以及导出时的 Git commit/dirty 标记；
- `SHA256SUMS`：训练机校验清单。

这个目录不可覆盖，避免把一个已有 release 悄悄替换为不同数据。新的采集批次或导出代码使用新的版本名，例如 `sft-v1-20260903-r1`。

准备源代码前，应将当前变更提交为一个可识别的 Git commit。训练记录只引用 commit hash，不引用“当前工作区”。`config/agent.json` 可由 Git 同步，但训练机不应放置教师 API 文件，也不需要复制 Overlay 的本地布局。

## 2. 传输与校验

网络稳定时优先使用支持续传和 checksum 的 `rsync`。Windows 可从 WSL 或 Git Bash 执行：

```bash
rsync -aP --checksum data/transfer/sft-v0-<YYYYMMDD>-r1/ \
  <user>@<host>:/data/sts-agent/datasets/sft-v0-<YYYYMMDD>-r1/
```

没有 `rsync` 时，可以打包后用 `scp`：

```powershell
tar -czf data\transfer\sft-v0-<YYYYMMDD>-r1.tar.gz -C data\transfer sft-v0-<YYYYMMDD>-r1
scp data\transfer\sft-v0-<YYYYMMDD>-r1.tar.gz <user>@<host>:/data/sts-agent/incoming/
```

训练机解包并校验：

```bash
mkdir -p /data/sts-agent/datasets
tar -xzf /data/sts-agent/incoming/sft-v0-<YYYYMMDD>-r1.tar.gz -C /data/sts-agent/datasets
cd /data/sts-agent/datasets/sft-v0-<YYYYMMDD>-r1
sha256sum -c SHA256SUMS
```

校验必须在训练前通过。`manifest.json` 中的 `dataset.sha256`、样本数、教师模型分布和终局分布应写入每一次训练的 run manifest。

## 3. 初始化训练机

克隆与数据 release 应分离：

```bash
git clone <repo-url> /workspace/sts-agent
cd /workspace/sts-agent
git checkout <source-commit>

uv venv .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r tools/requirements-train.txt
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

`cu128` 只是与当前项目兼容的例子；应按训练机驱动和 PyTorch 官方 wheel 索引选择 CUDA wheel。Python 环境、缓存和 Hugging Face cache 都在训练机新建，不能从 Windows 直接复制。

Qwen 基础模型应在训练机下载到明确目录，训练记录固定 revision。常规联网场景可由 Hugging Face Python API 完成：

```bash
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='/data/sts-agent/models/base/Qwen3-0.6B')"
```

仅在训练机完全无法联网时，才从本机导出一个真实模型目录后传输；不要复制 Hugging Face cache 的 `blobs/` 和 `snapshots/` 内部布局。

## 4. 8 卡的合理用法

当前 Qwen3-0.6B 和 6,417 条样本不需要 8 卡 DDP。把小数据强行分给 8 卡会使每张卡的有效 batch 太小，通信开销反而主导。

第一轮 SFT 推荐先用 1 张 4090：4-bit NF4 QLoRA、BF16 计算、gradient checkpointing、动态 padding，最大长度先设为 3,072。现有教师请求长度中约 95.4% 不超过该长度；超过部分应做结构化 observation 压缩，并始终保留当前状态和全部 `legal_actions`，不能从尾部截掉动作表。

其余 GPU 在当前阶段更适合并行做：

- 不同 LoRA rank、学习率、训练长度的 SFT 消融；
- 固定种子上的独立评测；
- 数据质量、重复状态和教师错误分析；
- 后续具备多游戏实例后再运行 GRPO rollout worker。

当数据规模增长到数十万以上样本或模型显著变大后，再用 `accelerate`/DDP/FSDP 扩展多卡训练。8 张消费级 4090 不带 NVLink，当前小模型不应为了“用满卡”引入分布式复杂度。

## 5. 训练产物契约

每一次训练运行应写入一个新目录，例如：

```text
/data/sts-agent/outputs/sft/qwen3-0.6b/sft-v0-r001/
  adapter_model.safetensors
  adapter_config.json
  tokenizer/
  training_args.json
  run_manifest.json
  metrics.json
  eval_by_seed.json
```

## 5.1 首次 SFT 运行

仓库提供 `tools/train_sft.py`，其训练输入严格匹配线上 selector：先按 `screen_type` 路由 trace 中的 observation，再拼接本步动态 `A1/A2/...` 合法动作表，输出只监督目标 alias 与结束标记。不会将动作 ID 直接训练为跨样本固定含义。

先从 bundle 目录运行 checksum，再执行一轮单卡 smoke test：

```bash
cd /data/wanghao/sts-agent
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python tools/train_sft.py \
  --dataset data/transfer/sft-v0-20260826-r2/dataset/teacher-sft.jsonl \
  --output outputs/sft/qwen3-0.6b/sft-v0-smoke \
  --max-train-samples 128 \
  --epochs 1
```

成功完成 smoke test 后，换一个新的输出目录执行完整首训：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_sft.py \
  --dataset data/transfer/sft-v0-20260826-r2/dataset/teacher-sft.jsonl \
  --output outputs/sft/qwen3-0.6b/sft-v0-r001 \
  --validation-runs 0 \
  --epochs 3 \
  --learning-rate 1e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --max-length 4096 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 2 \
  --save-total-limit 3
```

`--validation-runs 0` 会让全部 30 局进入训练，并关闭仅有训练损失的内部验证。训练器仍会在每个 epoch 保存 checkpoint；应以固定未见种子的实际游戏结果选择 checkpoint，不能按训练 loss 选择。当前 30 局已展开为 6,417 个逐步决策标签，首轮建议跑 3 epoch、rank 8、学习率 `1e-4`，而不是直接跑 10 epoch。模型使用 4-bit NF4 QLoRA、BF16（4090 支持时）和 gradient checkpointing。超过 `--max-length` 的样本会被记录并整条跳过，而不是截断导致合法动作表不完整。

只有在首轮实机评测确认仍欠拟合时，才做 10 epoch 对照：将学习率降至 `5e-5`，设置 `--save-total-limit 10`，并比较第 1、3、5、10 个 epoch 的固定种子结果。不要只因训练 loss 继续下降就采用后面的 checkpoint。

若希望用本地下载好的基座，可增加 `--model /data/wanghao/sts-agent/models/base/Qwen3-0.6B`；训练器会先校验相邻 bundle `manifest.json` 中的 JSONL SHA-256。输出目录必须是新的；断点继续才传入 `--resume-from-checkpoint <checkpoint-path>`。

`run_manifest.json` 至少记录：源码 commit、训练 bundle 名和 sha256、基础模型 ID 与 revision、tokenizer revision、训练参数、随机种子、代码版本和启动时间。只将这一目录同步回本机；基础模型和 optimizer state 留在训练机。

```bash
rsync -aP --checksum <user>@<host>:/data/sts-agent/outputs/sft/qwen3-0.6b/sft-v0-r001/ \
  outputs/sft/qwen3-0.6b/sft-v0-r001/
```

## 6. SFT 之后

在未见种子上先检查合法动作率、卡死率、平均层数、Act 进度和推理延迟，而不是只比较 training loss。当前 30 局全败的教师数据只能产生 `sft_base_v0`；在继续 GRPO 前，应补充更深层、不同角色和人工纠正的数据。

GRPO 的更新可以在 4090 机器进行，但 rollout 必须从冻结 checkpoint 开始并由真实游戏环境收集。当前最简单的闭环是：训练机导出 adapter -> 本机加载 adapter 采集固定种子 rollout -> bundle 回传训练机 -> 组内更新 -> 固定种子验证。不要在一局游戏中更新权重。
