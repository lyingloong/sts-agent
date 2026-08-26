# StS Agent

《Slay the Spire 1》结构化游戏 Agent。项目用 CommunicationMod 采集可回放的状态-合法动作-结果轨迹，以远程教师构建冷启动 SFT，并为后续离线评测和 GRPO 保留数据契约。

当前阶段是“本地采集 + 远程 8x RTX 4090 训练”：游戏与 CommunicationMod 继续运行在本机，训练机只接收版本化数据集和模型产物。

## 当前能力

- 支持原版四角色、A0-A20、固定/随机显示种子，以及单游戏进程中的连续多局采集。
- controller 对 CommunicationMod 的状态进行规范化，提供完整公开地图、战斗意图、牌组、合法动作、最近动作和静态知识库补全。
- 支持人工终端、透明 Overlay、Qwen Candidate Assist/置信度自动执行，以及 OpenAI 兼容远程教师采集。
- 轨迹按局保存；教师 SFT 导出只接受稳定、合法且 `source=teacher_auto` 的动作。
- `data/knowledge/` 是受 Git 跟踪的 StS1 静态知识快照；轨迹、数据集、模型和运行时文件均不提交。

`tools/train_sft.py` 已实现第一版 SFT；固定种子游戏评测、GRPO rollout worker 和远程推理 endpoint 尚未实现。训练逻辑不应放入 `communication_controller.py`。

## 快速入口

### 本地采集

1. 在 CommunicationMod 的 `config.properties` 设置 controller：

```properties
command=D\:/miniconda3/python.exe D\:/Code/spire-mod/sts-agent/tools/communication_controller.py
runAtGameStart=true
maxInitializationTimeout=300
```

2. 启动游戏后运行一个控制客户端：

```powershell
python tools\overlay_terminal.py
# 或 python tools\human_terminal.py
```

3. 在主菜单输入 `s [character] [ascension] [seed]`，例如 `s IRONCLAD 0 4QTKQ7VBMFX47`。游戏内输入 `h` 查看动作帮助。

`communication_controller.py` 必须由 CommunicationMod 启动；Overlay 与文本终端不能同时连接。配置位于 [config/agent.json](config/agent.json)，API 密钥只放在被忽略的 `*_api.txt` 文件中。

### 导出与迁移训练数据

```powershell
python tools\prepare_training_bundle.py `
  --output data\transfer\sft-v0-<YYYYMMDD>-r1 `
  --include-traces
```

该命令生成可校验 bundle：SFT JSONL、可选的原始 action traces、`manifest.json` 和 `SHA256SUMS`。不包含 API 密钥、`data/raw/`、模型缓存、`.venv` 或 `.uv-cache`。完整流程见 [训练机与迁移指南](doc/TRAINING_ON_4090.md)。

训练入口是 `tools/train_sft.py`：它复用线上 alias selector prompt，按完整 run 切分验证集，并只训练 LoRA adapter。训练机运行方式见迁移指南中的“首次 SFT 运行”。

## 目录

```text
config/                 运行时配置
data/knowledge/         受版本控制的静态 StS1 知识快照
data/trajectories/      本地 action trace，忽略
data/raw/               原始 CommunicationMod 消息，仅排错，忽略
data/datasets/          导出的训练数据，忽略
data/transfer/          可迁移训练 bundle，忽略
doc/                    架构、协议、训练与设计文档
tests/                  不依赖游戏的单元测试
tools/                  controller、知识库、导出与客户端工具
```

## 文档

- [实施计划](doc/PLAN.md)
- [通信协议与数据 schema](doc/PROTOCOL.md)
- [教师 SFT 与 GRPO 路线](doc/TEACHER_SFT_GRPO.md)
- [8x RTX 4090 训练机与迁移指南](doc/TRAINING_ON_4090.md)
- [在线纠正式模仿学习设计参考](doc/ONLINE_DAGGER.md)
- [教师场景化提示词说明](doc/TEACHER_PROMPTS.md)
- [数据目录和保留策略](data/README.md)

## 验证

```powershell
python -m py_compile tools\communication_controller.py tools\teacher_client.py tools\export_teacher_sft.py tools\prepare_training_bundle.py tools\human_terminal.py tools\overlay_terminal.py
python -m unittest discover -s tests -v
```

完整游戏测试仍需启动 ModTheSpire。CommunicationMod 协议限制见其[官方仓库](https://github.com/ForgottenArbiter/CommunicationMod)。
