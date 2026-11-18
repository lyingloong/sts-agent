# StS Agent

StS Agent 通过 CommunicationMod 连接《Slay the Spire 1》，支持采集玩家操作、让模型选择合法动作，以及按固定种子评测完整对局。

仓库继续维护游戏接入、录制和评测工具。论文计划及配套跑批已于 2026-11-18 停止。

| 目标 | 从这里开始 |
| --- | --- |
| 记录自己的对局，保留每一步的合法动作和所选动作 | 按下面的安装与启动步骤使用默认的 `collect_human` 模式 |
| 让模型给建议、自动打牌，或连接 Codex | [选运行模式](doc/MODES.md) · [配置说明](config/README.md) |
| 在固定种子上比较模型表现 | [自建或托管模型服务](doc/BASELINE_INFERENCE.md) · [运行与计分](doc/BASELINE_EVALUATION.md) |
| 使用已有数据训练模型 | [导出人类 SFT 数据](doc/HUMAN_SFT_DATASET.md) · [训练环境](doc/TRAINING_ON_4090.md) |

## 安装

离线工具需要 Git 和 Python 3.10+。请保留源码目录并使用可编辑安装，程序会从这份检出读取
`config/`、`prompts/` 并在 `data/` 下写入输出。可以从游戏安装目录以外的位置克隆项目。

Windows PowerShell：

```powershell
git clone https://github.com/lyingloong/sts-agent.git
cd sts-agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\sts-agent-config.exe
```

Linux 或 macOS 可在克隆后的仓库根目录运行：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/sts-agent-config
```

全新检出的配置摘要应显示 `collect_human`。离线检查及人工采集使用 Python 标准库。
使用本机 Candidate Assist 时再安装 [策略依赖](requirements/policy.txt) 和适合显卡的 PyTorch。
训练机依赖见[训练机指南](doc/TRAINING_ON_4090.md)。本机游戏接入与 Overlay 主要在 Windows 验证，远端服务和训练主要在 Linux 验证。

需要切换模式时，将 [agent.example.json](config/agent.example.json) 复制为同目录的
`agent.json`，在副本中修改 `active_mode`。Git 会忽略本机配置。如果从仍跟踪
`config/agent.json` 的旧提交升级，请先备份该文件，更新后放回原路径。

## 启动

在游戏端安装《Slay the Spire 1》、ModTheSpire、BaseMod 和 [CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod)。
要录制玩家操作，还要启用 [ActionRecorder](https://github.com/lyingloong/sts-action-recorder)。它提供卡牌等实体的运行时说明及动作与状态的对应关系。通过 ModTheSpire 启用这些 Mod。

Windows 上首次载入 CommunicationMod 后，退出游戏，编辑
`%LOCALAPPDATA%\ModTheSpire\CommunicationMod\config.properties`。把下面的
`C\:/path/to/sts-agent` 换成虚拟环境所在仓库的绝对路径。CommunicationMod 按空白拆分
`command`，请将示例中的虚拟环境放在无空格路径下。路径中的目录分隔符使用正斜杠。

```properties
command=C\:/path/to/sts-agent/.venv/Scripts/pythonw.exe -m sts_agent.runtime.controller
runAtGameStart=true
maxInitializationTimeout=300
```

保存配置后通过 ModTheSpire 启动游戏，在默认模式下直接游玩。在项目的 `data/trajectories/human/`
可以看到桥接进程的 `communication-bridge-*.jsonl`，开始录制对局后会出现 `human-*.jsonl`。
若只有桥接日志，请检查 ActionRecorder 是否启用，以及两端的录制端口是否一致（默认 `127.0.0.1:8766`）。
ActionRecorder 的原始日志位于游戏工作目录的 `data/actionrecorder/`。
切换到需要交互控制的模式后，可在项目根目录打开 Overlay：

```powershell
.\.venv\Scripts\python.exe -m sts_agent.ui.overlay
```

也可使用纯文本客户端：

```powershell
.\.venv\Scripts\python.exe -m sts_agent.ui.terminal
```

controller 由 CommunicationMod 启动，一次只接受一个终端类客户端。主菜单输入 `s [角色] [进阶] [种子]`，种子可省略。切换 `active_mode` 后重启游戏/controller 和 UI。各模式的参数见[运行配置](config/README.md)。

### 固定种子评测

选择 `baseline_inference` 可用远端常驻模型服务测试官方 Qwen3.8-27B BF16：本机仅控制游戏，
通过 SSH 通道请求单一动作；不启用训练或教师模式，评测轨迹与人工采集分开。
远端与本机使用同仓库的不同入口，准备命令和全部参数见 [基线推理](doc/BASELINE_INFERENCE.md)。
进度分、通关率及实验比较口径见 [基线评测方案](doc/BASELINE_EVALUATION.md)。

已有托管 API 账户时，可选择 `baseline_api`，按[运行配置](config/README.md)设置模型名、
API 地址和本机密钥文件路径。示例使用中国科大网关，个人密钥保存在 Git 忽略的文件中。

选好模式后，在仓库根目录用下面的命令预览种子、复用结果和待跑列表。这一步可离线执行。

```powershell
.\.venv\Scripts\python.exe -m sts_agent.commands.evaluate_baseline --dry-run
```

实际评测需要可用的模型后端。重启游戏和 controller，停在普通主菜单并关闭 Overlay 或终端客户端，再运行：

```powershell
.\.venv\Scripts\python.exe -m sts_agent.commands.evaluate_baseline
```

种子与预算位于 [固定测试集配置](config/evaluation/a0_ironclad_10.json)。脚本串行完成 10 局 A0 战士，
每局正常终局并返回主菜单后才开下一局；结果写入 `data/experiments/baseline_batches/`。
技术故障或人工接管会停止批次，不按死亡计分、不自动重跑。详见[批量运行说明](doc/BASELINE_EVALUATION.md#固定-10-种子批量测评)。

### Codex 自动游玩采集

将 `config/agent.json` 的 `active_mode` 设为 `collect_codex`，启动游戏使 CommunicationMod 拉起 controller，再在 Codex 会话中启用 `sts-agent-codex-mcp`。Codex 通过 MCP 获取当前模型可见 observation 和合法动作，并以动作 ID 提交选择；轨迹写入 `data/trajectories/codex/`，与人类数据分开。配置 MCP 的客户端步骤和逐步游玩规程见 [A0 战士 Codex 游玩说明](doc/CODEX_PLAY_A0_IRONCLAD.md)。

controller 会在向 MCP 提供决策状态前完成已有的环境自动动作。自动动作保留在轨迹和 `room_history` 中，但不会作为待选动作交给 Codex；Codex 不应重复执行它们。

### 人类游玩采集

模型可见的状态、细粒度动作和历史格式见 [模型输入与动作空间](doc/MODEL_OBSERVATION_AND_ACTIONS.md)，
完整输入示例见 [Prompt Preview](doc/POLICY_PROMPT_PREVIEW.md)。所有房间统一显示最近 30 步历史，
二级选牌不清空历史，翻牌的已公开牌面还保留在当前棋盘记忆中。

在默认的 `collect_human` 模式下，controller 会自行启动状态桥。
CommunicationMod 的状态通过 stdin 进入 bridge；ActionRecorder 事件协议 0.5 在实际发布处
编号，同步把完整状态和动作事务保存到本地并通过 `recording.listen` 的 TCP 端口
发送副本。bridge 按 `before_state_id` 精确匹配，动作接受后等待可决策状态或下一
动作入口状态，不按时间戳猜测。人类采集保留细粒度入口/选牌/确认动作，未匹配
或无法唯一映射的动作不会作为可用SFT样本。TCP闲置、重连和终局有独立处理。

完整轨迹写入 `data/trajectories/human/`。每步包含：

- `pre_state`：完整规范化状态与历史（实体效果只取当前游戏状态中的运行时字段）；
- `policy_observation`：模型实际可见的状态，完整地图仅在地图界面保留；
- `legal_actions`：使用在线 agent 的目标展开规则，但保留人类操作的细粒度锻造/删牌入口，不应用策略导航过滤；
- `action` 和 `selected_action_id`：映射到上述动作空间的人类动作；
- `raw_action`：ActionRecorder 捕获的原始游戏内动作；
- `post_state`：动作后观测；结算动画可能尚未完全结束，不应无条件作为动作效果真值。

ActionRecorder 事件格式 0.5 使用 `state_published` 与
`action_begin`/`action_accepted`/`action_rejected` 事务。Java Mod 保存并引用
CommunicationMod 实际发布的状态，不维护另一份转换器快照。轨迹还保留
`observation_before/after`、`raw_message_before/after` 和原始 `available_actions`。
旧的不完整轨迹不进入当前采集链路。

只有 `usable_for_sft=true` 的步骤才同时具备有效状态和可验证的合法动作标签，应进入后续 SFT 数据集。若未同时启用 CommunicationMod，接收器会在终端明确告警并将对应步骤标记为不可训练，而不是静默生成空状态样本。

TCP 断开不影响 ActionRecorder 的本地原始文件；bridge 恢复后会继续为新的事务
生成轨迹。已有不完整旧日志不自动迁移，避免把错误的动作前状态带入训练集。

## 数据与训练

生成迁移 bundle：

```powershell
.\.venv\Scripts\python.exe -m sts_agent.learning.training_bundle `
  --output data\transfer\sft-v0-<YYYYMMDD>-r1 `
  --include-traces
```

SFT：

```bash
.venv/bin/python -m sts_agent.learning.sft \
  --dataset data/transfer/<bundle>/dataset/teacher-sft.jsonl \
  --model /path/to/Qwen \
  --output outputs/sft/<run>
```

训练与远端运行参数见 [训练机指南](doc/TRAINING_ON_4090.md) 和 [远端在线 RL](doc/REMOTE_ONLINE_RL.md)。API key、轨迹、数据集、模型和运行时状态均被忽略，不应提交。

## 目录

```text
sts_agent/
  core/          公共 observation/action/policy 契约与模式校验
  runtime/       CommunicationMod controller 和运行协调
  game/sts1/     StS1 状态与动作适配器
  policy/        本地 Qwen、远程教师和策略 Prompt
  learning/      SFT、Prompt Evolution、反馈归因和在线 RL
  ui/            Overlay 与文本客户端
  commands/      模型下载、协议探针和批量测评
config/          运行模式、固定测试集及远端 RL 示例
prompts/         角色与场景 skill
requirements/    推理与训练依赖
tests/           无需启动游戏的单元测试
doc/             架构、协议和实验说明
```

## 核心文档

- [架构与依赖边界](doc/ARCHITECTURE.md)
- [命令入口](doc/COMMANDS.md)
- [运行配置](config/README.md)
- [运行模式](doc/MODES.md)
- [通信协议](doc/PROTOCOL.md)
- [数据与轨迹契约](doc/DATA_AND_TRACE_SCHEMA.md)
- [策略 Prompt](doc/POLICY_PROMPT.md)
- [教师 SFT 与 GRPO](doc/TEACHER_SFT_GRPO.md)
- [Prompt Evolution](doc/PROMPT_EVOLUTION.md)
- [固定种子评测与计分](doc/BASELINE_EVALUATION.md)
- [历史研究计划](doc/PAPER_PLAN.md) · [30 种子批次记录](doc/EXPERIMENT_HANDOFF.md)
- [变更记录](CHANGELOG.md)
- [Agent 协作约定](AGENTS.md)

## 验证

```powershell
.\.venv\Scripts\python.exe -m compileall -q sts_agent tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

完整端到端测试仍需启动 ModTheSpire。CommunicationMod 的协议约束见其[官方仓库](https://github.com/ForgottenArbiter/CommunicationMod)。
常规问题可提交到 [GitHub Issues](https://github.com/lyingloong/sts-agent/issues)。参与开发见[贡献指南](CONTRIBUTING.md)，安全问题请按[私密报告流程](SECURITY.md)处理。

## 语义动作接口

策略层默认使用细粒度游戏动作。营火升级和商店删牌按入口、选牌、确认分别决策，
与人类采集及 SFT 数据保持同一粒度；模型输出当前合法动作编号，由控制器映射为命令。
历史宏动作代码只在显式请求粗粒度时使用。详细约定见 [`doc/SEMANTIC_ACTIONS.md`](doc/SEMANTIC_ACTIONS.md)。

## 许可证

本仓库代码采用 [MIT License](LICENSE)。游戏、第三方 Mod、模型权重和数据集的许可与分发要求另行适用。
