# StS Agent

基于 CommunicationMod 的《Slay the Spire 1》Agent 框架。它把游戏接入、状态与合法动作、策略后端、UI、轨迹和学习流程组织在同一个 Python 包中；运行方式由一份配置选择，而不是由不同脚本分叉实现。

## 当前能力

人类数据流程：先在 ActionRecorder 提取通用轨迹数据集，再用本仓库构建细粒度 SFT。
完整命令、筛选职责和 prompt 预览说明见 [人类 SFT 数据集](doc/HUMAN_SFT_DATASET.md)。

- 支持原版四角色、A0-A20、随机或指定种子，以及同一游戏进程连续运行多局。
- 从 CommunicationMod 构建公开状态、合法动作、地图、战斗数值、牌组和历史动作；卡牌、药水、遗物和 Power 的效果说明优先取 ActionRecorder 扩展后的游戏运行时对象，敌人状态/意图取当前游戏状态。
- 支持人工控制、Candidate Assist、置信度自动执行、远程教师、Prompt Evolution、固定远端 Qwen 和在线 RL。
- 每局独立保存轨迹；可导出教师 SFT 数据和带校验和的迁移 bundle。
- 可接收独立 ActionRecorder Mod 的人类游戏动作，并用在线 agent 同一套适配器生成完整训练轨迹。
- 可通过本地 MCP 将 Codex 接入 controller，以 A0 战士配置单独采集 AI 游玩轨迹。

## 安装

```powershell
uv venv
uv pip install --python .venv\Scripts\python.exe -e .
uv pip install --python .venv\Scripts\python.exe -r requirements\policy.txt
```

训练机改用 `requirements/train.txt`。CUDA 版 PyTorch 应按机器驱动单独安装，不应从普通 PyPI 镜像误装 CPU wheel。

## 启动

CommunicationMod 的 `config.properties`：

```properties
command=D\:/Code/spire-mod/sts-agent/.venv/Scripts/pythonw.exe -m sts_agent.runtime.controller
runAtGameStart=true
maxInitializationTimeout=300
```

启动游戏后，在项目根目录打开 Overlay：

```powershell
.\.venv\Scripts\python.exe -m sts_agent.ui.overlay
```

也可使用纯文本客户端：

```powershell
.\.venv\Scripts\python.exe -m sts_agent.ui.terminal
```

controller 必须由 CommunicationMod 启动，一次只接受一个终端类客户端。主菜单输入 `s [角色] [进阶] [种子]`；种子可省略。运行模式由 [config/agent.json](config/agent.json) 的 `active_mode` 决定，模式专属参数位于 `config/modes/`；修改配置后重启游戏/controller 和 UI。

### 官方权重基线测试

选择 `baseline_inference` 可用远端常驻 vLLM 测试官方 Qwen3.8-27B BF16：本机仅控制游戏，
通过 SSH 通道请求单一动作；不启用训练或教师模式，评测轨迹与人工采集分开。
远端与本机使用同仓库的不同入口，准备命令和全部参数见 [基线推理](doc/BASELINE_INFERENCE.md)。

### Codex 自动游玩采集

将 `config/agent.json` 的 `active_mode` 设为 `collect_codex`，启动游戏使 CommunicationMod 拉起 controller，再在 Codex 会话中启用 `sts-agent-codex-mcp`。Codex 通过 MCP 获取当前模型可见 observation 和合法动作，并以动作 ID 提交选择；轨迹写入 `data/trajectories/codex/`，与人类数据分开。配置 MCP 的客户端步骤和逐步游玩规程见 [A0 战士 Codex 游玩说明](doc/CODEX_PLAY_A0_IRONCLAD.md)。

controller 会在向 MCP 提供决策状态前完成已有的环境自动动作。自动动作保留在轨迹和 `recent_actions` 中，但不会作为待选动作交给 Codex；Codex 不应重复执行它们。

### 人类游玩采集

人类直接在游戏界面操作、并采集轨迹时，将 `config/agent.json` 中的活动模式设为：

```json
"active_mode": "collect_human"
```

此模式使用 CommunicationMod 启动状态桥；不要再分别启动一个只收 ActionRecorder
事件的进程。CommunicationMod 的 `command` 使用统一入口（模式切换无需改它）：

```powershell
.\.venv\Scripts\pythonw.exe -m sts_agent.runtime.controller --config config/agent.json
```

并设置 `runAtGameStart=true`。同时启用 ActionRecorder 和 CommunicationMod。
CommunicationMod 的状态通过 stdin 进入 bridge；ActionRecorder 0.5 在实际发布处
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
- `post_state`：动作后稳定状态。

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
  commands/      模型下载和协议探针
config/          唯一运行配置及远端 RL 示例
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

## 验证

```powershell
.\.venv\Scripts\python.exe -m compileall -q sts_agent tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

完整端到端测试仍需启动 ModTheSpire。CommunicationMod 的协议约束见其[官方仓库](https://github.com/ForgottenArbiter/CommunicationMod)。

## 语义动作接口

策略层默认使用细粒度游戏动作。营火升级和商店删牌按入口、选牌、确认分别决策，
与人类采集及 SFT 数据保持同一粒度；模型输出当前合法动作编号，由控制器映射为命令。
历史宏动作代码只在显式请求粗粒度时使用。详细约定见 [`doc/SEMANTIC_ACTIONS.md`](doc/SEMANTIC_ACTIONS.md)。
