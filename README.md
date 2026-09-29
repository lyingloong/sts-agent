# StS Agent

基于 CommunicationMod 的《Slay the Spire 1》Agent 框架。它把游戏接入、状态与合法动作、策略后端、UI、轨迹和学习流程组织在同一个 Python 包中；运行方式由一份配置选择，而不是由不同脚本分叉实现。

## 当前能力

- 支持原版四角色、A0-A20、随机或指定种子，以及同一游戏进程连续运行多局。
- 从 CommunicationMod 构建公开状态、合法动作、地图、战斗数值、牌组、历史动作和静态知识。
- 支持人工控制、Candidate Assist、置信度自动执行、远程教师、Prompt Evolution、固定远端 Qwen 和在线 RL。
- 每局独立保存轨迹；可导出教师 SFT 数据和带校验和的迁移 bundle。
- 静态知识快照和人工审核的实体经验位于 `data/knowledge/`，并受 Git 跟踪。
- 可接收独立 ActionRecorder Mod 的人类游戏动作，并用在线 agent 同一套适配器生成完整训练轨迹。

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
command=D\:/Code/spire-mod/sts-agent/.venv/Scripts/python.exe -m sts_agent.runtime.controller
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

controller 必须由 CommunicationMod 启动，两个客户端不能同时连接。主菜单输入 `s [角色] [进阶] [种子]`；种子可省略。具体模式由 [config/agent.json](config/agent.json) 的 `framework` 决定，修改配置后重启游戏/controller 和 UI。

### 人类游玩采集

人类直接在游戏界面操作时，将 `config/agent.json` 的 framework 设置为：

```json
{
"framework": {
  "game": "sts1",
  "control_mode": "manual",
  "policy_backend": "none",
  "learning_method": "collect_human"
}
}
```

此模式不启动上面的 CommunicationMod controller。启动统一配置驱动的 ActionRecorder 轨迹接收端：

```powershell
.\.venv\Scripts\python.exe -m sts_agent.recording.action_recorder --config config/agent.json
```

然后通过 ModTheSpire 同时启用 ActionRecorder 和 CommunicationMod。CommunicationMod 只作为游戏进程内的状态提供者，其 `runAtGameStart` 应设为 `false`。ActionRecorder 的 TCP 地址由 [config/agent.json](config/agent.json) 的 `recording.listen` 配置。

完整轨迹写入 `data/trajectories/human/`。每步包含：

- `pre_state`：完整、规范化并注入知识和历史的审计状态；
- `policy_observation`：模型实际可见的状态，完整地图仅在地图界面保留；
- `legal_actions`：与在线 agent 相同的目标展开语义动作列表；
- `action` 和 `selected_action_id`：映射到上述动作空间的人类动作；
- `raw_action`：ActionRecorder 捕获的原始游戏内动作；
- `post_state`：动作后稳定状态。

ActionRecorder 事件格式 0.2 直接输出“锻造指定卡牌”和“删除指定卡牌”的完整语义决策，携带进入流程前的状态；接收端直接匹配动作 ID。中间的 `decision_started/detail/cancelled` 属于 Mod 的辅助记录，不生成训练样本，也不参与重建动作。旧版细粒度日志应单独迁移，不会在这里自动拼接。

只有 `usable_for_sft=true` 的步骤才同时具备有效状态和可验证的合法动作标签，应进入后续 SFT 数据集。若未同时启用 CommunicationMod，接收器会在终端明确告警并将对应步骤标记为不可训练，而不是静默生成空状态样本。

TCP 断开不影响 ActionRecorder 的本地原始文件。之后可以离线补做同样的转换：

```powershell
.\.venv\Scripts\python.exe -m sts_agent.recording.action_recorder `
  --input "D:\Steam\steamapps\common\SlayTheSpire\data\actionrecorder\run-....jsonl"
```

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
  game/sts1/     StS1 适配器与静态知识补全
  policy/        本地 Qwen、远程教师和策略 Prompt
  learning/      SFT、Prompt Evolution、反馈归因和在线 RL
  ui/            Overlay 与文本客户端
  commands/      模型下载、知识刷新和协议探针
config/          唯一运行配置及远端 RL 示例
data/knowledge/  版本化静态知识快照
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
- [实体经验知识库](doc/KNOWLEDGE_EXPERIENCE.md)

## 验证

```powershell
.\.venv\Scripts\python.exe -m compileall -q sts_agent tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

完整端到端测试仍需启动 ModTheSpire。CommunicationMod 的协议约束见其[官方仓库](https://github.com/ForgottenArbiter/CommunicationMod)。

## 语义动作接口

策略层使用按场景定义的语义动作，而不是直接学习 Communication Mod 的按钮和下标。例如营火会暴露 `CAMPFIRE:REST` 或 `CAMPFIRE:SMITH:card=<uuid>`，商店会暴露 `SHOP:BUY_CARD:<id>`、`SHOP:PURGE:card=<uuid>`。控制器负责把一个语义动作展开为选择、确认等底层协议步骤，并只将完整流程记为一个训练决策。详细约定见 [`doc/SEMANTIC_ACTIONS.md`](doc/SEMANTIC_ACTIONS.md)。
