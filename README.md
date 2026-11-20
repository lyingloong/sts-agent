# StS Agent

[![离线测试](https://github.com/lyingloong/sts-agent/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/lyingloong/sts-agent/actions/workflows/tests.yml)

把《Slay the Spire 1》的一局游戏录成可分析的动作轨迹，或让模型通过同一个接口游玩。StS Agent 支持人工录制、出牌建议、自动决策、固定种子评测，以及训练数据导出。

录制时，CommunicationMod 提供游戏状态，ActionRecorder 记录玩家实际执行的动作。StS Agent 按状态和动作 ID 将两者关联，每局生成一份 JSONL。在模型模式下，StS Agent 把局面和合法动作交给模型，再通过 CommunicationMod 执行所选动作。

## 录制第一局

以下流程使用 Windows PowerShell。默认的 `collect_human` 模式只需玩家正常游玩，无需 GPU、模型或 API 密钥。

### 安装

准备 Git、Python 3.10+ 和已安装的《Slay the Spire 1》。在路径不含空格的目录中克隆仓库，并使用可编辑安装，让游戏进程也能找到仓库中的配置、提示词和输出目录。

```powershell
git clone https://github.com/lyingloong/sts-agent.git
cd sts-agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\sts-agent-config.exe
```

最后一条命令会显示配置文件位置和当前模式。全新检出应显示 `collect_human`。

<details>
<summary>Linux 或 macOS 上安装离线工具</summary>

在克隆后的仓库根目录运行以下命令。游戏接入和 Overlay 主要在 Windows 验证，模型服务与训练主要在 Linux 验证。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/sts-agent-config
```

</details>

### 启动

1. 安装 ModTheSpire、BaseMod、CommunicationMod 和 ActionRecorder。在 ModTheSpire 中启用后三个 Mod，首次加载后退出游戏。[CommunicationMod](https://github.com/ForgottenArbiter/CommunicationMod) · [ActionRecorder 安装指南](https://github.com/lyingloong/sts-action-recorder)
2. 打开 `%LOCALAPPDATA%\ModTheSpire\CommunicationMod\config.properties`，设置下面三个值。将 `C\:/path/to/sts-agent` 替换为仓库的绝对路径，沿用正斜杠。CommunicationMod 按空白拆分命令，因此解释器路径不能含空格。

   ```properties
   command=C\:/path/to/sts-agent/.venv/Scripts/pythonw.exe -m sts_agent.runtime.controller
   runAtGameStart=true
   maxInitializationTimeout=300
   ```

3. 通过 ModTheSpire 重新启动游戏，开始一局并打出一张牌。CommunicationMod 会自动启动录制进程，人工录制无需打开 Overlay。

### 查看轨迹

在仓库根目录可以列出最近的对局文件。

```powershell
Get-ChildItem .\data\trajectories\human\human-*.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 3 Name,Length
```

`human-*.jsonl` 按局保存动作前的局面、合法动作和玩家选择。ActionRecorder 的原始事件另存在**游戏工作目录**的 `data/actionrecorder/`。如果仓库里只有 `communication-bridge-*.jsonl`，请检查 ActionRecorder 是否启用，以及它的发送端口与 StS Agent 的录制监听端口是否一致（默认 `127.0.0.1:8766`）。[轨迹字段](doc/DATA_AND_TRACE_SCHEMA.md) · [动作与状态的关联规则](doc/PROTOCOL.md)

## 切换模式

首次自定义配置时，将 `config/agent.example.json` 复制为同目录的 `agent.json`。在副本中设置 `active_mode`，将特定模式的参数放入 `mode_overrides`。本机配置由 Git 忽略。运行 `sts-agent-config --list` 可列出模式，`--mode 模式名` 可在切换前离线检查配置。修改生效需要重启游戏和正在使用的客户端。[配置模板](config/agent.example.json) · [配置示例与参数优先级](config/README.md)

| 想做什么 | 选用模式或工具 | 使用说明 |
| --- | --- | --- |
| 获取出牌建议，或交给本机、远端模型操作 | `candidate_assist_local`、`local_inference` 等，可用 Overlay 或终端交互 | [运行模式](doc/MODES.md) · [启动命令](doc/COMMANDS.md) |
| 用相同种子评测完整对局 | 托管 API 用 `baseline_api`，自建模型服务用 `baseline_inference`。先用 `sts-agent-evaluate-baseline --dry-run` 查看计划 | [配置后端](doc/BASELINE_INFERENCE.md) · [运行与计分](doc/BASELINE_EVALUATION.md) |
| 让 Codex 玩一局 A0 铁甲战士 | `collect_codex`，通过本地 MCP 服务连接控制器 | [Codex 游玩指南](doc/CODEX_PLAY_A0_IRONCLAD.md) |
| 导出训练数据或运行在线更新 | 从 ActionRecorder 数据集导出人类 SFT，或使用教师轨迹和远端训练工具 | [人类 SFT](doc/HUMAN_SFT_DATASET.md) · [教师数据与训练](doc/TRAINING_ON_4090.md) · [在线 RL](doc/REMOTE_ONLINE_RL.md) |

使用上表的短命令时，先激活虚拟环境，或直接调用 `.venv\Scripts\` 下的对应程序。交互控制和固定种子批量评测各自占用控制器的单一客户端连接。完整命令与模块名见[命令索引](doc/COMMANDS.md)。

## 开发与维护

[文档目录](doc/README.md) · [代码架构](doc/ARCHITECTURE.md) · [提示词示例](doc/POLICY_PROMPT_PREVIEW.md) · [变更记录](CHANGELOG.md)

项目继续维护游戏接入、录制与评测工具，原论文计划和配套实验已停止。[论文计划与历史结果](doc/PAPER_PLAN.md)

普通问题请在 GitHub Issues 反馈，安全问题请使用私密渠道。贡献指南列出了离线测试命令和提交流程。[反馈问题](https://github.com/lyingloong/sts-agent/issues) · [安全报告](SECURITY.md) · [贡献指南](CONTRIBUTING.md)

本仓库代码采用 MIT 许可证。游戏、第三方 Mod、模型权重和数据集各自遵循其许可。[MIT 许可证](LICENSE)
