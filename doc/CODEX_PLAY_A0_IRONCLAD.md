# Codex 游玩说明：Slay the Spire 1 · A0 铁甲战士

这个文件供在 sts-agent 仓库中新开的 Codex 会话读取。Codex 通过本地 `sts-agent` MCP 工具操作游戏；游戏状态和合法动作以 MCP 返回为准，不要用屏幕猜测动作。

## 开始前

1. 在 `config/agent.json` 中设置 `"active_mode": "collect_codex"`。Codex 专属选项在 `config/modes/collect_codex.json`；共享 controller/Overlay 设置仍放在 `agent.json`。

2. `config/modes/collect_codex.json` 默认使用 `IRONCLAD`、A0 和随机种子；如需固定种子，只改该 profile 的 `codex.seed`。Codex 模式不启用内部教师的实体经验问答。
3. 安装可选 MCP 依赖：

   ```powershell
   uv pip install --python .venv\Scripts\python.exe -r requirements\codex.txt
   ```

4. 检查 `%LOCALAPPDATA%/ModTheSpire/CommunicationMod/config.properties`：`runAtGameStart=true`，`command` 指向项目 `.venv/Scripts/pythonw.exe -m sts_agent.runtime.controller`。Windows properties 路径使用正斜杠，盘符冒号写作 `\:`。如果此前用 ActionRecorder 人类采集，`runAtGameStart` 可能仍是 `false`；旧的 `tools/communication_controller.py` 路径已失效。先完成配置，再通过 ModTheSpire 启动启用了 CommunicationMod 的游戏。controller 会在配置的 localhost 端口等待 MCP 客户端。
5. 不要同时启动 sts-agent Overlay 或 terminal；controller 当前一次只接受一个客户端。Codex MCP 桥接是唯一客户端。

在本机 Codex 的 MCP 配置中注册 `sts-agent-codex-mcp`。Windows 示例（把占位符替换为本机仓库绝对路径）：

```toml
[mcp_servers.sts_agent]
command = "<STS_AGENT_ROOT>/.venv/Scripts/pythonw.exe"
args = ["-m", "sts_agent.codex_mcp", "--config", "config/agent.json"]
cwd = "<STS_AGENT_ROOT>"
```

Windows 中这两个后台入口使用 `pythonw.exe`，避免启动控制台窗口。MCP 和 CommunicationMod 都通过管道提供标准输入输出，因此协议仍正常工作；在终端手动排错时使用 `python.exe` 以查看报错。修改启动器后，下次启动对应进程时生效。

注册后重启/新建 Codex 会话，并确认工具列表中出现 `get_observation`、`start_a0_ironclad`、`execute_action` 和 `wait_for_observation`。Codex MCP 的配置方式随客户端而异；如果当前 Codex 客户端没有加载本地 MCP，不要假装已连接或退回到盲目控制。

若 MCP 显示 `failed`，先检查 controller 是否已启动：PowerShell 执行 `Get-NetTCPConnection -State Listen -LocalPort 8765`（使用自己的 `server.port`）。没有监听时，检查上述 CommunicationMod 设置及游戏目录的 `communication_mod_errors.log`。游戏窗口存在不代表 controller 已启动；修改配置后需要重新启动游戏。当前 MCP 桥接在初始化时连接 controller，因此应先让 controller 监听，再启动 Codex MCP。

读取和等待观察的工具带有 MCP 只读标注，但工具审批仍由宿主客户端决定，标注不会绕过审批。工具返回 `submitted=true` 仅表示请求已发往 controller，不代表游戏已经执行；必须等到下一观察才可确认。若 controller 报错退出，先检查上述日志、修复并重启游戏，再重新连接 MCP，不要盲目重发 START。

轨迹输出目录会在首次开局时自动创建。客户端断开时，controller 在当前决策处等待重新连接，不自动重放游戏动作；MCP 断线后不会继续返回缓存的旧状态。重新连接后应先读取状态，再决定是否继续。

## 游玩循环

1. 调用 `get_observation` 查看当前状态。`can_start_run=true` / `game_status="ready_to_start"` 表示游戏明确允许开局：其依据是 CommunicationMod 同时报告 `in_game=false`、`ready_for_command=true`，且 `available_commands` 包含 `start`。此时通过 `start_a0_ironclad` 开始本次授权的对局；除非配置了固定种子，否则使用随机种子。主菜单通常没有 `game_state`，因此 `screen_type`、`screen_name` 为 `null` 且 `legal_actions=[]` 是正常的，不应据此停止。`in_game=false` 本身不足以允许开局；若 `can_start_run=false` 则报告当前状态，不猜测。
2. 每次只针对最新 observation 决策。优先从 `legal_actions` 中选择一项，并把该 observation 的 `state_id` 原样传给 `execute_action`。
3. 每提交一个动作后必须调用 `wait_for_observation(after_state_id=上一步的 state_id)`，等 controller 返回新状态后再继续。开局工具也会返回提交时的 `state_id`，同样应等待新状态。超时后先重读状态；不要重复提交上一动作。
4. 先判断开局能力和终局。只有在局内仍需决策却没有合法动作、出现错误状态或状态无法解释时，停止操作并报告。待开局时空 `legal_actions` 不构成异常；START 通过专用开局工具执行，不在局内动作列表中。
5. 根据游戏实际终局判断结果。达到本次目标后停止操作；不要在终局菜单上误开下一局。

## 决策原则

- 目标是尽可能稳定地完成整局，而不是只最大化当前回合伤害。战斗中按敌人意图权衡伤害与格挡，避免可以合理防止的失血；同时不要为了“防御”而放任危险敌人拖长战斗。
- 只使用 observation、`recent_actions`、牌组/遗物/药水信息、地图（仅地图场景提供）和合法动作中实际公开的信息。不臆测隐藏牌、事件结果或敌人未知意图。
- 结合当前角色、牌组、遗物、药水、生命、金币、钥匙和地图路线作选择；卡牌奖励重视当前牌组缺少的功能与近期精英/Boss 压力，不机械地总是拿牌或总是跳过。
- 每个动作都必须来自最新的合法动作列表。不要手工编写通信命令、伪造动作 ID，或重放历史动作。

## controller 自动动作与轨迹

controller 已有的简单环境动作（例如奖励领取流程中的自动领取/打开，以及合法动作仅剩一步时的推进）会在 MCP 决策状态出现前先执行。它们已写入 sts-agent 轨迹和 `recent_actions`，来源是环境自动处理，不是 Codex 的选择。**不要再次提交这些动作，也不要把它们描述成自己选择的动作。** 对需要策略判断的剩余选择，Codex 仍通过 MCP 正常决策。

每局轨迹和 run-events 单独写入 `codex.output_dir`，默认 `data/trajectories/codex/`；`run_config.actor` 标记为 `codex`。失败或中断的局也保留作审计，后续构建“胜局数据集”时再按终局结果筛选，不要删除或伪标胜利。

## 新会话启动提示

> 请先完整读取并遵循 `doc/CODEX_PLAY_A0_IRONCLAD.md`，通过 sts_agent MCP 接管游戏。先调用 get_observation：若 in_game=true 就继续当前局；若 can_start_run=true 就用 start_a0_ironclad 开始一局随机种子的 A0 铁甲战士。待开局时场景字段为 null、legal_actions 为空是正常的，以 can_start_run 为准。提交开局或游戏动作后，使用提交前的 state_id 等待新状态，再作下一步决策。只使用当前合法动作，遵守自动动作规则。持续游玩到本局结束；连接或状态异常时停止报告。结束后汇报种子、终局和轨迹位置，不再自动开第二局。
