# 通信协议与数据格式

## CommunicationMod 配置

CommunicationMod 启动外部进程，进程 stdout 只能用于协议。controller 和人类
采集 bridge 都在启动时发送 `ready`。controller随后输出动作命令，采集bridge则只监听状态；诊断
写入 stderr 和项目日志。

```properties
command=C\:/path/to/sts-agent/.venv/Scripts/pythonw.exe -m sts_agent.runtime.controller
runAtGameStart=true
```

将示例路径替换为本机虚拟环境的绝对路径，`command` 中的路径需避开空格。
在 `collect_human` 模式下，同一入口会启动 `sts_agent.recording.communication_bridge`。

此 bridge 只发送一次 `ready` 握手，不提交游戏命令或轮询 `WAIT`；CommunicationMod 的状态是唯一 observation 来源；
ActionRecorder 0.5 通过 `recording.listen` 端口发送完整状态及动作事务副本。
`action_begin.before_state_id` 对应同一状态发布结果的 `recorder_state_id`；配对
不依赖两通道接收时间。原始本地日志也保留完整状态，能独立审计。
当前采集链路只
允许这一套 bridge，旧的独立 TCP 接收器已移除。

Windows 后台入口使用 `pythonw.exe`，避免启动时弹出控制台窗口；CommunicationMod 提供的 stdin/stdout/stderr 管道仍用于协议和诊断。手动在终端排错时使用 `python.exe`。

Recorder 的主动采集快照复用该管道，并带 `recorder_snapshot_only=true` 和
`recorder_snapshot_kind=action_before/execution_before/effects_settled`。
controller 会保存它们，但不能用其 ready_for_command 确认动作或推进决策；
人类采集 bridge 则仍用于精确状态 ID 配对。正常 CM 发布标记为 false。
游戏动作的错误包不包含有效观察：执行控制器只发一次 `STATE`，拿新观察后再按恢复状态机处理。
STATE 本身失败时，控制器按本地实际提交的命令识别协议故障，保存错误回执和命令 ID，
通知已连接的界面后以非零状态退出；不继续刷新、重试游戏动作或使用旧观察猜测操作。
即使解码失败回执缺少 command 字段，这一停止保护仍然生效。

### 控制命令确认与恢复

启用新版 ActionRecorder 后，controller 使用 `AR1 <command_id> <state_id> <base64_command>`
封装命令。`CommandChannel` 是单一提交与确认通道：同一时刻只允许一个在途命令，
旧状态、其他命令的回执和采集快照不能确认当前命令。只有 ID 与命令匹配、处于可决策状态，
且队列执行不再为 `pending` 的回执才允许继续。超时或关联命令不一致时停止，不猜测是否执行。
原版 CommunicationMod 仍支持未关联的旧协议，但不能据其错误包自动重试。

`sts_agent.runtime.recovery.CommandRecovery` 独立管理恢复，控制器不会匹配错误文案来决定重试：

- 当前关联回执的 `error_code` 为 `stale_observation` 或 `state_changed`，并明确给出
  `status=rejected`、`execution_status=not_executed` 时，刷新后允许自动重新决策。
  丢弃原模型候选，按新状态重新计算合法动作；不是重放旧命令。
- 其他错误、旧版无结构化证明的错误、执行中失败或执行情况未知时，刷新后暂停并交给人工。
- 连续自动恢复最多 3 次；成功确认一个非 `STATE` 游戏命令后才清零（包括 `START` 和内部续接动作，
  不能在提交时提前清零）。耗尽后只能人工操作，
  反复输入 `state` 不能绕过这个上限。
- 人工输入 `state`（或局内输入 `s`）是只读控制操作：可刷新状态并显式恢复未耗尽的自动决策，
  不作为游戏动作写入 trace、history 或 SFT 样本。模型提交前的只读校验不会清除候选。

每个被拒绝的游戏动作仍保存 `accepted=false`、`command_confirmation` 和 `recovery`；
恢复事件写入 `run-events`。执行确认不等于动画或所有效果已经结算。

Mod 的命令校验使用独立状态投影，不改录制的完整 JSON：手牌、敌人、HP、能量、费用、
升级、牌堆成员与顺序、未知字段等保持精确比较；只忽略抽牌/弃牌/消耗堆中有基础值佐证的
临时伤害/格挡，以及可由已知 `D/B/M` 模板和数值重建的描述变化。
这允许游戏重置非手牌的临时牌面值，而不会把真正的决策状态变化当作相同状态。
完整契约见 ActionRecorder 的 `docs/STATE_SCHEMA.md`。

### 基线批量客户端协议

`sts_agent.commands.evaluate_baseline` 使用原有 `server.host/port` JSONL TCP 接口，
与 Overlay/终端互斥。它只协调开局与汇总，不选择局内动作。

`remote_api` 后端的 `view` 与 `model_update` 增加结构化 `baseline`：

```json
{
  "protocol_version": 1,
  "mode": "baseline_inference",
  "experiment_dir": "<本机绝对路径>",
  "model": "Qwen/Qwen3.8-27B",
  "run_id": "<当前 run ID 或 null>",
  "ready_for_command": true,
  "can_start_run": false,
  "batch_attached": true,
  "controller_error": null,
  "run_summary": "<summary.json 中本局对象或 null>"
}
```

`run_summary` 是独立复制的对象，不发送体积很大的 `terminal_state`。
菜单的场景字段为空是正常的，是否能开局依据 `in_game=false`、明确 readiness 和
CommunicationMod 的 `start` 能力，不能猜测场景名或解析格式化 UI 文本。

监督握手：主菜单发送 `{"type":"baseline_batch_attach","protocol_version":1}`，
等待新的 `view` 中 `baseline.batch_attached=true`。只允许 baseline 的可开局菜单，
不会执行任何游戏动作或绕过控制器已有暂停。随后发送普通
`{"type":"input","text":"s IRONCLAD 0 <seed>"}`。局内 `model_update` 仅供读取，
不应向 controller 回传动作或重复自动命令；局内要求人工输入的 `view` 是批次停止条件。

正常终局的摘要会一直保留到主菜单。客户端验证对应 run ID、种子、角色/进阶、
最终自主评分后，必须等 `can_start_run=true` 的菜单 `view` 才能开下一局。
结束时发送 `{"type":"baseline_batch_detach"}`，等待菜单确认
`batch_attached=false` 后正常关闭连接。

只有显式 attach 的客户端受 `BatchClientGuard` 监督；普通 UI 关闭不改变已有行为。
批量客户端意外断开或 Ctrl+C 时，controller 在自动环境动作、模型请求返回、
STATE 复核及提交前的安全边界检查连接，清掉待执行模型候选并暂停。
已经提交的命令不能撤销。失败后须人工检查并通过显式刷新/合法动作恢复，
重连/重新 attach 本身不能清掉失败保护。详情见[批量测评](BASELINE_EVALUATION.md#固定-10-种子批量测评)。

### 人类采集格式

`human-*.jsonl` 为每局独立文件，保留 `observation_before/after`、原始
`raw_message_before/after`、`available_actions`、`raw_action`，以及下游派生的
`pre_state/post_state/policy_observation/legal_actions/action`。配对结果为
`state_match.match_status=exact_state_id`，可训练性由 `usable_for_sft` 独立判断。
营火锻造/商店删牌不折叠为房间级动作；不以agent策略过滤人类重复进入或取消选择。

Recorder 0.1.5 的奖励领取优先按 `reward_id + reward_index` 在动作前完整奖励列表中校验，
再映射 CommunicationMod 的 `CHOOSE n`，不能用多个同名奖励的标签猜测。卡牌奖励子操作
校验 `source_reward_id`。原始奖励身份、关联钥匙和卡牌来源字段始终保留。

`run_resumed` 延续原 run_id 与文件，携带新的 `segment_id` 和 checkpoint 来源。bridge 在
恢复边界关闭旧事务，不把读档后状态当作旧动作结果；清空动作历史、已访问商店和奖励记忆。
每步与 SFT provenance 保存 segment_id，不跨片段拼接历史；回滚前的原始动作不删除。
`run_checkpoint` 是保存存档的生命周期通知，没有动作事务 ID；bridge 将其记为
`run_checkpoint_marker`，不创建动作样本、不结算待处理动作，也不清空历史。真正的读档
边界以 `run_resumed` 为准；checkpoint 原文始终保留在 Recorder 日志和提取包中。
训练筛选另行处理：需要确认的选牌尝试暂存在内存，返回/取消时将该尝试和返回动作
标为 `usable_for_sft=false`，保留真实history；重选并确认的步骤仍可训练。原始mod日志不筛选。
动作后状态除ready外，还检查钥匙持有、药水/卡牌获得、删除/升级完成等可验证效果；
快速下一步前效果未完成时，不标为稳定。
文件也含 `type=run_finished/run_ended` 生命周期行，这些行不是动作样本。

ActionRecorder 0.1.1为出牌/结束回合增加`action_execution_begin/result`和
`action_effects_settled`，以同一事务ID保留提交与执行两个时刻。`action_accepted`仅表示
已提交到游戏队列；`execution_status=executed`才证明实际执行，skipped/cancelled保留审计
但不作为成功动作训练。两条通道按明确状态ID配对，不按毫秒时间或状态差分倒推动作。

原`observation_before/pre_state/action`仍对应人类提交；执行前原始状态另存于
`execution_observation_before/execution_raw_message_before`。单步SFT应使用
`training_pre_state/training_policy_observation/training_legal_actions/training_action`，
其`training_observation_source`为execution（排队动作）或submission（其他动作）。
执行时按卡牌UUID重新映射位置与实际目标，只导出`usable_for_sft=true`；未取得正确
执行前状态的排队动作不可训练。旧日志中的动画中战斗提交也标为不可训练，不能自动补造
执行前观察。HAND_SELECT/GRID等真正的交互选择不因action_phase=EXECUTING_ACTIONS而排除。

`stable_after_action`只接受源的可决策后状态或游戏明确的逻辑结算边界；下一次点击时
仍在执行动画的观察只作为post审计，不标稳定。输出按提交顺序保存，即使后提交的队列项
先收到取消结果，也不会颠倒动作/history。

下文 `action-trace-*` 格式是controller控制模式，不是人类采集的原始格式。
旧教师SFT导出器仍仅接受teacher_auto，人类数据不能直接用该命令导出。

协议支持的关键命令包括 `START`、`PLAY`、`END`、`POTION`、`CHOOSE`、`PROCEED`、`RETURN` 和 `STATE`。详细定义以 [CommunicationMod README](https://github.com/ForgottenArbiter/CommunicationMod) 为准。

## START 命令

控制器把终端输入转换为 CommunicationMod 命令：

```text
s                         -> START IRONCLAD 0
s ABC123                  -> START IRONCLAD 0 ABC123
s SILENT 5 ABC123        -> START THE_SILENT 5 ABC123
s DEFECT 10              -> START DEFECT 10
s WATCHER 20 ABC123      -> START WATCHER 20 ABC123
```

角色别名：`IRONCLAD`、`SILENT`/`THE_SILENT`、`DEFECT`、`WATCHER`。进阶范围为 0-20；CommunicationMod 的 `START` 种子必须是字母数字。游戏运行状态中的 `game_state.seed` 可能带前导 `-`，但该值不能直接作为 `START` 参数。旧版 `s [seed]` 兼容为 Ironclad A0。

## 轨迹记录

每局有独立的 `run-events` 和 `action-trace` 文件，文件名包含 session、局序号、角色、进阶和种子，例如：

```text
action-trace-20260823-120000-run-002-THE_SILENT-A5-seed-4QTKQ7VBMFX47.jsonl
run-events-20260823-120000-run-002-THE_SILENT-A5-seed-4QTKQ7VBMFX47.jsonl
```

`run-events-*.jsonl` 记录单局生命周期，例如：

```json
{"event":"run_start","run_id":"...","run_config":{"character":"THE_SILENT","ascension":5,"seed":"ABC123","seed_source":"requested","internal_seed":null},"command":"START THE_SILENT 5 ABC123"}
```

未指定种子时，控制器先写 `seed_source: "random"`，收到第一条游戏运行状态后，从 CommunicationMod 的 `game_state.seed` 记录内部数值，并按 StS1 的 `0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ` 编码（省略 `O`）还原显示种子。`run_seed_observed` 事件同时包含 `display_seed` 和 `internal_seed`，后续轨迹的 `run_config.seed` 使用显示种子，`run_config.internal_seed` 保留协议原始值。种子字段用于复现实验；它们不会进入模型的规范化 observation。

每个动作的单局 `action-trace-*.jsonl` 至少包含：

```json
{
  "schema_version": "0.1",
  "session_id": "...",
  "run_id": "...",
  "run_config": {"character":"IRONCLAD","ascension":0,"seed":"ABC123"},
  "step_id": 12,
  "pre_state": {},
  "legal_actions": [],
  "action": {},
  "post_state": {},
  "accepted": true,
  "stable_after_action": true,
  "terminal": false
}
```

轨迹的 `decision` 字段保存 `model_action`、top-k `model_candidates`、`candidate_source`、`executed_action`、`source`（模型接受/人类覆盖/fallback）和 `overridden`，用于区分模型建议与最终执行标签。Qwen 只生成当前 `legal_actions` 中的 alias，controller 会再次过滤；置信度路由还会保存受约束的全动作打分及自动/人工路由结果。完整字段和保留策略见[数据与轨迹契约](DATA_AND_TRACE_SCHEMA.md)。

`pre_state`/`post_state` 是规范化状态，去除抽牌顺序、UUID 和其他不应作为玩家观察输入的字段。`messages-*.jsonl` 仍按 controller session 保存，只用于调试和重新清洗。随机种子未指定时，轨迹文件先使用 `seed-random`，收到内部 `game_state.seed` 后还原出显示种子并自动改名。

`START` 的种子参数仍必须是字母数字。`game_state.seed` 的负号只是 Java 有符号 64 位内部表示；控制器不会把这个数值直接发送给 `START`，而是把它保存为 `internal_seed` 并解码成可输入的显示种子。

规范化状态中的两个重要字段：

- `map.nodes`：当前 Act 已公开的完整地图节点及其 `x/y/symbol/children/parents` 关系；`map.current_node` 和 `map.next_nodes` 仍保留作快捷字段。
- `combat.monsters[*].intent_damage`、`intent_hits`：映射游戏当前显示的敌人意图伤害和段数；不再从基础招式值或离线表重算。若当前游戏未提供准确数值，规范化状态保留 `null`。当伤害可用时附带 `intent_total_damage`，按当前单段值和段数相乘，不重复应用玩家易伤等效果。
- `act_boss`：当前地图公开的本幕首领名称。不会再注入 Boss 静态攻略画像。
- `keys`：三色心脏钥匙状态，结构为 `ruby`（红）、`emerald`（绿）、`sapphire`（蓝）。当前标准 CommunicationMod 本身不序列化这些标志；启用带状态增强补丁的 ActionRecorder 后，TCP 状态和本地轨迹会同时提供该字段。字段缺失时表示未知，不应当作全部未持有。
- `decision_constraints`：当前遗物带来的长期规则，例如苏祖禁止获得药水、灵体外质禁止获得金币、添水的本回合出牌余量，以及营火/宝箱限制。能由状态确定为无效的药水或金币奖励会直接从 `legal_actions` 移除；其余限制仅作为显式上下文保留。持有符文圆顶时，敌人当前意图字段也会从模型观察和终端视图隐藏。
- `combat.player.powers`、`combat.monsters[*].powers`：保留游戏当前提供的 `id/name/amount/type`，并在运行时描述可取得时附带 `description`。
- 卡牌、药水、遗物及 Power 的说明由 ActionRecorder 在游戏线程读取当前游戏对象；动态卡牌变量使用游戏当前数值展开，无法展开的变量会列入 `unresolved_description_variables`。信息源标为 `game_runtime`，没有值时保留 `null`。战斗手牌的 `final_single_hit_damage` 和 `final_block` 使用游戏缓存的实时牌面值，控制器不再重复叠加 Power；多段牌的单段值不等于总伤害。
- `room_history`：模型观察保留当前房间的详细操作、对象/目标、选择目的、来源和公开变化；所有房间统一显示最近 30 步并标明省略数，进入二级界面不清空。事件初始说明和当前棋盘的已公开位置/牌面独立保留，较早翻牌动作行被省略后仍能看到已知牌面；未揭示牌面保持未知。动作后观测不保证动画结算完毕。`recent_actions` 仅是 UI 的六行摘要，不再限制模型上下文。详见[模型输入与动作空间](MODEL_OBSERVATION_AND_ACTIONS.md)。
- `screen_type=GRID`：若当前是事件/营火等触发的选牌界面，`screen_state.selection_mode` 和 `selection_instruction` 会明确表示 `PURGE`（移除）、`TRANSFORM`（变化）或 `UPGRADE`（升级）等目的。`for_purge` 等原始布尔字段仍保留作兼容。
- `screen_type=TRANSITION`：CommunicationMod 返回地图界面但 `room_phase` 仍不是 `COMPLETE` 时的保护态。该状态不包含可训练的地图选择动作，只允许等待刷新或返回当前房间。
- `screen_name=SETTINGS` 且 `is_screen_up=true`：表示设置/暂停覆盖层挡住了当前房间。此时 CommunicationMod 通常只暴露 `KEY`、`WAIT`、`STATE` 等通用命令，controller 将其显示为 `KEY:CANCEL`，终端输入 `x` 会发送 `KEY Cancel`；关闭覆盖层后才恢复战斗动作。
- 药水栏已满或持有苏祖时，controller 不会把 `COMBAT_REWARD` 中的 `POTION` 或商店中已识别的药水商品列为合法动作。持有灵体外质时，`COMBAT_REWARD` 中的金币也会被移除。CommunicationMod 对这类操作可能不返回错误或状态变化；终端会说明对应原因。
- `COMBAT_REWARD` 在请求模型前按“金币、药水、卡牌奖励入口”的顺序自动推进：金币在可获得时自动拾取；遗物不自动拾取，因为可能与钥匙互斥；药水仅在有空位时必取；卡牌奖励只自动打开，选牌或跳过仍交给模型。除地图和地图过渡态外，只要 CommunicationMod 当前声明支持 `potion` 命令，合法动作中还会保留可使用或可丢弃的持有药水；战斗中的目标药水展开为敌人目标。非战斗场景只允许主动使用 `BLOODPOTION`（鲜血药水）和 `FRUITJUICE`（果汁），其他药水即使通信层误报 `can_use=true` 也只保留丢弃动作。这样在领取新药水前，可以先由模型决定使用或丢弃旧药水。首次进入 `SHOP_ROOM` 也会自动打开商人界面，购买和移除仍是模型决策。环境动作完成后，如果语义合法动作列表只剩一个动作（例如离开商人后的 `PROCEED`，或战斗中没有可出牌、可用药水时的 `END_TURN`），controller 直接执行该动作，不请求模型；自动执行前会短暂停顿 `0.25` 秒，让界面先显示动作。每个这类动作都以 `decision.source=environment_auto` 正常写入 trace 和 room_history，但不会进入教师 SFT 或在线 RL 的策略更新；`BOSS_REWARD` 的遗物选择始终保留为策略决策。
- 启用 `posthoc_feedback.enabled` 且远端 `attribution.enabled` 后，在线 RL 的每个终局会额外产生本地 `data/reflections/` 与 `data/attributions/` artifact。`posthoc_feedback.mode="human"` 时前者保存人类的可强化/应弱化复盘；`mode="auto"` 时保存自动复盘标记，教师根据账本独立给出双向候选。后者保存本地校验过的 `step_id/direction/confidence/severity`；只有后者的已应用字段进入远端 episode。教师原始 completion 与思考不写入 artifact 或更新协议。详情见[终局反思归因 GRPO](POSTHOC_ATTRIBUTION_GRPO.md)。

Candidate Assist 由 controller 在 CommunicationMod 启动握手前预热独立的 `sts_agent.policy.local_qwen`。模型、端口、超时和模式全部由 `config/agent.json` 管理；模型必须先通过 `sts_agent.commands.download_model` 从 `policy.model.hub_endpoint` 预下载。脚本会在没有显式代理变量时读取 Windows 系统代理。游戏运行时 policy server 只从本地缓存加载，以免网络问题阻塞回合。默认对所有游戏内场景请求候选；可将 `policy.request.screens` 配置为 `["COMBAT"]` 收窄。`MAP` 才发送完整公开 `map.nodes`、当前位置和可选节点；战斗、事件、商店、奖励、休息和地图过渡态都移除地图。

controller 维护环境记忆 `run_memory.visited_shop_rooms`。首次观察到 `SHOP_SCREEN` 时按 `(Act, Floor)` 记录该商店；已访问不会屏蔽同层商人入口，也不会拒绝人工重入。星系仪等遗物的奖励流程可能自动关闭商人界面，模型或玩家仍可根据需求重新打开。自动打开商人只在首次访问时触发；之后由模型或玩家选择重入或离开，避免自动进出循环。

合法动作映射为本请求内短别名 `A1/A2/...`，policy server 返回至多三个别名，controller 再映射、校验并编译 CommunicationMod 命令。这避免小模型复制长的卡牌/目标 ID 时产生空候选。终端 `a` 接受首选，`pick <序号>` 选择其他候选；裸数字不被候选逻辑截获，保留给游戏界面自己的选项。模型无候选、超时或加载失败时，`decision` 会记录 `assist_status` 和 `assist_reason`，终端显示原因后退回人工操作。将 `policy.enabled` 设为 `false` 可关闭，`policy.development.mock=true` 可在不加载模型时测试协议。

设置 `policy.confidence_auto.enabled=true` 后，controller 不再请求生成式 top-k 排序，而请求 policy server 为每个当前合法别名计算精确续写 `A<n>\\n` 的条件 `logprob`。controller 仅在这一个合法集合内 softmax，完整 `action_scores` 按规范 action ID 写入 trace；`confidence` 保存首选/次选概率、margin、entropy、`min_probability` 和路由结果。默认 `min_probability=0.85`；首选满足 `p1 >= min_probability` 即执行 `source=model_auto`。margin 与 entropy 用于校准记录，不作为第二道门槛。评分缺项、格式异常、超时或上一动作返回协议错误时，自动执行被禁用并回退人工，不能绕过 controller 的合法动作表。

卡牌、药水、遗物和 Power 的模型输入只取当前游戏状态提供的字段，不再加载本地实体快照或使用静态 fallback。缺失信息保持未知，不根据名称补造效果。商店商品保留额外的 `price`。事件选项与奖励只使用 CommunicationMod 已经公开的实体 ID/名称；裸 `reward_type` 不会泄漏尚未公开的随机奖励。战斗卡牌奖励被跳过后，controller 以 `(Act, Floor)` 记录到 `run_memory.declined_card_rewards` 并从父 `COMBAT_REWARD` 动作表移除同一入口，防止导航造成重开循环。

## 数据使用规则

- 只把动作已接受、状态稳定的 step 放入正式纠正式数据集；`executed_action` 是监督标签，`model_action` 用于统计错误和 DAgger 重采样。
- 中途退出、错误响应、状态不稳定的 run 保留在 raw 目录，标记为调试数据。
- train/validation/test 按 `run_id` 或 `seed` 切分，不能按动作随机切分。
- 不把 raw JSON 直接喂给模型；先经过版本化 normalizer 和 action compiler。
