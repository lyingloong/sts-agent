# 通信协议与数据格式

## CommunicationMod 配置

CommunicationMod 启动外部进程，进程 stdout 只能用于协议。控制器启动时发送 `ready`，随后接收 JSON 状态并输出协议命令；诊断写入 stderr 和项目日志。

```properties
command=D\:/miniconda3/python.exe D\:/Code/spire-mod/sts-agent/tools/communication_controller.py
runAtGameStart=true
```

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

`pre_state`/`post_state` 是规范化状态，去除抽牌顺序、UUID 和其他不应作为玩家观察输入的字段。`messages-*.jsonl` 仍按 controller session 保存，只用于调试和重新清洗。随机种子未指定时，轨迹文件先使用 `seed-random`，收到内部 `game_state.seed` 后还原出显示种子并自动改名。

`START` 的种子参数仍必须是字母数字。`game_state.seed` 的负号只是 Java 有符号 64 位内部表示；控制器不会把这个数值直接发送给 `START`，而是把它保存为 `internal_seed` 并解码成可输入的显示种子。

规范化状态中的两个重要字段：

- `map.nodes`：当前 Act 已公开的完整地图节点及其 `x/y/symbol/children/parents` 关系；`map.current_node` 和 `map.next_nodes` 仍保留作快捷字段。
- `combat.monsters[*].intent_damage`、`intent_hits`：从 CommunicationMod 的当前意图字段映射而来，用于表达敌人当前显示的攻击伤害和段数。
- `combat.hand[*]`：保留 `damage`、`block`、`magic_number`、`description` 等字段（只有协议实际提供时才出现）。动态值不根据卡牌 ID 临时猜测；静态卡牌知识统一放在下方的 `effect` 字段。
- `screen_type=TRANSITION`：CommunicationMod 返回地图界面但 `room_phase` 仍不是 `COMPLETE` 时的保护态。该状态不包含可训练的地图选择动作，只允许等待刷新或返回当前房间。

当前已增加本地静态知识库 fallback：`combat.hand[*].effect` 包含卡牌的 `type/rarity/color/target/cost`、基础 `damage/block/magic_number/hit_count`、`exhaust/ethereal/innate/retain`、`tags/keywords`、基础/升级描述和完整 `upgrade` 字段，`effect.source` 标记为 `static_knowledge`。若协议提供运行时字段，则这些字段覆盖静态值，并标记为 `runtime_over_static`。静态知识库由 `tools/fetch_card_knowledge.py` 从 [spire-archive StS1 cards 数据](https://github.com/nkhoit/spire-archive/blob/main/data/sts1/cards.json)生成，数据文件不提交 Git。

## 数据使用规则

- 只把完整结束且动作全部接受的 run 放入正式 SFT 集。
- 中途退出、错误响应、状态不稳定的 run 保留在 raw 目录，标记为调试数据。
- train/validation/test 按 `run_id` 或 `seed` 切分，不能按动作随机切分。
- 不把 raw JSON 直接喂给模型；先经过版本化 normalizer 和 action compiler。
