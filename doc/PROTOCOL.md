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

角色别名：`IRONCLAD`、`SILENT`/`THE_SILENT`、`DEFECT`、`WATCHER`。进阶范围为 0-20；种子必须是字母数字。旧版 `s [seed]` 兼容为 Ironclad A0。

## 轨迹记录

`run-events-*.jsonl` 记录局生命周期，例如：

```json
{"event":"run_start","run_id":"...","run_config":{"character":"THE_SILENT","ascension":5,"seed":"ABC123","seed_source":"requested"},"command":"START THE_SILENT 5 ABC123"}
```

未指定种子时，控制器先写 `seed_source: "random"`，收到第一条游戏运行状态后，从 CommunicationMod 的 `game_state.seed` 写入 `run_seed_observed` 事件，并更新后续轨迹的 `run_config.seed`。这个字段用于复现实验；它不会进入模型的规范化 observation。

每个动作的 `action-trace-*.jsonl` 至少包含：

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

`pre_state`/`post_state` 是规范化状态，去除抽牌顺序、UUID 和其他不应作为玩家观察输入的字段。`messages-*.jsonl` 保留原始协议消息，只用于调试和重新清洗。

## 数据使用规则

- 只把完整结束且动作全部接受的 run 放入正式 SFT 集。
- 中途退出、错误响应、状态不稳定的 run 保留在 raw 目录，标记为调试数据。
- train/validation/test 按 `run_id` 或 `seed` 切分，不能按动作随机切分。
- 不把 raw JSON 直接喂给模型；先经过版本化 normalizer 和 action compiler。
