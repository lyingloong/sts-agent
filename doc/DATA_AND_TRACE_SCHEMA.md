# 数据与轨迹

## 运行时数据生命周期

```text
CommunicationMod message
  -> normalized observation
  -> routed policy observation
  -> legal actions
  -> DecisionRecord
  -> action result / next state
  -> trace JSONL
```

## 核心契约

### Observation

至少包含 `screen_type`，其余字段由 StS1 适配器提供，例如角色、楼层、牌组、战斗、界面状态、历史和决策限制。若状态源可提供三色心脏钥匙，则包含 `keys: {ruby, emerald, sapphire}`；字段缺失表示未知，不应解释为全部未持有。地图全貌只有 `screen_type=MAP` 时进入策略 observation。

### Action

每个合法动作必须有唯一短 ID。常见 ID 为 `A1`、`PLAY:card=2:target=0`、`END_TURN`。展示文本放在 `label`，CommunicationMod 命令放在 `command`，模型只输出 ID。

### DecisionRecord

```json
{
  "step_id": 12,
  "observation": {},
  "legal_actions": [{"id": "A1", "label": "结束回合"}],
  "selected_action_id": "A1",
  "source": "human",
  "candidates": [],
  "outcome": {}
}
```

SFT、DAgger 和 RL 可以使用同一份记录；差异只在于 `source`、反馈和更新方式。

## 目录约定

- `data/knowledge/`：StS1 静态知识和经验，Git 跟踪。
- `data/trajectories/`：原始游戏轨迹，Git 忽略。
- `data/datasets/`：SFT/DAgger 导出数据，Git 忽略。
- `data/transfer/`：传输 bundle，Git 忽略。
- `data/prompt_evolution/`、`data/runtime/`：运行状态和审核 artifact，Git 忽略。
- `models/`、`outputs/`、`checkpoints/`：模型和 adapter，Git 忽略。

API key、远端机器专属路径和 `online_rl.json` 不得提交。
