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

### 人类采集轨迹的动作后状态

`post_state` / `observation_after` 保留用于复盘和调试，不能默认视为该动作完整结算后的
结果。动画、延迟效果和连续操作可能使后状态仍保留旧值，或包含其他动作的影响；例如
领取绿钥匙后立刻点击下一奖励，后状态可能仍为 `emerald=false`，稍后才更新为 true。

`resolution=next_decision` 表示用下一次决策前快照关闭关联，不证明效果已完成。
`stable_after_action=false` 明确表示未确认完整结算；即使该字段为 true，也应结合动作类型
和执行/结算标记检查，尤其不要把前后差异直接解释成某个动作的独立奖励。

当前人类 SFT 只使用确切动作前状态、合法动作及实际执行动作，不依赖后状态或该稳定标记。
后状态缺失或未结算不单独排除样本。用于状态转移、效果分析或奖励计算时，应额外校验。
执行前状态与实际执行结果仍需保留，以排除排队后取消、跳过或未执行的操作。

## 目录约定

- `data/trajectories/`：原始游戏轨迹，Git 忽略。
- `data/datasets/`：SFT/DAgger 导出数据，Git 忽略。
- `data/transfer/`：传输 bundle，Git 忽略。
- `data/prompt_evolution/`、`data/runtime/`：运行状态和审核 artifact，Git 忽略。
- `models/`、`outputs/`、`checkpoints/`：模型和 adapter，Git 忽略。

API key、远端机器专属路径和 `online_rl.json` 不得提交。
