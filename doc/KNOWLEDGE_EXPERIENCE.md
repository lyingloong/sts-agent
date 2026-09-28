# 实体经验知识库

这是一套面向 StS1 的轻量知识维护闭环。它不引入向量检索或独立记忆库，而是在现有静态快照的实体上增加可选字段：

```json
{
  "id": "CORRUPTION",
  "name": "Corruption",
  "description": "...",
  "experience": [
    "【战斗】在能稳定防御的回合打出，之后要利用零费技能尽快完成关键回合。"
  ]
}
```

支持的实体类型为 `card`、`relic`、`potion`、`monster`、`event` 和 `power`。`experience` 缺失与空数组等价；只有审核通过的内容才会写回 `data/knowledge/sts1_*.json`。每个实体最多三条，写回时是完整替换而不是无界追加，因此经验不会无限增长。`data/knowledge/experience_audit.jsonl` 保存审核通过的来源、旧列表和新列表，便于回溯。

## 运行流程

知识问答是教师动作流程中的可选能力，不需要切换成另一种决策模式。推荐在当前使用的教师配置中加入：

```json
"knowledge_experience": {
  "enabled": true,
  "request_style": "encouraged",
  "max_entries_per_entity": 3,
  "max_question_chars": 2000,
  "max_answer_chars": 6000
}
```

在使用教师的配置中开启该能力；同时将 `framework.policy_backend` 设为 `remote_teacher`，并把 `framework.learning_method` 设为 `teacher_sft` 或 `prompt_evolution`：

```powershell
python -m sts_agent.runtime.controller
python -m sts_agent.ui.overlay
```

1. 教师正常接收状态和合法动作。当它对某个当前公开实体有不明白之处时，可以返回一个 `knowledge_request`，而不是执行动作。controller 暂停当前状态，通过 Overlay 或文本客户端收集人类回答。
2. DeepSeek 根据实体静态资料、已有最多三条经验、当前场景和人类回答，输出新的完整 `experience` 列表。整理阶段复用现有教师客户端的超时、重试和限流退避。
3. 人类审核候选列表。接受后 controller 原子更新对应 JSON，并立即清空运行时知识缓存；拒绝或放弃不会改动知识库。随后在同一个游戏状态重新请求动作。

`request_style` 提供两种动作提示版本：`encouraged`（默认）规定模型只要对当前公开实体的机制、交互、风险或策略含义有不明白之处，就鼓励先提问、不要猜测；`normal` 保留相同的“有疑问即可问”协议，但措辞更克制，优先使用已有状态与静态资料。两种版本都支持卡牌、遗物、药水、怪物、事件和 Power。

## 给动作模型的注入

经验只跟随实体直接注入：卡牌、药水、遗物在现有规范字段旁增加 `experience`；战斗怪物的静态 `knowledge` 以及全局 Boss profile 也可带该字段；活动 Power 的摘要同样支持经验。不存在实体检索和跨场景拼接，避免把一个场景的建议误用于另一个实体。当前实时字段始终优先于静态值。

`knowledge_request` 是严格受控的带外协议：请求必须包含当前 observation 已公开的实体类型、ID/名称和具体问题，controller 会再次从本地快照校验实体，不允许模型凭空写入新实体或直接修改文件。请求本身不产生游戏动作，也不会创建新的决策模式。

## 数据与 Git

静态知识库快照由 `sts_agent.commands.fetch_knowledge` 维护并纳入 Git。经验字段是本项目的人工维护内容，刷新上游快照时必须审查 diff，生成器应保留已有经验。审计日志是否提交由实验需要决定；它不参与模型输入，只用于追踪谁在何时接受了哪次替换。
