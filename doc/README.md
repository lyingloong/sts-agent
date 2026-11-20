# StS Agent 文档

[项目中文首页](../README-zh.md)包含安装步骤和首次人工录制流程。下表按你要完成的工作列出详细指南。

| 用途 | 文档 |
| --- | --- |
| 选择或切换模式 | [模式清单](MODES.md) · [配置文件、覆盖顺序与示例](../config/README.md) · [命令索引](COMMANDS.md) |
| 接入游戏或排查录制问题 | [通信协议](PROTOCOL.md) · [轨迹字段与结算状态](DATA_AND_TRACE_SCHEMA.md) · [输出目录](../data/README.md) |
| 接入模型并评测完整对局 | [基线推理与后端服务](BASELINE_INFERENCE.md) · [固定种子、运行预算与计分](BASELINE_EVALUATION.md) |
| 通过 MCP 玩一局游戏 | [Codex A0 铁甲战士指南](CODEX_PLAY_A0_IRONCLAD.md) |
| 从轨迹导出样本并训练 | [人类 SFT 数据集](HUMAN_SFT_DATASET.md) · [教师 SFT 与 GRPO 设计记录](TEACHER_SFT_GRPO.md) · [训练机与数据迁移](TRAINING_ON_4090.md) |
| 在线生成轨迹和更新策略 | [远端推理与在线 RL](REMOTE_ONLINE_RL.md) · [终局反馈与动作归因](POSTHOC_ATTRIBUTION_GRPO.md) |
| 修改状态、动作或提示词 | [模型可见状态与合法动作](MODEL_OBSERVATION_AND_ACTIONS.md) · [语义动作接口](SEMANTIC_ACTIONS.md) · [策略提示词](POLICY_PROMPT.md) · [实际渲染示例](POLICY_PROMPT_PREVIEW.md) |
| 使用教师模型或迭代提示词 | [教师提示词](TEACHER_PROMPTS.md) · [Prompt Evolution 流程](PROMPT_EVOLUTION.md) |
| 参与项目开发 | [架构与依赖边界](ARCHITECTURE.md) · [迁移说明](MIGRATION.md) · [贡献指南](../CONTRIBUTING.md) |

## 设计与实验记录

[提示词评审](POLICY_PROMPT_AUDIT.md)及[待实现的 Markdown 展示方案](POLICY_PROMPT_PREVIEW_MARKDOWN.md)记录了提示词设计的取舍。
此前的[SFT 基线计划](SFT_BASELINE_NEXT_PLAN.md)、[论文计划](PAPER_PLAN.md)和[中断的 30 种子批次](EXPERIMENT_HANDOFF.md)保留供查阅历史结果。
