# 运行模式与学习方法

运行配置把“谁控制游戏”“模型从哪里来”“是否更新”拆成三个字段，避免把一组实验配置编码成一个不断膨胀的模式名。

## 控制方式

| `control_mode` | 行为 |
| --- | --- |
| `manual` | 不请求策略，人类选择合法动作。 |
| `candidate_assist` | 策略提供候选，人类确认；置信度满足配置时可自动执行。 |
| `autonomous` | 直接执行策略或教师选出的动作。 |

## 策略后端

| `policy_backend` | 适用场景 |
| --- | --- |
| `none` | 纯人工或只采集环境轨迹。 |
| `local_qwen` | 本机 Qwen 推理/Candidate Assist。 |
| `remote_qwen` | 通过 SSH 启动远端固定推理或在线 RL worker。 |
| `remote_teacher` | DeepSeek 教师决策、SFT 采集或 Prompt Evolution。 |
| `mock` | 协议和 UI 测试，不代表模型效果。 |

## 学习方法

| `learning_method` | 数据/更新行为 |
| --- | --- |
| `none` | 只运行策略，不写训练更新。 |
| `collect_human` | 接收 ActionRecorder 的原始状态和语义动作，并整理为包含 `pre_state`、`policy_observation`、`legal_actions`、`action` 和 `post_state` 的 sts-agent 人类轨迹。 |
| `teacher_sft` | 用教师动作导出 response-only SFT 数据。 |
| `dagger` | 模型提出动作，人类纠正后保存状态-专家动作。 |
| `prompt_evolution` | 在固定角色/种子上由教师生成候选 skill，人工审核后更新 Prompt。 |
| `online_rl` | 收集完整 rollout，按组奖励和可选事后归因更新 adapter。 |

## 推荐组合

| 目标 | 控制 | 后端 | 学习 |
| --- | --- | --- | --- |
| 人工游玩 | manual | none | none |
| 人类轨迹采集 | manual | none | collect_human |
| Candidate Assist | candidate_assist | local_qwen | none |
| 教师冷启动 | autonomous | remote_teacher | teacher_sft |
| 在线纠正式模仿 | candidate_assist | local_qwen/remote_qwen | dagger |
| Prompt 演化 | autonomous | remote_teacher | prompt_evolution |
| 远端固定推理 | autonomous | remote_qwen | none |
| 远端在线 RL | autonomous | remote_qwen | online_rl |

无效组合由 `sts_agent.config` 拒绝。仓库只使用 `config/agent.json`，每次切换配置后重启 controller 和 UI。
