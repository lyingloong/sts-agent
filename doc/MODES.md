# 运行模式

全新检出使用 `collect_human`。要更改设置，将[公开模板](../config/agent.example.json)复制为
`config/agent.json`。在 `active_mode` 里填写下表中的模式名，跨模式共用的连接参数放在 `shared`。
需要为某个模式单独换模型、API 地址等时，写在 `mode_overrides.<模式名>` 中。合并顺序与示例见[配置说明](../config/README.md)。

| 模式 | 作用 |
| --- | --- |
| `collect_human` | 人类直接在游戏内游玩，记录完整状态-动作轨迹。当前推荐的采集方式。 |
| `baseline_inference` | 使用自建模型服务自动游玩并评测完整对局。 |
| `baseline_api` | 通过托管的 OpenAI 兼容 API 游玩，可使用相同的固定种子套件。 |
| `candidate_assist_local` / `candidate_assist_remote` | Qwen 提供动作候选，由玩家确认或修改；按置信度配置可自动执行首选。 |
| `teacher_sft` | 远端教师模型游玩并生成 SFT 训练数据。 |
| `dagger_local` / `dagger_remote` | 模型候选由人类确认或纠正，形成 DAgger 样本。 |
| `prompt_evolution` | 教师在完整轨迹证据上提议角色/场景 skill 更新，由人类审核后验证。 |
| `local_inference` / `remote_inference` | 分别在本机或 SSH 远端运行固定 Qwen 策略。 |
| `online_rl` | 远端 Qwen 采样完整 rollout，并按组结果更新 adapter。 |
| `collect_codex` | 通过 Codex MCP 操作游戏，单独保存 Codex 轨迹。 |

运行 `sts-agent-config` 检查当前配置，或用 `sts-agent-config --mode online_rl` 在切换前单独检查某个模式。
改动生效需要重启游戏启动的 controller 和正在使用的 UI。
