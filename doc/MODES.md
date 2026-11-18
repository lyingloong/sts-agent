# 运行模式

日常通过 `config/agent.json` 的单一字段 `active_mode` 切换运行方式。具体模式参数在 `config/modes/<mode>.json`；Qwen adapter、远端连接、教师 API、游戏连接和 Overlay 等共享配置只需维护一次。所有 profile 名称和它们对应的内部控制/后端/学习组合列在 [配置说明](../config/README.md)。

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

模式组合集中在 `sts_agent.core.modes` 固定定义，配置文件不再要求用户同时编辑 `control_mode`、`policy_backend` 和 `learning_method`。`load_agent_config()` 将活动 profile 与共享设置合成为内部运行配置；改动后重启 controller 和 UI。
