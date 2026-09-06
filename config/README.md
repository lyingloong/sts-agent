# 运行配置

所有 JSON 文件使用严格 schema：未知字段会直接报错，避免拼写错误静默改变实验。JSON 不支持注释，本文件是每个字段的说明与运行边界。

## 配置归属

配置加载器以 `tools/agent_config.py` 的默认值为基线，对配置做严格稀疏合并。推荐直接编辑 `config/agent.json` 的 `active_mode`；各模式的覆盖写在同一个文件的 `mode_profiles` 下，因此 CommunicationMod 固定启动 controller 时不需要额外参数。`config/modes/*.json` 仍保留为兼容和独立实验入口。

例如：

```powershell
python tools\communication_controller.py
python tools\overlay_terminal.py
```

单文件模式切换只需修改：

```json
"active_mode": "prompt_evolution"
```

可选值为 `manual`、`candidate_assist`、`teacher_sft`、`prompt_evolution`、`remote_inference`、`remote_online_rl`。切换后必须重启 controller；Overlay 也应重启以读取同一份配置。

`policy.model.base`、`policy.model.initial_adapter` 和 `policy.model.initial_policy_version` 在所选模式配置中只设置一次。本机推理、远端固定推理、远端在线 RL 不会再从多个配置段重复读取同一 adapter。

远端被忽略的 `online_rl.json` 只描述一次在线训练如何执行，不能包含 `model`、`initial_adapter` 或 `initial_policy_version`。controller 通过 SSH 启动 worker 时显式传入这三个值，因此远端文件不能覆盖本机选择的初始策略。

如果需要换一套初始权重，请同时修改 `initial_adapter` 和 `initial_policy_version`，并为该实验使用新的 `online_rl.output_root`。已有 `output_root/state.json` 会优先恢复其记录的当前版本，修改初始 adapter 不会替换已开始实验的策略。

## `agent.json`

| 字段 | 含义 | 适用范围 |
| --- | --- | --- |
| `schema_version` | 当前配置格式，固定为 `1`。 | 全部 |
| `server.host` / `server.port` | CommunicationMod controller 的本机监听地址与端口。 | 全部 |
| `policy.enabled` | 是否启动 Qwen 策略。教师采集时必须为 `false`。 | 全部 |
| `policy.runtime` | `local` 本机推理；`remote_inference` 远端只推理；`remote_online_rl` 远端采样并更新。 | 全部 |
| `policy.model.base` | Qwen 基座的本机路径、远端绝对路径或 Hugging Face ID；路径必须对当前 runtime 实际可见。 | 全部 |
| `policy.model.initial_adapter` | 唯一的初始 LoRA adapter。可为空以运行未微调的固定推理；真实在线 RL 必须是完整 adapter 目录。 | 全部，在线 RL 必填 |
| `policy.model.initial_policy_version` | 新建在线 RL `output_root` 的初始版本标签，例如 `sft-v1`。 | 在线 RL |
| `policy.model.hub_endpoint` | 下载基座模型时使用的 Hugging Face 端点；游戏运行不会联网下载。 | `local`、`remote_inference` |
| `policy.request.top_k` | Overlay 显示的候选数；在线 RL 同时限制展示的策略候选数。 | 全部 |
| `policy.request.response_timeout_seconds` | 每个动作请求最多等待秒数。 | 全部 |
| `policy.request.startup_timeout_seconds` | controller 预热本机/远端 worker 的最多等待秒数。 | 全部 |
| `policy.request.screens` | 请求策略的场景，`["ALL"]` 或 `["COMBAT"]`。 | 全部 |
| `policy.local.python` | `local` 模式启动 policy server 的解释器；`auto` 优先项目 `.venv`。 | `local` |
| `policy.remote.ssh_host` | Windows SSH config 中的主机别名。 | 两个远端模式 |
| `policy.remote.project_root` | 远端仓库绝对路径。 | 两个远端模式 |
| `policy.remote.python` | 远端项目内 Python 命令。 | 两个远端模式 |
| `policy.remote.online_rl_config` | 相对远端项目根目录的、被忽略的在线 RL 运行配置。 | `remote_online_rl` |
| `policy.remote.connect_timeout_seconds` | 建立 SSH 连接的最多等待秒数。 | 两个远端模式 |
| `policy.remote.update_timeout_seconds` | 终局上传与第 K 局参数更新的最多等待秒数。 | `remote_online_rl` |
| `policy.remote.inference_gpu.device` | 远端固定推理卡，物理序号或 `"auto"`。 | `remote_inference` |
| `policy.remote.inference_gpu.minimum_free_memory_mib` | `device: "auto"` 时所需的最低可用显存。 | `remote_inference` |
| `policy.development.mock` | 不加载真实模型，仅测试 controller/JSONL 协议。不能用于训练结果。 | 开发测试 |
| `policy.development.debug` | 将完整异常写到 controller/worker 的 stderr。 | 排错 |
| `policy.confidence_auto.enabled` | 固定推理时满足阈值则自动执行首选；在线 RL 忽略此项并始终采样。 | `local`、`remote_inference` |
| `policy.confidence_auto.min_probability` | 自动执行所需的首选动作归一化概率。 | `local`、`remote_inference` |

### 教师与归因

`teacher` 用于远程教师 SFT 采集和 Prompt 演化中的实际决策/策略分析，`posthoc_feedback` 用于在线 RL 终局归因。后者可由人类反思定向，也可由教师自动复盘；这些功能都使用 OpenAI 兼容 API。

| 字段 | 含义 |
| --- | --- |
| `<group>.enabled` | 是否启用该功能；`teacher` 与 `policy` 不能同时启用，`posthoc_feedback` 仅允许在线 RL，`knowledge_experience` 需要启用 `teacher`。 |
| `posthoc_feedback.mode` | `human`（默认，终局显示正/负人工反思框）或 `auto`（不请求人工输入，教师根据终局与完整动作账本自行做双向归因）。自动模式只适合探索，建议定期审计归因 artifact。 |
| `<group>.base_url` / `<group>.model` | API 根地址与模型 ID。 |
| `<group>.api_key_file` | 相对项目根目录的密钥文件，必须被 Git 忽略。 |
| `<group>.timeout_seconds` / `max_retries` | 单次 API 请求超时和失败后的最多重试次数；网络错误、超时、429 和 5xx 会重试，认证/参数类 4xx 立即失败。 |
| `<group>.retry_backoff_seconds` / `retry_max_backoff_seconds` | 重试前的指数退避起始秒数和上限；默认第 1 次重试等待 5 秒，之后按 5、10、20... 秒增长。按 RPM=20 的远程服务限制设置。设为 `0` 可关闭等待但仍保留重试。 |
| `<group>.temperature` / `max_tokens` | 教师/归因模型生成温度与最大输出 token。 |
| `teacher.max_input_chars` / `teacher.screens` | 教师单步上下文的最大字符数，以及请求教师的场景。 |
| `teacher.thinking` / `json_mode` / `streaming` | 是否请求思考、结构化 JSON 和流式展示。 |
| `posthoc_feedback.max_input_chars` | 单局归因上下文最大字符数；动态状态和全动作列表会优先保留。 |
| `posthoc_feedback.max_reflection_chars` | 玩家正/负反思文本的最大字符数。 |
| `posthoc_feedback.max_candidates_per_direction` | 教师每个方向可提出的候选步骤数。 |
| `posthoc_feedback.max_applied_per_direction` | 本地校验后每个方向实际送入更新的最多步骤数。 |
| `posthoc_feedback.min_confidence` | 接受教师归因步骤的最低置信度。 |

### 实体经验知识库

`knowledge_experience` 复用 `teacher` 的 API 配置，不是独立的游戏模式。教师在任意游戏决策中，若当前公开实体确实缺少依据，可以暂时返回知识请求；Overlay/终端收集人类回答后，DeepSeek 输出最多三条经验的完整替换列表，最后由人类审核。若审核拒绝，上一版候选和拒绝理由会交回 DeepSeek 重写并再次审核，直到接受或放弃；只有接受后才原子写入 `data/knowledge/sts1_*.json`，并立即回到同一状态重新请求动作。字段如下：

| 字段 | 含义 |
| --- | --- |
| `knowledge_experience.enabled` | 开启教师动作流程中的带外知识请求与审核写库能力；关闭时教师只能返回动作。 |
| `knowledge_experience.request_style` | `encouraged`（默认）：只要当前公开实体有不明白之处，就鼓励优先澄清而非猜测；`normal`：同样允许提问，但措辞更克制，优先使用已有状态与静态资料。 |
| `max_entries_per_entity` | 每个实体经验上限，当前限制为 3。 |
| `max_question_chars` / `max_answer_chars` | 单次模型问题与人类回答的最大字符数。 |

除游戏中的教师问答流程外，也可以在 Overlay 动作面板点击 `知识库`，逐个浏览所有静态实体并手动维护最多三条 `experience`。保存会直接更新对应的 `data/knowledge/sts1_*.json`，不创建额外经验补充文件。

### Prompt 演化

`prompt_evolution` 是固定角色、进阶和游戏种子的人工审核式策略迭代模式。它与 `policy`、`posthoc_feedback` 互斥，使用同一份 `teacher` USTC API：教师只提出一个场景或角色 skill 的小型替换，候选先经过一次“验证前审核”，人工批准后才在同一种子上重跑，验证结果再经过第二次人工审核。首次 incumbent 可以使用随机种子，观察到游戏返回的真实种子后才建立候选复跑锚点。只有接受验证结果才会写入当前角色目录下 Git 跟踪的 `prompts/characters/<角色>/versions/pNNNN.json` 并更新 manifest；候选、轨迹、拒绝理由和 session 状态写入被忽略的 `data/prompt_evolution/`。

| 字段 | 含义 |
| --- | --- |
| `prompt_evolution.enabled` | 启用固定种子 prompt 演化；需 `teacher.enabled=true`、`policy.enabled=false`、`posthoc_feedback.enabled=false`。 |
| `policy_dir` | 角色 Prompt 库的项目相对目录；运行时使用其下的 `<角色>/manifest.json` 和 `versions/`。 |
| `output_dir` | 候选、重跑轨迹和审核状态的本地实验目录。 |
| `max_skill_chars` | 单个场景补充策略的最大字符数；防止候选变成长篇新 prompt。 |
| `max_analysis_input_chars` / `max_analysis_tokens` | 发送给 USTC 演化教师的单个 Act 压缩动作账本字符上限和输出 token 上限；默认分别为 `900000` 和 `8192`。完整运行按 Act 分段逐次审核；自动推进、地图图结构、Boss 详细招式表、静态物品描述和未变化的卡组/遗物/药水不会重复发送。输入上限使用字符计数，避免把字符和模型 token 混淆；若仍超限，Prompt Evolution 面板会显示告警并弹窗。 |
| `max_revisions_per_cycle` | 同一候选被拒绝后最多自动重写次数。 |
| `auto_continue` | 接受验证结果后是否自动生成下一候选；即使开启，也会先停在下一次“验证前审核”，不会跳过人工批准。关闭则接受后暂停。 |

### Overlay

`overlay.always_on_top` 控制窗口置顶；`frameless` 控制是否移除系统标题栏。`status`、`control`、`online_rl` 的 `geometry` 使用 `宽x高+X+Y`，`opacity` 是 `0.2` 到 `1.0` 的透明度；`status.divider_columns` 是状态窗口分隔线宽度。

## `online_rl.json`

此文件仅存于训练机并被 Git 忽略。可以从 `online_rl.example.json` 复制。所有字段都在以下范围内：

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 当前在线 RL 配置格式，固定为 `1`。 |
| `output_root` | 本次实验的远端输出根目录，保存 group、版本 adapter 与 `state.json`。每个独立实验必须不同。 |
| `group_size` | 同一冻结策略的完整 rollout 数，当前推荐 `4`。 |
| `seed` | worker 自身的随机采样种子，不是游戏显示 seed。 |
| `sampling.temperature` | 动作 logprob 除以前的采样温度；更高更多样，更新时使用相同值重算 ratio。 |
| `reward.version` | 记录用的奖励定义版本。 |
| `reward.weights.victory` / `heart` / `act` / `floor` / `hp_fraction` | 终局奖励权重，分别对应胜利、心脏、层数、楼层和 HP 比例。 |
| `gpu.device` | 更新 worker 使用的远端物理 GPU 序号，或 `"auto"`。 |
| `gpu.minimum_free_memory_mib` | `auto` 选择 GPU 时所需最低可用显存。 |
| `attribution.enabled` | 是否只更新人类反思教师选中的步骤；必须和本机 `posthoc_feedback.enabled` 同时开关。 |
| `attribution.max_abs_advantage` | 归因更新中优势值的绝对值裁剪上限。 |
| `attribution.min_tied_advantage` | 组内终局奖励完全相同时，归因动作使用的最小优势绝对值；只对已归因动作生效，避免把整局动作一并强化。 |
| `update.max_length` | 更新时每个动作 observation 的最大 token 长度。 |
| `update.learning_rate` | LoRA 更新学习率。 |
| `update.clip_range` | PPO ratio 裁剪范围。 |
| `update.update_epochs` | 每个 K 条完整轨迹上的优化轮数。 |
| `update.max_grad_norm` | 梯度范数裁剪上限。 |
| `update.reinforce_ce_weight` | 归因 `reinforce` 动作的局部交叉熵纠错系数；`0` 可关闭该附加项。 |
| `update.weaken_unlikelihood_weight` | 归因 `weaken` 动作的 unlikelihood 纠错系数；只有存在其他合法动作时生效，`0` 可关闭。 |
| `update.load_in_4bit` | 以 4-bit 量化加载基座以降低显存。 |
| `update.gradient_checkpointing` | 用重算激活降低反向传播显存。 |
| `update.empty_cache_between_steps` | 变长动作更新间归还空闲 CUDA 缓存，降低碎片化。 |

远端同步迁移见 [远程推理与在线 RL](../doc/REMOTE_ONLINE_RL.md)。
