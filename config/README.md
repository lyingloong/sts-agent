# 运行配置

新环境可从 [agent.example.json](agent.example.json) 复制生成 `agent.json`，模板默认只采集人工轨迹，
不会部署模型或请求教师。`gpu-server` 和 `/srv/sts-agent` 是远端模式的占位值，需要替换。
已有配置请保留，不要用模板覆盖；切换模式仍只需修改 `agent.json` 的 `active_mode`。

## 切换运行模式

日常只改 [`agent.json`](agent.json) 里的 `active_mode`，不要分别拼装控制方式、模型后端和学习方法：

```json
"active_mode": "collect_human"
```

模式专属参数放在 `config/modes/<模式名>.json`。当前支持：

| `active_mode` | 用途 | 配置文件 |
| --- | --- | --- |
| `collect_human` | 人类在游戏内游玩并采集轨迹 | [collect_human.json](modes/collect_human.json) |
| `baseline_inference` | 官方 27B BF16 基线；本机控制、远端常驻模型、自动 SSH 隧道，不训练 | [baseline_inference.json](modes/baseline_inference.json) |
| `baseline_api` | 托管 OpenAI 兼容 API 基线（USTC / 阿里云）；客户端限流与重试，不需要 GPU | [baseline_api.json](modes/baseline_api.json) |
| `candidate_assist_local` | 本机 Qwen 给出候选，由玩家选择或修正 | [candidate_assist_local.json](modes/candidate_assist_local.json) |
| `candidate_assist_remote` | 远端 Qwen 候选辅助 | [candidate_assist_remote.json](modes/candidate_assist_remote.json) |
| `teacher_sft` | 远端教师自动游玩并生成 SFT 数据 | [teacher_sft.json](modes/teacher_sft.json) |
| `dagger_local` / `dagger_remote` | Qwen 候选 + 人类纠正，采集 DAgger 数据 | [dagger_local.json](modes/dagger_local.json) / [dagger_remote.json](modes/dagger_remote.json) |
| `prompt_evolution` | 教师提出 skill 候选，人工审核迭代 | [prompt_evolution.json](modes/prompt_evolution.json) |
| `local_inference` / `remote_inference` | 本机 / 远端 Qwen 自动推理 | [local_inference.json](modes/local_inference.json) / [remote_inference.json](modes/remote_inference.json) |
| `online_rl` | 远端 Qwen rollout 与在线参数更新 | [online_rl.json](modes/online_rl.json) |
| `collect_codex` | Codex MCP 操作游戏并单独保存轨迹 | [collect_codex.json](modes/collect_codex.json) |

所有组合由 `sts_agent.core.modes` 固定映射并验证，避免出现“人工控制 + 模型后端”这类意外组合。切换后要重启游戏/controller 和 Overlay。启动前可检查活动模式：

```powershell
python -c "from sts_agent.config import load_agent_config; c=load_agent_config(); print(c['framework'])"
```

## 配置归属

- `agent.json`：活动模式选择，以及所有模式共享的 controller 地址、录制接收地址、模型/API 连接信息和 Overlay 布局。Qwen 的 `policy.model.initial_adapter` 只维护一次；Teacher API key 只写在被 Git 忽略的 key 文件中。
- `modes/*.json`：所选模式需要额外修改的参数。没有覆盖的字段使用 `sts_agent.config` 中经过校验的默认值。
- `evaluation/a0_ironclad_10.json` / `evaluation/a0_ironclad_20_cont.json` / `evaluation/a0_ironclad_30.json`：独立的固定测试集，分别包含前 10 个、第 11–30 个、全部 30 个种子，另有 smoke 单种子集；它们不是运行模式，也不重复定义模型或 prompt。批量脚本必须连接已经运行在 `baseline_inference` 或 `baseline_api` 的 controller；10 + 20 两批可用 `sts_agent.commands.merge_batches` 合并。
- `online_rl.json`：仅训练机上的在线 RL 实验参数，被 Git 忽略；结构见 [`online_rl.example.json`](online_rl.example.json) 和 [远端在线 RL 文档](../doc/REMOTE_ONLINE_RL.md)。

模式加载后，内部仍生成统一的运行配置供 controller、Overlay、MCP 和训练组件使用；该运行时配置不需要手动编辑。JSON 有严格字段校验，未知字段会在启动时明确报错。

## 官方模型基线

`baseline_inference` 的模型与全部 API/服务参数集中在该模式 profile 中，SSH 主机沿用共享的
`policy.remote.ssh_host`。本机使用 controller，远端使用 `python -m sts_agent.policy.serving`；
远端服务入口显式加载 baseline profile，无需修改远端 `active_mode`。
远端部署、输入限制与日志解释见 [基线推理文档](../doc/BASELINE_INFERENCE.md)。
首次实验关闭 thinking、无 adapter、生成一个动作并复核状态后自动执行，不使用 0.85 置信度门槛。
`policy.api.ssh_tunnel: true` 由本机 controller 自动创建隧道，不需要手动运行 SSH 转发。
`policy.api.serving.backend` 当前为 `transformers`，适配已有服务器驱动；也支持显式选择 `vllm`。
`tensor_parallel_size` 在 Transformers 后端表示可见 GPU 数，使用 `device_map=auto` 按层分配，
不是 vLLM 的张量并行。`gpu_memory_utilization`、`enforce_eager` 仅对 vLLM 生效。

`policy.api.prompt_format` 选择提示词的展示格式，默认 `plain_text`，使用现有中文分区和字段形式。
并列的 `markdown` 方案标为**待开发**，预览见 [Markdown 提示词方案](../doc/POLICY_PROMPT_PREVIEW_MARKDOWN.md)；
现在显式选择 `markdown` 会在配置校验时提示“待开发”，不会发送请求或静默回退。
`policy.api.prompt_template` 只控制区块顺序（`canonical`、`state_first`、`decision_first`），默认仍为 `canonical`。
当前继续使用 [原格式预览](../doc/POLICY_PROMPT_PREVIEW.md)，Markdown 方案不作为原格式的迭代替代。

合法动作仍使用当前请求内的 A 别名，同时保留手牌、敌人、药水槽和选项的位置引用；原始动作 ID 和
UUID 留在控制及审计层，不作为位置引用发送给模型。

批量测评入口为 `python -m sts_agent.commands.evaluate_baseline`，默认读取 `agent.json` 和
`evaluation/a0_ironclad_10.json`。`--dry-run` 只校验配置并显示计划，不联网、开局或写结果。
修改测试集后应使用新的 `suite_id`，保留原测试集用于配对比较。超时/动作预算是技术停止条件，
不是提前结束后计作死亡的规则；操作细节见[批量测评](../doc/BASELINE_EVALUATION.md#固定-10-种子批量测评)。

## 人类轨迹采集

使用 `active_mode: "collect_human"`。CommunicationMod 的统一入口 `sts_agent.runtime.controller` 会自动转入 `sts_agent.recording.communication_bridge`（也可直接启动后者）：它从 CommunicationMod stdin 和 ActionRecorder TCP 接收同一编号的状态，并在 `recording.listen.host` / `recording.listen.port`（默认 `127.0.0.1:8766`）接收动作事务标记。整理后的轨迹写入 `recording.output_dir`，默认 `data/trajectories/human/`。此模式不启动策略模型或教师模型，不需要打开终端/Overlay 下指令。

## Codex MCP 采集

选择 `active_mode: "collect_codex"`。模式参数位于对应 profile 中；A0 Ironclad 是当前唯一支持的配置。客户端注册和游玩规程见 [Codex 游玩说明](../doc/CODEX_PLAY_A0_IRONCLAD.md)。

## 教师与 Prompt Evolution

`teacher` 的 API、重试、thinking/streaming、token 上限等提供方设置是共享项。Prompt Evolution 的 skill 长度、轨迹输入上限、改写次数等放在 `modes/prompt_evolution.json`。游戏实体说明从当前游戏状态读取，不需要知识库路径或经验开关。

## 在线 RL

选择 `active_mode: "online_rl"`，并确认共享项中的 `policy.model.initial_adapter` 指向可用的初始 adapter。人类/自动事后归因参数在该模式 profile 中；远端 worker 的 group size、奖励和更新超参数写入训练机的 `config/online_rl.json`。每个新实验使用独立 `output_root`；已有实验会从自己的 `state.json` 恢复当前策略。
