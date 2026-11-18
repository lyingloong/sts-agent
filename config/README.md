# 运行配置

新检出默认使用 [agent.example.json](agent.example.json) 的 `collect_human` 模式，可直接采集人工轨迹。
需要改模式或共享参数时，将它复制为同目录的 `agent.json`，在副本中修改。
`config/agent.json` 是 Git 忽略的本机设置，存在时优先生效。省略 `--config` 或传入标准
`config/agent.json` 路径都会遵循上述规则。指定其他文件时，程序读取该文件及其同目录
的 `modes/`。文件旁没有 `modes/` 目录时，使用项目的 `config/modes/`。
`gpu-server` 和 `/srv/sts-agent` 是远端模式的占位值，使用远端模式前需要替换。

从旧版本升级且希望沿用原配置，请在拉取更新前另存当前的 `config/agent.json`，
更新后将副本放回原位置。若已完成更新，之前提交过的配置可从相应 Git 版本恢复。

## 切换运行模式

在本机 `agent.json` 里选择 `active_mode`，框架会据此设置控制方式、模型后端和学习方法。
以下是只指定一个模式的完整配置。

```json
{"schema_version": 2, "active_mode": "collect_human"}
```

每行链接指向仓库提供的模式默认值。个人设置写在下文的 `shared` 或 `mode_overrides` 中。

| 模式 | 用途 | 首次使用前准备 |
| --- | --- | --- |
| [collect_human](modes/collect_human.json) | 记录玩家操作 | 启用 ActionRecorder 和 CommunicationMod，录制端口与 `recording.listen.port` 一致 |
| [collect_codex](modes/collect_codex.json) | 通过 MCP 控制 A0 铁甲战士 | 安装 `.[codex]` 可选依赖并按[客户端指南](../doc/CODEX_PLAY_A0_IRONCLAD.md)接入 |
| [baseline_api](modes/baseline_api.json) | 托管 API 选择动作 | 填模型名、网关地址和密钥文件，见[配置示例](#托管-api) |
| [baseline_inference](modes/baseline_inference.json) | 自建服务运行 27B BF16 基线 | 准备远端权重及常驻模型服务，设置 SSH 主机和 GPU 参数 |
| [candidate_assist_local](modes/candidate_assist_local.json) / [candidate_assist_remote](modes/candidate_assist_remote.json) | Qwen 提供候选动作 | 模型所在机器安装 `requirements/policy.txt`，核对 `policy.model.base`；远端模式还需设置 `policy.remote`。默认允许高置信度动作自动执行 |
| [dagger_local](modes/dagger_local.json) / [dagger_remote](modes/dagger_remote.json) | 记录玩家对候选动作的纠正 | 准备与候选模式相同的模型环境，默认由玩家确认 |
| [local_inference](modes/local_inference.json) / [remote_inference](modes/remote_inference.json) | 固定模型自动游玩 | 使用相应的本机或 SSH 模型环境，按需指定 `policy.model.initial_adapter` |
| [teacher_sft](modes/teacher_sft.json) | 教师游玩，保存教师轨迹 | 设置 `shared.teacher` 的 `base_url`、`model` 和 `api_key_file` |
| [prompt_evolution](modes/prompt_evolution.json) | 教师提议调整角色提示，由玩家审核 | 配置同一组教师参数，确保提示目录和输出目录可写 |
| [online_rl](modes/online_rl.json) | 远端采样并更新策略 | 设置 SSH、已有的 SFT adapter，训练机准备 `config/online_rl.json`；启用事后归因时还需教师 API |

切换模式后重启游戏/controller 和 Overlay。启动前可用离线命令检查配置：

```powershell
python -m sts_agent.commands.check_config
python -m sts_agent.commands.check_config --list
python -m sts_agent.commands.check_config --mode baseline_api
python -m sts_agent.commands.check_config --all
```

检查只解析配置。`--all` 会列出每个模式的结果；新模板会提示 `online_rl` 缺少初始 adapter。
只有启用该模式才需要填入真实路径。

## 参数覆盖顺序

配置依次加载内建默认值、`agent.json` 的 `shared`、版本库的 `modes/<模式名>.json`、
最后是本机 `agent.json` 的 `mode_overrides.<模式名>`。后面的同名字段优先。只覆盖需要改的字段；
列表与 `policy.api.extra_body` 整项替换。多个模式共用的 SSH 地址或教师 API 放在 `shared`，
某个模式的模型、网关或自动执行的置信度阈值放在它自己的 `mode_overrides` 中。

## 托管 API

下面是一份可复制的 `config/agent.json`。换成服务商实际的模型名、版本标记和 API 根地址，
将密钥存为项目根目录的 `provider_api.txt`。项目的 `*_api.txt` 规则会忽略该文件。

```json
{
  "schema_version": 2,
  "active_mode": "baseline_api",
  "mode_overrides": {
    "baseline_api": {
      "policy": {
        "model": {"base": "my-model", "revision": "provider-version"},
        "api": {
          "base_url": "https://api.example.org/v1",
          "api_key_file": "provider_api.txt",
          "output_dir": "data/experiments/baseline-hosted",
          "rpm_limit": 18
        }
      }
    }
  }
}
```

公开 profile 含科大网关的示例值。使用其他服务时，`revision` 应记录可核对的模型版本或
访问日期，`rpm_limit` 按账户配额填写。供应商扩展参数放进 `extra_body`；`model`、
`messages`、`temperature`、`max_tokens` 和 `stream` 由客户端构建。其余参数见[基线推理](../doc/BASELINE_INFERENCE.md)。

## 其他配置文件

- `evaluation/a0_ironclad_10.json` / `evaluation/a0_ironclad_20_cont.json` / `evaluation/a0_ironclad_30.json`：独立的固定测试集，分别包含前 10 个、第 11–30 个、全部 30 个种子，另有 smoke 单种子集；它们不是运行模式，也不重复定义模型或 prompt。批量脚本必须连接已经运行在 `baseline_inference` 或 `baseline_api` 的 controller；批次默认按种子复用同模型、同提示词配置的已完成对局（`--no-reuse` 可关闭），10 + 20 两批也可用 `merge_batches` 手动合并。
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

选择 `active_mode: "collect_codex"`。需要固定种子时，在 `mode_overrides.collect_codex.codex.seed`
中填写游戏显示的种子。当前支持 A0 铁甲战士。客户端注册和游玩规程见 [Codex 游玩说明](../doc/CODEX_PLAY_A0_IRONCLAD.md)。

## 教师与 Prompt Evolution

在 `shared.teacher` 设置教师网关的 `base_url`、`model` 和 `api_key_file`，两个教师模式都可使用。
密钥文件路径相对于项目根目录，内容可为单行密钥或 `key: ...`。角色提示长度、轨迹输入上限和改写次数的
默认值见[模式文件](modes/prompt_evolution.json)，本机可用 `mode_overrides.prompt_evolution.prompt_evolution` 调整。

## 在线 RL

选择 `active_mode: "online_rl"`。在 `mode_overrides.online_rl.policy.model.initial_adapter` 指定远端训练机上的初始 SFT adapter；
若多个策略模式共用此 adapter，也可写入 `shared.policy.model.initial_adapter`。事后归因使用的 API 与开关
位于 `mode_overrides.online_rl.posthoc_feedback`。远端 worker 从训练机的 `config/online_rl.json`
读取组大小、奖励和更新超参数。每个新实验使用独立 `output_root`；已有实验会从自己的 `state.json` 恢复当前策略。
