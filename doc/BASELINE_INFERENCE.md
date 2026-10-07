# 官方模型游戏基线：Qwen3.8-27B BF16

## 架构与实验边界

本机与远端使用同一个 Git 仓库、同一份 `baseline_inference` profile，但运行不同入口：

| 位置 | 入口 | 职责 |
| --- | --- | --- |
| 本机 Windows | `sts_agent.runtime.controller` | 接 CommunicationMod，整理公开状态/合法动作，执行动作、保存轨迹 |
| 本机 Windows | `sts_agent.ui.overlay` | 开局、人类接管、展示当前模型动作/请求状态 |
| 远端 Linux | `sts_agent.policy.serving` | 一次加载模型，持续提供兼容 API；不运行游戏、不训练 |

本机建立 `127.0.0.1:18080 → SSH → 远端 127.0.0.1:8000` 的持久通道。
关闭本机 controller 只关闭它拥有的 SSH 通道，不关闭远端模型服务。
代码只在本机修改，通过 Git 同步；服务不会自行 pull、修改代码或切换 GPU。

首次测试使用官方 `Qwen/Qwen3.8-27B` 后训练权重，不挂 adapter，不量化权重。
其名称没有 `Instruct` 后缀，不意味着它是未经后训练的 Base 模型。
模型与服务支持情况见 [官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)。

目标是分清原生模型能力、状态信息缺失、输出格式错误与控制器错误。
不是这次就开始 SFT/GRPO，也不能据此保证通关。

## 默认设置

所有实验参数位于 [`config/modes/baseline_inference.json`](../config/modes/baseline_inference.json)。
SSH 主机来自 `agent.json.shared.policy.remote.ssh_host`。模型只在 `policy.model.base` 定义一次。

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `policy.model.revision` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | Qwen 官方仓库的固定 SHA；镜像 SHA 已与 Hugging Face 官方 API 核对 |
| `policy.model.hub_endpoint` | `https://hf-mirror.com` | 远端下载入口；只影响获取权重，不改模型内容或版本 |
| `policy.api.temperature` | 0 | 贪心生成；多 GPU 算子等因素仍可能导致细微非确定性 |
| `policy.api.thinking` | false | 显式传入 `chat_template_kwargs.enable_thinking=false`，只生成一个动作别名 |
| `policy.api.max_output_tokens` | 64 | 回答预算；`finish_reason=length` 是错误，即使回答看起来包含合法别名也不执行 |
| `policy.api.prompt_template` | canonical | 与 SFT 同源的自然语言单动作 prompt |
| `policy.api.timeout_seconds` | 120 | 每次 HTTP 请求的超时，不是远端模型加载时间 |
| `policy.api.max_retries` | 2 | 初次之外最多重试两次；仅网络/429/部分 5xx，等待 5、10 秒 |
| `policy.api.ssh_tunnel` | true | controller 自动建立持久隧道；不要另开同端口手动隧道 |
| `serving.backend` | transformers | 固定 Transformers 5.17.0 的服务适配器；可显式选择 vllm |
| `serving.tensor_parallel_size` | 4 | 可见 GPU 数；Transformers 按层分配，vLLM 张量并行，均不是四个独立模型 |
| `serving.devices` | auto | 启动时选卡一次；也可指定如 `[0, 2, 4, 6]`，必须四张互不重复的物理 GPU |
| `serving.minimum_free_memory_mib` | 21000 | 每张卡至少 21000 MiB 空闲，不够则启动失败，不占用/杀掉别人的任务 |
| `serving.maximum_gpu_utilization_percent` | 10 | 筛掉当前明显忙碌的 GPU；查询不是调度器预留，请与其他使用者协调 |
| `serving.max_model_len` | 16384 | 输入与输出合计上限，不启用模型的超长上下文 |
| `serving.gpu_memory_utilization` | 0.85 | 仅 vLLM：每卡显存预算比例；不是只加载 85% 权重 |
| `serving.enforce_eager` | true | 仅 vLLM：关闭 CUDA graph；Transformers 本就未开启 compile |
| `serving.disable_nccl_p2p/ib` | true/true | 消费级 4090 的稳妥初测设置，由启动器设置进程环境，不需维护环境变量 |
| `serving.deployment_file` | data/runtime/baseline-deployment.json | 远端下载 SHA、snapshot、部署命令、GPU 与依赖版本记录；不提交 Git |

`serving` 是 `policy.api.serving` 的简写。服务以 BF16 权重和自动（随模型 dtype）KV 精度运行，
纯文本、串行生成、无 LoRA、无 MTP/推测解码。实际 HTTP 请求明确指定温度和回答预算。
Transformers 适配器复用上游 `ChatCompletionHandler` 的生成实现，增加固定 SHA 元数据、
同模板 `/tokenize` 和服务端超限检查；仅加载 deployment 中的本地 snapshot，不跟随浮动 main。
上游 `transformers serve` CLI 单独运行缺少本客户端需要的 SHA 与 `/tokenize`，不能直接替代此入口。

资源预估：权重约 50–55 GiB，四张 24 GiB 4090 可以分摊；KV、运行缓冲等另占显存。
当前四卡已成功加载权重，短请求每卡占用约 12–14 GiB；长游戏输入仍会增加缓存与运行缓冲。
准备约 65–80 GB 可用磁盘空间；HF cache snapshot 通过链接复用 blobs，不应手工复制两份权重。
本机不加载权重，3050 不新增模型显存占用。
已验证短请求不代表完整游戏吞吐或通关能力已经验证。

## 远端：同步并单独准备服务环境

在远端仓库目录执行（目录与 SSH 主机沿用自己的现有配置）：

```bash
git pull --ff-only
uv venv .venv-serving --python 3.12
uv pip install --python .venv-serving/bin/python torch==2.11.0 torchaudio==2.11.0 torchvision==0.26.0 --torch-backend=cu128
uv pip install --python .venv-serving/bin/python -e '.[serving-transformers]'
.venv-serving/bin/python -m sts_agent.policy.serving --prepare
```

不要覆盖原训练 `.venv`。下载使用 profile 的 `policy.model.hub_endpoint`；当前配置使用 hf-mirror，
权重文件放在项目内 `models/hub/`。`--prepare` 才会下载，并记录精确 SHA，下载完成后就退出，不占 GPU。
模型源、SHA 或镜像变更时，先检查新版本 SHA，再更新 profile 并提交。
本机不需要远端的 deployment 文件，也不需要模型权重。

当前服务器驱动 535.104.05，vLLM 0.24.0 的 CUDA 12.9/NCCL 在这台机器报驱动不足，
所以改用已能加载与生成的 Transformers + Torch 2.11.0/cu128，不升级系统驱动、不影响其他人的任务。
适配器依赖上游内部接口，严格锁定 Transformers 5.17.0；升级前应重新验证接口。
以后若驱动兼容且需要更高吞吐，可设置 `backend: "vllm"` 并安装 `.[serving]`；
不要回退到旧 Qwen3/4-bit 入口冒充 BF16 基线。
服务会记录实际依赖版本。当前部分混合注意力算子使用参考实现，真实长输入延迟需游戏实测。

```bash
.venv-serving/bin/python -m sts_agent.policy.serving --dry-run
.venv-serving/bin/python -m sts_agent.policy.serving
```

第一条检查 snapshot、空闲卡和基础依赖并打印命令，不加载 GPU；不等价于完整引擎启动验证。
第二条以前台方式运行。建议在自己的 `tmux` 会话中运行，等待 Uvicorn 的服务就绪消息。
只有 loopback 监听，不需要对公网开放端口。
若四张卡不足够空闲，等待或手动指定可用四卡，不能强行绕过其他任务。

## 本机：先查接口，再开游戏

远端就绪后，在本机仓库根目录执行：

```powershell
uv pip install --python .venv\Scripts\python.exe -e .
.\.venv\Scripts\python.exe -m sts_agent.policy.api
```

此命令使用 baseline profile，即使当前 `active_mode` 仍是 `collect_human`；不会修改配置或开游戏。
它临时开 SSH 通道，验证 `/v1/models`、用 `/tokenize` 检查输入预算，测试生成 `A1`，然后关闭通道。
需要已有无交互 SSH 登录能力，不会弹出密码终端；失败信息和 SSH stderr 写入实验目录。
不要与正在游玩的 baseline controller 同时运行这个检查命令，以免争用本机 18080 端口。
手动维护隧道时，可将 `ssh_tunnel` 设为 false，自己负责端口对应正确的服务。

建议还用已有人工轨迹做少量离线比较，不会发送任何游戏命令：

```powershell
.\.venv\Scripts\python.exe -m sts_agent.policy.api --replay <sts-agent整理后的人类轨迹.jsonl> --max-states 20
```

必须使用 sts-agent 的 `pre_state + legal_actions` 格式，不是 ActionRecorder 原始 JSONL。
比较只检查模型与人的动作一致性，不能当作最优动作准确率或通关率；首个接口/格式错误会停止。

确认接口正常后，把 `config/agent.json` 的 `active_mode` 改为 `baseline_inference`，重启游戏/controller 和 Overlay：

```powershell
.\.venv\Scripts\python.exe -m sts_agent.ui.overlay
```

CommunicationMod 继续使用统一的 `sts_agent.runtime.controller` 入口，不用改 command。
Overlay 在主菜单输入 `s IRONCLAD 0` 开随机种子 A0 战士，或加上指定的种子。
它自动完成本局，但不会自动开第二局。模型请求期间不要在游戏中手动点击操作。
若 controller 接入时已经在局内，会继续该局并标注 `partial_run=true`，不能计作完整自主通关样本。
恢复人工采集时把 `active_mode` 改回 `collect_human` 并重启，不需要卸载模型或改远端模式。

## 每步路径、安全与数据

1. CommunicationMod 提供可决策状态；已有环境自动动作优先执行，保持 0.25 秒间隔并保留来源。
2. 普通决策使用同一套 `policy_observation` 路由和 `build_scoring_prompt`：地图只在地图界面注入，
   保留当前牌组、运行时实体说明、可见战斗信息、姿态及现有动作历史；不恢复静态知识库或教师 skill。
3. 对当前合法动作建立 `A1…An → 原始动作/命令` 映射，调用 API 生成一个别名。
4. `/tokenize` 使用与生成相同的 messages/chat template 设置。输入加完整回答预算超过上限则报错；
   不删字段、不截断字符串，也不设置 `truncate_prompt_tokens`。
5. 只接受完整回答中的一个精确合法别名。`A2` 可接受，`我选择 A2`、JSON、非法 ID 或截断都暂停，
   不随机回退，不强行提取看起来像答案的子串。
6. 回答后发 `STATE` 请求新观察；run ID、模型可见状态和完整动作列表的指纹都一致才执行。
   发生变化时丢弃旧回答并重新决策；查询不算游戏动作，不进入动作训练数据。
   这不是游戏事务锁，仍不能保证复核后到提交之间用户手动点击完全无竞态。
7. 游戏返回结果后，保留 controller 的动作轨迹以及执行接受/拒绝日志。
   `accepted=true` 表示协议未拒绝，不保证动画已彻底结算；后状态沿用现有记录语义。

HTTP/模型错误后交回人工。需要恢复时可在 UI 输入 `s` 刷新当前状态，或手动执行合法动作；
人工接管会留在本次评测记录中，统计无干预通关率时必须单独区分。

每个 controller/检查会话使用独立目录：`data/experiments/baseline/<时间-随机后缀>/`，其中包含：

- `manifest.json`：模型名、请求 revision、可从服务器 snapshot 识别的 SHA、代码 commit/dirty 标记、prompt 代码哈希、配置。
- `requests.jsonl`：完整拼接 prompt、别名映射、run ID、原始 API 回答、token 使用、总请求延迟、重试/错误、状态复核及执行结果。
- `messages-*.jsonl`：游戏消息，仅游戏 controller 会话产生。
- `action-trace-*.jsonl` / `run-events-*.jsonl`：逐局轨迹、种子、来源及终局事件，仅开局后产生。
- `ssh.log`：本机会话自己的 SSH stderr。
- `summary.json`：各局终局状态与模型/环境/人工动作数、拒绝动作数；未终局时胜负为 null。

这里不记录未经请求的候选概率；生成输出的概率并不等同于所有合法动作的归一化概率。
manifest 不能单凭 API 验证实际 GPU dtype，BF16 依据远端部署命令和 deployment 记录确认。
这些文件被 Git 忽略，与 `data/trajectories/human` 分离。
ActionRecorder 仍可能独立记录 AI 操作到原始 run 日志，它尚不能自动辨识操作者；
人工数据整理时必须结合种子/时间排除这些 AI 局，不能因为它们也是 Recorder 文件就当成人工数据。

## 验证现状

本地自动测试覆盖配置切换、单动作解析、输入超限拒绝、截断拒绝、HTTP JSON 通道与重试、
旧状态丢弃、controller 的 STATE 复核链路、GPU 筛选、部署 SHA/BF16 检查及 SSH 参数。
旧本机候选评分另修复了 `DynamicCache` 复用污染：各候选从独立缓存副本开始，
用微型 Qwen3 的完整前向概率验证与候选遍历顺序无关。

远端已下载并在四张 4090 上加载 27B BF16，尚未完成真实游戏评测。接下来先跑实际客户端检查与少量离线样本，
再跑完整 A0 游戏，记录合法输出率、人工接管、每步延迟、到达幕/楼层、通关与心脏结果。
