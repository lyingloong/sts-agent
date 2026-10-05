# Prompt 演化（随机首局、同种子复跑）

该模式用于在一个固定的《杀戮尖塔 1》种子上迭代动作模型使用的角色 skill 或场景 skill。它不是 SFT、GRPO 或硬编码规则：Qwen 不参与此模式，当前 `shared.teacher` 配置的 DeepSeek endpoint 以玩家身份负责实际动作，并以独立的 Prompt 演化器身份事后提出 skill 候选。每次只替换一个 skill，动作仍然经过 controller 的合法动作表和编译器。

## 开关

在 `config/agent.json` 中把 `active_mode` 改为 `prompt_evolution`。Prompt Evolution 参数在 `config/modes/prompt_evolution.json`；Teacher API 等共享设置在 `agent.json` 的 `shared.teacher`。

当前配置使用阿里云 DeepSeek；如需改服务，修改 `shared.teacher`。活动 profile 会自动选择教师后端并关闭 Qwen 与终局归因，避免手工开关冲突。

教师请求支持配置化重试：网络断开、超时、429、503 等临时错误会按指数退避自动重试；认证或请求参数错误不会无意义地重复请求。达到 `teacher.max_retries` 后才回退为不可用并保留本局轨迹，不会使 controller 进程退出。相关参数见 [`config/README.md`](../config/README.md)。

## 初版 Prompt

基础动作 Prompt 在 [`sts_agent.policy.teacher_prompts`](../sts_agent/policy/teacher_prompts.py)，包含通用决策要求和 MAP、COMBAT、EVENT、商店、奖励、删牌等场景化提示。教师 SFT 使用 `sts1-teacher-zh-v7` 的“远程教师”身份；本模式的实际动作请求使用 `sts1-player-zh-v3` 的“正在游玩的玩家”身份。角色基础 Prompt 位于 [`prompts/characters`](../prompts/characters)，每个角色拥有独立的 `base.md` 和 `p0001` 策略版本；后续人工接受的演化只影响当前角色，不会覆盖其他角色的 Prompt。

## 动作 Prompt 组成

Prompt 演化运行时，每次实际动作请求的 system prompt 按以下顺序拼接：

1. **玩家身份与通用决策规则**：声明模型正在亲自游玩，要求提高整局胜率、只使用公开信息、服从当前合法动作表和 `decision_constraints`，并以严格 JSON 返回一个 `action_id`。
2. **角色专属基础 skill**：从 `prompts/characters/<角色>/base.md` 读取当前角色的力量/毒/充能球/姿态等构筑原则。角色库彼此隔离。
3. **场景 skill**：根据 `screen_type`、`room_phase` 和 `selection_context` 选择一个场景块，例如 `MAP`、`COMBAT`、`CARD_REWARD`、`GRID_PURGE`、`SHOP_SCREEN` 或 `EVENT`。敌人及其他实体描述只使用当前游戏状态中提供的运行时字段；缺失时保持未知，不从离线知识库补全。
4. **版本化 skill**：从当前角色 `manifest.json` 指向的策略版本中读取 `role_skill` 和当前场景的 `scene_skills`。它们就是实际参与决策的角色/场景 skill，不再存在额外的“策略补充”层；它们不能覆盖合法动作表或虚构状态。当前版本标识为 `sts-prompt-evolution-v2`。

system prompt 之后的 user prompt 另行包含压缩后的公开 observation 和本步完整合法动作别名表 `A1/A2/...`。动作模型只从该别名表中选择一个动作；动作结果不会把教师思考或候选生成器内容写入训练数据。

终局后用于产生下一个 skill 候选的是另一条独立请求：它的身份是“Prompt 演化器”，输入一个 Act 的公开动作账本、终局指标和当前 skill，只能提出一个角色 skill 或场景 skill 的小幅替换。一个完整运行会按 Act 顺序逐段分析，每次只把当前 Act 发给教师；候选必须经过人工批准才会复跑同一种子，验证通过后再进入下一个 Act。它不是游戏动作模型，不会直接执行命令。

## 一轮流程

1. 使用 Overlay 或终端输入 `s <角色> <进阶>` 或带种子的 `s <角色> <进阶> <种子>`。首次 incumbent 可以使用随机种子；游戏状态暴露实际显示种子后，controller 会把它写入本轮运行配置并作为后续候选复跑的锚点。
2. controller 以当前角色目录 `prompts/characters/<角色>/manifest.json` 的 `active_version` 作为 incumbent，在该锚定种子完整运行；现有 `recent_actions`、战斗历史、事件历史和合法动作表照常送给教师。
3. 终局后，DeepSeek 按 Act 读取该段压缩后的动态动作账本和终局指标，提出一个候选 JSON。它必须引用 1-8 个日志步骤，只修改一个 skill，不能把当前种子、楼层或具体随机结果写成规则。
4. 候选只写入 `data/prompt_evolution/sessions/<session>/candidates/`，先弹出“候选验证前审核”。此时只显示 incumbent 的结果、候选 skill diff、证据和风险，不会启动下一局。
5. 人工接受后，controller 才以候选 skill 在同一角色、进阶、种子上重开验证局；拒绝则把理由传回 DeepSeek 重写候选，并再次等待审核；放弃则结束本轮。候选在验证前不会改变已接受的 active version。
6. 候选验证跑完后，Overlay 再弹出“验证结果审核”，显示新旧 Act/Floor/HP/终局、分数变化、策略 diff、证据步骤和风险。只有人工接受，才在当前角色的 `prompts/characters/<角色>/versions/pNNNN.json` 生成不可变版本并原子更新 manifest；开启自动继续时，也会先审核下一候选再运行。

## 评分与限制

评分仅用于同一种子的可解释比较：胜利、Act、Floor、HP 比例。它不是自动接受门槛；即便分数上升，也必须由人工确认。同一种子结果不能证明跨种子泛化，之后需要另行做独立评测。如果游戏没有向 CommunicationMod 暴露可复用的显示种子，本轮仍可运行，但终局后不会自动启动候选复跑，并会提示无法建立种子锚点。

策略版本文件直接保存接受后的 `role_skill` 与 `scene_skills`。候选、重跑轨迹、教师诊断、拒绝理由和 session state 在 `.gitignore` 中，不进入 Git；接受的版本应和相关代码/文档一起提交。

### 教师输入的完整性

完整原始账本仍会写入运行 artifact，但远程候选生成器收到的是按 Act 压缩后的公开账本，而不是只截取错误步骤：策略动作保留动作前动态状态、屏幕转换、必要的合法动作参数、执行命令、动作来源和模型候选概率；通常一步的 `state_after` 会等于下一步的 `state`，因此只保留 `state_after_ref` 或跨屏幕/终局时的状态变化。卡组、遗物和药水在 Act 开始或发生变化时发送。自动推进/自动拾取只保留一行边界标记；地图图、Boss 详细招式表、静态卡牌/遗物/药水描述和完整终局状态副本都不发送。字段、列表和单项文本均有长度上限。

因此它比此前 GRPO 事后归因请求更适合做 prompt 演化：归因账本主要提供采样动作的别名和动作前状态，而候选生成器还可以比较动作前后变化、环境自动动作和人工覆盖，并检查完整的局部因果链。发送给教师的 Act 账本由以下部分组成：

1. `analysis_scope`：Act 编号和该段首尾 step。
2. `incumbent_policy`：角色 skill、当前 Act 实际出现的场景 skill，以及可选 skill 键名；不会重复发送未出现的场景全文。
3. `terminal` 与 `metrics`：终局类型、Act/Floor、HP、胜负和评分指标；不再发送完整终局状态副本。
4. `act_context`：该 Act 首次观测到的卡组、遗物和药水。
5. `action_ledger`：策略动作的合法动作摘要、选中动作、模型概率元数据、关键动态状态和必要的状态变化引用。卡组/遗物/药水只在首次或发生变化时发送；地图完整图、Boss 详细招式表、静态物品描述不会进入该请求。控制器自动推进、自动拾取、自动打开奖励等只保留 `step_id`、场景和“自动推进”一行标记。

完整账本仍原样写入每局的 `runs/*.json`，用于审计和人工复盘；压缩后的 Act 账本只用于远程教师请求。默认分析输入上限为 `900000` 字符、输出上限为 `8192` token；如果真实序列化内容仍超出上限，系统会标记 `analysis_input_truncated`，并在 Overlay 中显示告警和弹窗，不应忽略该轮候选。

## 产物

```text
prompts/characters/<角色>/base.md
prompts/characters/<角色>/manifest.json
prompts/characters/<角色>/versions/p0001.json
data/prompt_evolution/sessions/<session>/
  state.json
  runs/<run>.json
  candidates/cNNNN.json
```

`runs/*.json` 保留完整动态账本（包括动作历史），但会移除静态卡牌、遗物、药水和能力描述，避免重复膨胀。运行期教师请求的思考和原始 completion 不写入这些持久文件。
