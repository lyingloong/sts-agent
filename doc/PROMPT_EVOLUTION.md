# Prompt 演化（随机首局、同种子复跑）

该模式用于在一个固定的《杀戮尖塔 1》种子上迭代远程教师的场景策略。它不是 SFT、GRPO 或硬编码规则：Qwen 不参与此模式，DeepSeek 通过当前 `teacher` USTC API 负责实际动作和事后提出策略候选。每次只替换一个场景 skill 的小段补充文字，动作仍然经过 controller 的合法动作表和编译器。

## 开关

推荐直接选择 `config/modes/prompt_evolution.json`。它是完整运行配置的严格稀疏覆盖；旧版也可以把 `config/agent.prompt_evolution.example.json` 的字段合并到 `config/agent.json`：

```json
{
  "policy": {"enabled": false},
  "teacher": {"enabled": true},
  "posthoc_feedback": {"enabled": false},
  "prompt_evolution": {"enabled": true}
}
```

其余 USTC 地址、模型和密钥沿用现有 `teacher` 配置。四个开关有意互斥：本模式不同时启动 Qwen、在线 RL 或终局归因。

## 初版 Prompt

初始教师 Prompt 在 [`tools/teacher_prompts.py`](../tools/teacher_prompts.py)，包含通用决策要求和 MAP、COMBAT、EVENT、商店、奖励、删牌等场景化提示，当前版本为 `sts1-teacher-zh-v4`。角色基础 Prompt 位于 [`prompts/characters`](../prompts/characters)，每个角色拥有独立的 `base.md` 和 `p0001` 策略版本；后续人工接受的演化只影响当前角色，不会覆盖其他角色的 Prompt。

## 一轮流程

1. 使用 Overlay 或终端输入 `s <角色> <进阶>` 或带种子的 `s <角色> <进阶> <种子>`。首次 incumbent 可以使用随机种子；游戏状态暴露实际显示种子后，controller 会把它写入本轮运行配置并作为后续候选复跑的锚点。
2. controller 以当前角色目录 `prompts/characters/<角色>/manifest.json` 的 `active_version` 作为 incumbent，在该锚定种子完整运行；现有 `recent_actions`、战斗历史、事件历史和合法动作表照常送给教师。
3. 终局后，DeepSeek 读取完整的动态动作账本和终局指标，提出一个候选 JSON。它必须引用 1-8 个日志步骤，只修改一个 skill，不能把当前种子、楼层或具体随机结果写成规则。
4. 候选只写入 `data/prompt_evolution/sessions/<session>/candidates/`，并以候选策略在同一角色、进阶、种子上自动重开一局。候选不会改变已接受的 active version。
5. 候选跑完后，Overlay 弹出人工审核窗口，显示新旧 Act/Floor/HP/终局、分数变化、策略 diff、证据步骤和风险。人工可以接受、拒绝并填写理由，或放弃。
6. 接受：在当前角色的 `prompts/characters/<角色>/versions/pNNNN.json` 生成不可变版本，原子更新该角色 manifest，并可立即开始下一轮。拒绝：理由传回 DeepSeek，只允许在配置的次数内重写同一个候选；放弃：当前角色的 active version 不变。

## 评分与限制

评分仅用于同一种子的可解释比较：胜利、Act、Floor、HP 比例。它不是自动接受门槛；即便分数上升，也必须由人工确认。同一种子结果不能证明跨种子泛化，之后需要另行做独立评测。如果游戏没有向 CommunicationMod 暴露可复用的显示种子，本轮仍可运行，但终局后不会自动启动候选复跑，并会提示无法建立种子锚点。

策略版本文件只保存接受的 skill 补充，基础中文场景提示仍在 `tools/teacher_prompts.py`。候选、重跑轨迹、教师诊断、拒绝理由和 session state 在 `.gitignore` 中，不进入 Git；接受的版本应和相关代码/文档一起提交。

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
