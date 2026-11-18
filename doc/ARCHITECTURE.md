# Agent 架构

项目以 `sts_agent` 为唯一源码包。StS1/CommunicationMod 是当前游戏实现；策略和学习模块只依赖规范化 observation、合法动作与决策记录，不直接解析 Mod 原始消息。

## 运行链路

```text
CommunicationMod
  -> runtime.controller
  -> game adapter / StS1 runtime state projection
  -> core observation + legal actions
  -> policy backend or human UI
  -> validated action compiler
  -> CommunicationMod
```

每个稳定动作同时写入统一轨迹，供 SFT、Prompt Evolution、事后归因和在线 RL 使用。

## 模块边界

| 包 | 职责 | 不应负责 |
| --- | --- | --- |
| `core` | 数据契约、策略接口、framework 组合校验 | 游戏解析、模型加载、UI |
| `runtime` | CommunicationMod 协议、运行协调、轨迹生命周期 | 模型训练实现 |
| `game` | 游戏适配器、StS1 状态与运行时字段规范化 | API 调用、窗口、静态实体知识库 |
| `policy` | Qwen 推理、教师 API、策略 Prompt | 游戏命令执行 |
| `learning` | 数据导出、SFT、Prompt/RL 更新 | CommunicationMod 原始协议 |
| `ui` | 展示 controller view、收集人工选择和审核 | 游戏规则与训练更新 |
| `commands` | 一次性下载、刷新和诊断任务 | 常驻运行协调 |

依赖方向：

```text
core <- game
core <- policy
core + policy <- learning
core + game + policy + learning <- runtime
runtime view <- ui
```

`core` 不得导入 Transformers、Tk、HTTP 客户端或 CommunicationMod 实现，因此核心测试无需游戏和 GPU。

## 配置模型

`sts_agent.core.modes` 将模式名映射到固定的控制、策略后端和学习组合。加载器用共享设置、所选模式的公开配置和本机覆写合成 controller 的实际运行配置。数据来源、优先级与模式示例见[运行配置](../config/README.md)。

## 进程边界

- CommunicationMod 启动 `python -m sts_agent.runtime.controller`。
- controller 可在本地或 SSH 会话中启动 `sts_agent.policy.local_qwen`。
- 在线 RL 使用远端 `sts_agent.learning.online_rl.worker`，每组 rollout 结束后更新 LoRA adapter。
- Overlay/terminal 通过本机 socket 连接 controller，不能直接连接游戏或模型。
