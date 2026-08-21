# StS Agent

针对《Slay the Spire 1》的结构化游戏 Agent 原型。当前目标是先用 CommunicationMod 建立可靠的人类轨迹采集链路，再训练小型 Qwen 模型做分层决策和无人值守运行。

## 当前状态

- 游戏：StS1 + ModTheSpire + BaseMod + CommunicationMod
- 采集方式：`communication_controller.py` 作为 CommunicationMod 外部进程，`human_terminal.py` 通过本机 TCP 提供输入
- 当前不需要额外编写 Java Mod；原生鼠标动作记录仍是后续工作
- 状态输入使用 CommunicationMod 的结构化 JSON，并在轨迹中移除抽牌顺序、UUID 和随机种子等隐藏信息
- 支持原版四角色：Ironclad、Silent、Defect、Watcher；支持 A0-A20 和可选种子
- 数据、模型和运行日志默认不提交 Git

## 快速开始

### 1. 配置 CommunicationMod

在 CommunicationMod 的 `config.properties` 中将 `command` 指向本项目的控制器，例如：

```properties
command=D\:/miniconda3/python.exe D\:/Code/spire-mod/sts-agent/tools/communication_controller.py
runAtGameStart=true
```

路径按本机 Python 和项目目录修改。重启 ModTheSpire 后，CommunicationMod 会自动启动控制器。

### 2. 启动终端客户端

```powershell
cd D:\Code\spire-mod\sts-agent
python tools\human_terminal.py
```

### 3. 开始游戏

在终端输入：

```text
s                         Ironclad A0，随机种子
s ABC123                  Ironclad A0，固定种子 ABC123
s SILENT 5 ABC123        Silent A5，固定种子
s DEFECT 10              Defect A10，随机种子
s WATCHER 20 ABC123      Watcher A20，固定种子
```

角色名也接受 `THE_SILENT`。旧的 `s [seed]` 写法继续可用。角色、进阶和种子会写入 `run-events-*.jsonl`，轨迹中的每条动作也带 `run_config`。

## 终端动作

```text
p <手牌序号> [敌人序号]   出牌；单个存活敌人时可以省略目标
e                        结束回合
u <药水槽位> [敌人序号]  使用药水
d <药水槽位>             丢弃药水
<序号> / m <序号>        选择事件、奖励等选项
c                        确认或继续
x                        取消、返回或跳过
s                        游戏中请求当前状态；主菜单中开始游戏
h                        显示帮助
q                        退出控制器
```

终端客户端每次刷新状态、帮助或错误都会清屏。`p 1` 在单敌人战斗中会自动解析为实际的 `PLAY 1 0`；多敌人时必须显式指定目标。

## 输出文件

控制器会自动创建以下目录：

```text
data/raw/communication_controller/messages-<session>.jsonl  # 原始协议消息
data/trajectories/action-trace-<session>.jsonl              # 状态-动作-结果轨迹
data/trajectories/run-events-<session>.jsonl                # 每局开始和种子事件
```

正式训练主要使用 `action-trace`。原始消息只用于排错和重新规范化，不能直接作为模型输入。中途退出的局应标记为调试数据，不进入正式 SFT 集。

## 文档

- [实施计划与训练路线](doc/PLAN.md)
- [通信协议与数据格式](doc/PROTOCOL.md)

## 验证

```powershell
python -m py_compile tools\communication_controller.py tools\human_terminal.py
python tools\human_terminal.py --help
```

完整游戏测试需要启动 ModTheSpire；代码测试不会替代实际协议测试。

## 设计边界

第一阶段只做结构化状态 Agent，不做视觉识别，不把 Qwen 嵌入游戏 Mod，不在一局内更新模型权重。SFT 用于冷启动，后续通过 shadow、assist 和 DAgger 收集模型访问到的困难状态。

CommunicationMod 的协议和限制见其[官方仓库](https://github.com/ForgottenArbiter/CommunicationMod)。
