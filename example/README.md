# 强化学习八连例（RL by Example）

一条**由浅入深**的强化学习学习线，配套 [autoresearch](../README.md) 项目。
前四个例子层层递进，每一个都在上一个的基础上**多引入一个新概念**，落到
大模型后训练（RLHF / GRPO）——也就是 autoresearch 这类项目的同一个内核：
**试错 → 打分 → 保留更好的**。第 5 课不加新概念，把 GRPO **拆成最小骨架**做受控实验；
第 6 课把多步迷宫和 GRPO 拼起来，做一个会**调用工具**的 **Agentic RL** 最小例子；
第 7 课换上**真的跑单测的 Reward System**，亲眼看 **reward hacking**（弱奖励下 agent 学会作弊）；
第 8 课专门拆开 **训练 vs 推理** 这条界线（探索/梯度/checkpoint），所有深度学习/LLM 都通用。

> 想先看**整条路线的全景图**（SFT → DPO → GRPO → Agentic RL + Infra，对应一份 12 周学习计划），
> 浏览器打开 [`agentic_rl_roadmap.html`](agentic_rl_roadmap.html) —— 把这 7 个代码例子嵌进了完整的后训练主线里。

所有代码都带**详细中文注释**，在 macOS / Apple Silicon 上用系统 `python3` 就能直接跑
（依赖 numpy / torch(MPS) / matplotlib，均已具备；**不需要** gymnasium，CartPole 物理引擎是自己实现的）。

## 八个例子

| # | 文件 | 学到的新东西 | 一句话 |
|---|------|------------|--------|
| 01 | [`01_bandit.py`](01_bandit.py) | 探索 vs 利用、ε-greedy、UCB | 只有动作和奖励的最简 RL：该选哪台老虎机 |
| 02 | [`02_qlearning_gridworld.py`](02_qlearning_gridworld.py) | **状态**、Q 表、TD 更新、γ/α | 走迷宫，学会"在每个格子该往哪走" |
| 03 | [`03_cartpole_reinforce.py`](03_cartpole_reinforce.py) | **神经网络代替表格**、策略梯度 | 倒立摆，用小网络直接学"策略" |
| 04 | [`04_grpo_tiny_llm.py`](04_grpo_tiny_llm.py) | **在语言模型上做 RL**、GRPO 组相对优势、reward hacking | 用奖励信号教小 GPT 学会排序 |
| 05 | [`05_grpo_minimal.py`](05_grpo_minimal.py) | **把 GRPO 拆成最小骨架**、组归一化优势、KL 锚点、受控魔改实验 | 40 行只剩算法本身，能背能改能做消融 |
| 06 | [`06_agentic_rl_tool.py`](06_agentic_rl_tool.py) | **Agentic RL**：多步行动 + 调用工具 + 稀疏奖励 + 信用分配 | 黑暗走廊里 agent 自己学会"先 LOOK 探路、再走到终点" |
| 07 | [`07_agentic_codeRL.py`](07_agentic_codeRL.py) | **Reward System**：沙箱跑单测当奖励、**reward hacking**、过程vs结果奖励 | 弱奖励让 agent 学会作弊蒙混，稳健奖励才逼出真解 |
| 08 | [`08_train_vs_infer.py`](08_train_vs_infer.py) | **训练 vs 推理** 的界线：探索/梯度/模式/checkpoint | 同一个网络，随机权重撞墙、训练权重 6 步到终点——本事全在权重里 |

### 概念是怎么一层层加上去的

```
01 老虎机   : 动作 a、奖励 r                         —— 探索 vs 利用
02 迷宫     : + 状态 s、状态转移                      —— 完整 MDP，表格 Q-learning
03 倒立摆   : + 状态连续（表格装不下）→ 用神经网络      —— 深度 RL，策略梯度
04 小 GPT   : + "动作"就是生成下一个 token            —— RLHF / GRPO，对齐大模型
05 GRPO骨架 : 不加新概念，把 04 的 GRPO 剥到只剩算法     —— 背它 / 改它 / 做消融实验
06 Agentic : 02的多步 + 05的GRPO 拼起来 + 调用工具     —— 稀疏奖励/信用分配，任务式后训练
07 CodeRL  : + 真的 exec 跑单测当奖励 (Reward System)  —— reward hacking，奖励设计才是胜负手
08 训练vs推理: 横切一刀，把"学会"和"用好"分开看        —— 探索/梯度/checkpoint，所有DL/LLM通用
```

> **04 和 05 怎么选？** 04 是 GRPO 跑在一个真实 tiny GPT 上（看它怎么**用**），
> 05 是把同一个 GRPO 剥到只剩 ~40 行核心（看它**本身**长什么样）。
> 想理解算法先读 05，想看落地先读 04，两个互补。
>
> **06 是整条线的收口**：前面 04/05 的 GRPO 是"回一句话给一个分"（单步）；
> 06 让 agent **多步行动、中途调工具、只有到终点才给分**（稀疏奖励），正是
> Agentic RL（代码/搜索/操作电脑那类"任务式后训练"）的最小内核。最妙的是
> **"用工具"是 agent 自己从奖励里学出来的**（首步就 LOOK 率 25%→100%），没人教。

## 怎么跑

```bash
# 在仓库根目录下
python3 example/01_bandit.py              # 秒级，存 bandit_result.png
python3 example/02_qlearning_gridworld.py # 秒级，打印学到的策略箭头图
python3 example/03_cartpole_reinforce.py  # ~1-2 分钟，存 cartpole_result.png
python3 example/04_grpo_tiny_llm.py       # ~1 分钟，打印训练前后的生成对比
python3 example/05_grpo_minimal.py        # 秒级，纯 CPU，打印 reward 从 0.6 爬到 5.0
python3 example/06_agentic_rl_tool.py     # 秒级，纯 CPU，看 agent 学会"先 LOOK 再走"
python3 example/07_agentic_codeRL.py      # 秒级，纯 CPU，弱奖励 vs 稳健奖励的作弊对照
python3 example/08_train_vs_infer.py      # 秒级，纯 CPU，训练存checkpoint→加载推理→对照未训练
```

> 想要可视化讲解（手机 H5 也适配），用浏览器打开：
> - [`rl_concepts.html`](rl_concepts.html) —— **核心概念速查 + 缩写解释 + 一张关系图把所有概念串起来**
> - [`rl_visualized.html`](rl_visualized.html) —— 四个例子的图解与公式同源对比

## 跑出来大概长这样

- **01**：贪心 ~34% < ε-greedy ~81% < UCB ~86%（选中最优臂的比例）。说明纯利用会被困住，探索很重要。
- **02**：前 100 回合平均奖励约 −17（乱走掉坑），后 100 回合约 +2（学会走最短路），并打印一张箭头策略图。
- **03**：每局坚持步数从 ~40 涨到几百（REINFORCE 方差大，曲线会抖，属正常）。
- **04**：组内平均奖励从 ~0.26 升到 ~0.99，模型从乱序数字学会输出 `0 1 4 5 7 9` 这样的严格递增串。
  文件顶部还记录了**真实踩到的 reward hacking**（奖励被钻空子）和怎么改奖励堵住它——这部分比任务本身更值钱。
- **05**：组内平均奖励从 ~0.62 升到 5.00，greedy 输出收敛到 `[7, 7, 7, 7, 7]`。
  文件**末尾有 3 个魔改实验**（缩小组 → baseline 估不准、去掉 KL → 策略塌缩、换复杂奖励 → 仍能学），
  改一个参数重跑就能看到现象，是把 GRPO 真正"摸熟"的最快方式。
- **06**：到达率 12%→100%、**首步就 LOOK 率 25%→100%**、平均奖励 −0.57→+0.80。
  训练后贪心轨迹会打印成 `LOOK -> LEFT -> LEFT -> LEFT`（宝藏在左）/ `LOOK -> RIGHT -> RIGHT -> RIGHT`（在右）——
  **agent 自己学会了先用工具探路再行动**。文件末尾也有 3 个魔改实验（让 LOOK 失效→工具被无视、
  去掉步代价→变话痨、走廊加长→稀疏奖励变难训），把 Agentic RL 的痛点摸一遍。
- **07**：同一个 agent 训两遍——**弱奖励**（只看可见用例）学到 `return 6` 作弊，隐藏用例通过率 **0%**；
  **稳健奖励**（可见+隐藏）逼出 `return x*2`，隐藏用例通过率 **100%**。一眼看清 reward hacking。
  末尾 3 个魔改实验：过程奖励 vs 结果奖励、给可见用例加点把作弊后门焊死、扩大代码空间让偷分更严重。
- **08**：训练阶段到达率 31%→100% 并存出 `08_policy.pt`；推理阶段加载它贪心走出 `↓↓→→↓→` **6 步到终点**（奖励 +0.75）；
  对照组——**同结构但随机权重**的网络贪心只会 `→↑↑↑↑…` 撞墙打转（奖励 −2.50）。把"训练时探索+算梯度+更新权重"
  和"推理时加载权重+贪心+no_grad"并排标了出来，并点明这和 LLM 的 `训练 / model.generate` 是同一条界线。
  （`08_policy.pt` 是训练产物，已加进 `.gitignore`，可随时删。）

## 和 autoresearch 的关系

autoresearch 让 AI agent 反复"改代码 → 训练5分钟 → 看 val_bpb 降没降 → 留下更好的"。
例子 04/05 的 GRPO 让模型反复"生成答案 → 打分 → 强化高分的"；例子 06 更进一步，让 agent
**多步行动、中途调工具、只按任务成没成给稀疏奖励**——这正是 autoresearch 自己的形态：
它就是一个 **Agentic RL 系统**（agent 在真实环境里多步操作，用 val_bpb 这种可验证指标当奖励）。
几个例子是**同一种循环**的不同尺度：都是**用一个可量化的信号，驱动系统朝更好的方向自我迭代**。
理解了这八个例子，再回头看 autoresearch 的"自主研究循环"，会清楚很多。
