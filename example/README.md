# 强化学习四连例（RL by Example）

一条**由浅入深**的强化学习学习线，配套 [autoresearch](../README.md) 项目。
四个例子层层递进，每一个都在上一个的基础上**多引入一个新概念**，最后落到
大模型后训练（RLHF / GRPO）——也就是 autoresearch 这类项目的同一个内核：
**试错 → 打分 → 保留更好的**。

所有代码都带**详细中文注释**，在 macOS / Apple Silicon 上用系统 `python3` 就能直接跑
（依赖 numpy / torch(MPS) / matplotlib，均已具备；**不需要** gymnasium，CartPole 物理引擎是自己实现的）。

## 四个例子

| # | 文件 | 学到的新东西 | 一句话 |
|---|------|------------|--------|
| 01 | [`01_bandit.py`](01_bandit.py) | 探索 vs 利用、ε-greedy、UCB | 只有动作和奖励的最简 RL：该选哪台老虎机 |
| 02 | [`02_qlearning_gridworld.py`](02_qlearning_gridworld.py) | **状态**、Q 表、TD 更新、γ/α | 走迷宫，学会"在每个格子该往哪走" |
| 03 | [`03_cartpole_reinforce.py`](03_cartpole_reinforce.py) | **神经网络代替表格**、策略梯度 | 倒立摆，用小网络直接学"策略" |
| 04 | [`04_grpo_tiny_llm.py`](04_grpo_tiny_llm.py) | **在语言模型上做 RL**、GRPO 组相对优势、reward hacking | 用奖励信号教小 GPT 学会排序 |

### 概念是怎么一层层加上去的

```
01 老虎机   : 动作 a、奖励 r                         —— 探索 vs 利用
02 迷宫     : + 状态 s、状态转移                      —— 完整 MDP，表格 Q-learning
03 倒立摆   : + 状态连续（表格装不下）→ 用神经网络      —— 深度 RL，策略梯度
04 小 GPT   : + "动作"就是生成下一个 token            —— RLHF / GRPO，对齐大模型
```

## 怎么跑

```bash
# 在仓库根目录下
python3 example/01_bandit.py              # 秒级，存 bandit_result.png
python3 example/02_qlearning_gridworld.py # 秒级，打印学到的策略箭头图
python3 example/03_cartpole_reinforce.py  # ~1-2 分钟，存 cartpole_result.png
python3 example/04_grpo_tiny_llm.py       # ~1 分钟，打印训练前后的生成对比
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

## 和 autoresearch 的关系

autoresearch 让 AI agent 反复"改代码 → 训练5分钟 → 看 val_bpb 降没降 → 留下更好的"。
例子 04 的 GRPO 让模型反复"生成答案 → 打分 → 强化高分的"。
两者是**同一种循环**的不同尺度：一个在调代码、一个在调参数，但都是
**用一个可量化的信号，驱动系统朝更好的方向自我迭代**。理解了这四个例子，
再回头看 autoresearch 的"自主研究循环"，会清楚很多。
