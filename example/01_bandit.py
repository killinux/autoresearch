"""
例子 01 —— 多臂老虎机 (Multi-Armed Bandit)
===========================================
这是强化学习的 "Hello World"。整个 RL 后面那一大套理论，都可以从这个
最朴素的场景长出来。

场景想象：
    你面前有 K 台老虎机（K 条"摇臂"）。每拉一次，某台机器会吐给你一些钱
    （奖励 reward）。但每台机器吐钱的"平均水平"不一样，而且你事先不知道。
    你只有有限的次数，怎么拉才能赚最多？

这里没有"状态"也没有"状态转移"——每一步的处境都一样，你只需要决定
"这一拉，选哪条臂"。所以它是 RL 最退化、最干净的形态，专门用来讲清楚
强化学习永恒的核心矛盾：

    探索 (Exploration)  vs  利用 (Exploitation)
    ----------------------------------------------
    利用：选我目前认为最赚的那条臂（稳，但可能错过更好的）
    探索：去试试别的臂（可能亏，但能让我把世界看得更清楚）

本文件实现并对比三种策略：
    1) 纯贪心 greedy        —— 永远选当前估计最高的，从不探索（容易被早期运气坑死）
    2) ε-greedy             —— 大部分时间贪心，有 ε 的概率随机乱试一把
    3) UCB (置信上界)        —— 不靠掷骰子探索，而是"乐观地对待没试过的臂"

运行：
    python3 example/01_bandit.py
会打印每种策略的最终表现，并存一张对比图 bandit_result.png。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # 不弹窗，直接存文件（服务器/无界面也能跑）
import matplotlib.pyplot as plt

# 让图里的中文正常显示（Mac 自带这些字体）
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

# 固定随机种子，保证每次运行结果一样，方便对照讲解
RNG = np.random.default_rng(0)


class Bandit:
    """一个 K 臂老虎机。每条臂 i 的真实平均奖励是 q_true[i]（我们假装不知道）。
    拉第 i 条臂时，吐出的钱 = q_true[i] + 一点高斯噪声（所以同一条臂每次也不一样）。
    """

    def __init__(self, k=10):
        self.k = k
        # 每条臂的"真实价值"，从标准正态里随机生成。最好的那条臂事先未知。
        self.q_true = RNG.normal(0.0, 1.0, size=k)
        self.optimal = int(np.argmax(self.q_true))  # 记下哪条臂其实最好（用于评估）

    def pull(self, a):
        """拉第 a 条臂，返回这一次的奖励。"""
        return self.q_true[a] + RNG.normal(0.0, 1.0)


def run_agent(strategy, k=10, steps=1000, epsilon=0.1, c=2.0):
    """让某个策略在一台老虎机上玩 steps 步，记录学习曲线。

    返回：
        rewards   —— 每一步拿到的奖励（长度 steps）
        is_opt    —— 每一步是否选中了"真正最优"那条臂（0/1，长度 steps）

    核心数据结构：
        Q[a]  —— 我们对第 a 条臂价值的"当前估计"（一开始全是 0）
        N[a]  —— 第 a 条臂被拉过几次

    每拉一次就用"增量平均"更新估计：
        Q[a] <- Q[a] + (reward - Q[a]) / N[a]
    这其实就是在算这条臂历史奖励的平均值，但不用存全部历史，省内存。
    括号里的 (reward - Q[a]) 叫"误差"，整条式子是后面所有 RL 更新公式的祖先。
    """
    bandit = Bandit(k)
    Q = np.zeros(k)          # 价值估计
    N = np.zeros(k)          # 计数
    rewards = np.zeros(steps)
    is_opt = np.zeros(steps)

    for t in range(steps):
        # ---- 第一步：根据策略选一条臂 ----
        if strategy == "greedy":
            a = int(np.argmax(Q))                       # 永远选当前估计最高的
        elif strategy == "epsilon":
            if RNG.random() < epsilon:
                a = RNG.integers(k)                     # 以 ε 概率：随机探索
            else:
                a = int(np.argmax(Q))                   # 否则：利用
        elif strategy == "ucb":
            # UCB：给每条臂的估计加一个"不确定性奖金"。试得越少的臂，奖金越大，
            # 于是算法会"乐观地"主动去试冷门臂——一种更聪明的、不靠掷骰子的探索。
            bonus = c * np.sqrt(np.log(t + 1) / (N + 1e-6))
            a = int(np.argmax(Q + bonus))
        else:
            raise ValueError(strategy)

        # ---- 第二步：拉臂，拿到奖励 ----
        r = bandit.pull(a)

        # ---- 第三步：更新对这条臂的估计（增量平均）----
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]

        rewards[t] = r
        is_opt[t] = 1.0 if a == bandit.optimal else 0.0

    return rewards, is_opt


def moving_average(x, w=20):
    """平滑一下曲线，看趋势更清楚。"""
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    K, STEPS, RUNS = 10, 1000, 500
    configs = [
        ("greedy",  "纯贪心 (greedy)"),
        ("epsilon", "ε-greedy (ε=0.1)"),
        ("ucb",     "UCB (c=2)"),
    ]

    # 重点：单跑一台老虎机噪声太大（可能某策略碰巧走运）。所以我们在 RUNS 台
    # 各不相同的老虎机上各跑一遍，再把曲线平均——这才是公平、稳定的对比
    #（经典 Sutton & Barto 教材的标准做法）。
    plt.figure(figsize=(11, 4))
    print(f"{K} 臂老虎机，每种策略在 {RUNS} 台不同机器上各玩 {STEPS} 步后取平均：\n")
    for strat, label in configs:
        all_rewards = np.zeros((RUNS, STEPS))
        all_opt = np.zeros((RUNS, STEPS))
        for run in range(RUNS):
            r, o = run_agent(strat, k=K, steps=STEPS)
            all_rewards[run] = r
            all_opt[run] = o
        rewards = all_rewards.mean(axis=0)   # 沿"台"维度平均，得到每一步的平均表现
        is_opt = all_opt.mean(axis=0)
        print(f"  {label:<22} 末段平均奖励 = {rewards[-100:].mean():+.3f}   "
              f"末段选中最优臂比例 = {is_opt[-100:].mean()*100:5.1f}%")

        plt.subplot(1, 2, 1)
        plt.plot(moving_average(rewards), label=label)
        plt.subplot(1, 2, 2)
        plt.plot(moving_average(is_opt) * 100, label=label)

    plt.subplot(1, 2, 1)
    plt.title("每步平均奖励（越高越好）")
    plt.xlabel("步数")
    plt.ylabel("奖励")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("选中最优臂的比例 %（越高越好）")
    plt.xlabel("步数")
    plt.ylabel("%")
    plt.legend()
    plt.tight_layout()
    out = "example/bandit_result.png"
    plt.savefig(out, dpi=110)
    print(f"\n对比图已存到 {out}")
    print("\n要点：纯贪心常被早期运气困住、收敛到次优臂；ε-greedy 靠随机探索逃出来；")
    print("     UCB 靠'乐观对待没试过的臂'，通常最快锁定最优臂。")


if __name__ == "__main__":
    main()
