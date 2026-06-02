"""
例子 03 —— CartPole 倒立摆 + 策略梯度 REINFORCE
=================================================
前两个例子用一张"表格"存价值：老虎机是一维表 Q[a]，迷宫是二维表 Q[s,a]。
但真实问题的状态往往是连续的、无穷多的——你没法给每个状态都留一格。

CartPole（小车顶杆子）的状态是 4 个连续实数：
    [小车位置, 小车速度, 杆子角度, 杆子角速度]
这没法做成表格。解决办法：**用一个神经网络来代替表格**——这就是"深度强化学习"。

而且这次我们换一种思路。前面是"学价值（每个动作值多少分），再据此选动作"。
这里直接"**学策略**"：神经网络的输入是状态，输出是"该往左推还是往右推"的概率。
我们直接对这个"动作概率"下手去优化，这类方法叫**策略梯度 (Policy Gradient)**，
最基础的版本就是 **REINFORCE**。

REINFORCE 的核心思想，一句话：
    把一整局玩完，看最后得了多少分（回报 G）。
    如果这局得分高 → 提高这局里做过的那些动作的概率；
    如果这局得分低 → 降低它们的概率。
    "好的行为多做，坏的行为少做"，仅此而已。

写成损失函数（梯度上升 → 取负变梯度下降）：
    loss = - Σ_t  log π(a_t | s_t) * G_t
    其中 G_t = 从 t 时刻往后的折扣总回报（这一步之后还能拿多少分）。
    再减去一个基线 baseline（这里用回报均值），降低方差、让训练更稳。

环境我们**自己实现**（不依赖 gymnasium），就是经典的小车单摆物理方程。
杆子每多坚持一步 +1 分，倒了或小车跑出界就结束。坚持越久得分越高。

运行：
    python3 example/03_cartpole_reinforce.py
会打印训练过程中"每局坚持的步数"如何从几十步涨到几百步，并存曲线图。
"""

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "STHeiti", "Songti SC", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

torch.manual_seed(0)
np.random.seed(0)


class CartPole:
    """自己实现的 CartPole 物理环境（对标经典 gym CartPole-v1）。
    你不用看懂这堆物理公式——把它当成一个黑盒：给动作，它返回新状态、奖励、是否结束。
    这正是"环境"对智能体的样子：只通过 (状态, 奖励) 交互，内部规则智能体并不知道。
    """
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masscart + masspole
    length = 0.5            # 杆子半长
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02              # 每一步代表的物理时间(秒)
    x_threshold = 2.4       # 小车跑出 ±2.4 就算失败
    theta_threshold = 12 * math.pi / 180  # 杆子偏超过 ±12° 就算失败

    def reset(self):
        # 初始状态：四个量都来一点小随机扰动
        self.state = np.random.uniform(-0.05, 0.05, size=4).astype(np.float32)
        return self.state.copy()

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta, sintheta = math.cos(theta), math.sin(theta)
        # 经典倒立摆动力学方程
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        x_acc = temp - self.polemass_length * theta_acc * costheta / self.total_mass
        # 欧拉积分更新状态
        x += self.tau * x_dot
        x_dot += self.tau * x_acc
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_acc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        done = bool(abs(x) > self.x_threshold or abs(theta) > self.theta_threshold)
        reward = 1.0  # 只要还没倒，每活一步就 +1（所以"活得久"= 得分高）
        return self.state.copy(), reward, done


class PolicyNet(nn.Module):
    """策略网络：输入 4 维状态，输出 2 个动作(左/右)的概率。
    这就是用来"代替 Q 表"的那个函数逼近器——只不过它直接输出动作概率(策略)。
    """
    def __init__(self, n_state=4, n_action=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_action),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)  # 转成概率分布，两个动作概率加起来=1


def run_episode(env, policy):
    """玩一整局。每一步都从策略网络给的概率分布里**采样**一个动作
    （采样而非取最大 → 天然带探索）。记录每步的 log π(a|s) 和奖励，回合结束后用于更新。
    """
    state = env.reset()
    log_probs, rewards = [], []
    for _ in range(500):  # CartPole-v1 满分 500 步
        s = torch.from_numpy(state).float()
        probs = policy(s)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()                 # 按概率采样动作
        log_probs.append(dist.log_prob(action))
        state, reward, done = env.step(int(action.item()))
        rewards.append(reward)
        if done:
            break
    return log_probs, rewards


def compute_returns(rewards, gamma=0.99):
    """计算每一步的"折后回报" G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...
    从后往前累加最省事。
    """
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns, dtype=torch.float32)


def main():
    env = CartPole()
    policy = PolicyNet()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

    n_episodes = 600
    lengths = []  # 每局坚持的步数（= 得分），越大越好

    for ep in range(n_episodes):
        log_probs, rewards = run_episode(env, policy)
        lengths.append(len(rewards))

        returns = compute_returns(rewards)
        # 减基线 + 标准化：让"比平均好的动作"得正信号、"比平均差的"得负信号，训练更稳
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # REINFORCE 损失： -Σ log π(a|s) * G   （梯度上升期望回报 → 取负做梯度下降）
        loss = torch.stack([-lp * G for lp, G in zip(log_probs, returns)]).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 50 == 0:
            recent = np.mean(lengths[-50:])
            print(f"  回合 {ep+1:4d}   最近50局平均坚持步数 = {recent:6.1f}")

    # REINFORCE 方差很大，曲线会上下抖动（甚至偶尔"灾难性遗忘"掉回去），这是它出了名的毛病。
    best = max(np.mean(lengths[i:i+50]) for i in range(len(lengths) - 50))
    print(f"\n训练前(前50局)平均坚持 = {np.mean(lengths[:50]):.1f} 步")
    print(f"训练后(后50局)平均坚持 = {np.mean(lengths[-50:]):.1f} 步")
    print(f"训练中最好的50局窗口   = {best:.1f} 步（REINFORCE 方差大、曲线会抖，属正常）")

    # 画学习曲线
    plt.figure(figsize=(8, 4))
    plt.plot(lengths, alpha=0.3, label="每局步数")
    w = 20
    smooth = np.convolve(lengths, np.ones(w) / w, mode="valid")
    plt.plot(range(w - 1, len(lengths)), smooth, label=f"{w}局滑动平均")
    plt.axhline(500, ls="--", c="gray", label="满分 500")
    plt.xlabel("回合")
    plt.ylabel("杆子坚持的步数")
    plt.title("CartPole + REINFORCE 学习曲线（越高越好）")
    plt.legend()
    plt.tight_layout()
    out = "example/cartpole_result.png"
    plt.savefig(out, dpi=110)
    print(f"\n学习曲线已存到 {out}")
    print("\n要点：状态连续、没法做表格，于是用神经网络当'策略'。")
    print("     REINFORCE 用一整局的回报当信号：得分高就强化这局做过的动作。")


if __name__ == "__main__":
    main()
