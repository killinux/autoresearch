"""
例子 04 —— 在"小 GPT"上做 GRPO（语言模型的强化学习）
======================================================
这是把前三个例子的思想搬到**语言模型**上，也正是 autoresearch / 大模型后训练
(post-training) 里真正在用的东西。ChatGPT、DeepSeek 这类模型"对齐人类偏好"
那一步，用的就是这一类算法。GRPO (Group Relative Policy Optimization) 是
DeepSeek 提出、如今很流行的一种，特点是**不需要额外训练一个价值网络**，
省事又稳。

和例子 03 的关系：
    REINFORCE：玩一局 → 看回报 → 好就强化。
    GRPO：对**同一个题目**，让模型生成**一组 (group)** G 个答案 → 每个打分 →
          谁比这组的平均分高，就强化谁；谁比平均低，就抑制谁。
    关键创新就在"组相对"：用"这一组答案的平均分"当基线 (baseline)，
    省掉了额外的价值网络，advantage 直接在组内算：

        advantage_i = (reward_i - mean(组内奖励)) / std(组内奖励)

    然后还是策略梯度那一套：loss = - Σ_i advantage_i * logπ(答案_i)

我们这里搭一个**真正的迷你 GPT**（带 self-attention 的 transformer，就一层），
词表是数字 0-9。任务是一个"可自动判分"的玩具任务（RL 不需要标准答案，只需要
一个能打分的奖励函数）：

    任务：生成一串数字，要求**严格递增**（如 0 1 2 3 4 5 6 7）。
    奖励：从开头起"能连续严格递增多长"，占总长度的比例（详见 reward_fn）。
          全程乱猜大约 0.2 分，学好后接近 1.0。

你会看到：训练前模型吐的是乱序数字；GRPO 几百步后，它学会吐**一路递增**的数字——
完全没人告诉它"正确答案"，它只是被一个打分函数推着，自己摸索出了规律。
这正是 RLHF / GRPO 的精髓：**用奖励信号塑造语言模型的输出**。

⚠️ 一个真实世界的大坑（reward hacking / 奖励钻空子）—— 这是写这个例子时真实踩到的：
    第1版奖励 = "非递减(左<=右)的比例"。模型发现馊主意：输出"4 4 4 4 …"全一样！
              每个相邻对都满足"<="，轻松满分，但根本没学会排序。
    第2版奖励 = "严格递增(左<右)的比例"。堵住了全相同，但又冒出新捷径：输出
              "1 2 3 8 1 2 3 8"循环，只在断点丢一分，拿 0.86 高分照样不会真排序。
    第3版奖励(本文件) = "从开头连续递增的长度"。断一次就大幅扣分，捷径彻底失效，
              模型只能老老实实从头排到尾。
    这就是著名的 reward hacking——你以为在教A，奖励却被某条捷径B骗走了，是 RLHF
    最头疼的问题之一。教训比任务本身更值钱：**RL 的难点常常不在算法，而在奖励怎么定。**

运行：
    python3 example/04_grpo_tiny_llm.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---- 超参数（都很小，CPU/MPS 几十秒就能跑）----
VOCAB = 11          # 数字 0-9 共 10 个，再加一个特殊起始符 BOS=10
BOS = 10
SEQ_LEN = 6         # 每次生成 6 个数字（0..9 里凑 6 个严格递增很宽裕，能稳定到满分）
DIM = 48            # 词向量/隐藏维度
N_HEAD = 3
GROUP = 64          # GRPO 每步对同一题生成多少个答案（组大小 G）
STEPS = 500


class Block(nn.Module):
    """一个标准 Transformer 块：因果自注意力 + 前馈网络（和大模型里的一模一样，只是小）。"""
    def __init__(self, dim, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        T = x.size(1)
        # 因果mask：第 t 个位置只能看到 <=t 的位置（语言模型不能"偷看未来"）
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """一个最小可用的 GPT：词嵌入 + 位置嵌入 + 一个 Transformer 块 + 输出头。
    输入一串 token，输出每个位置"下一个 token"的预测分布。这就是"策略 π"。
    """
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, DIM)
        self.pos = nn.Embedding(SEQ_LEN + 1, DIM)
        self.block = Block(DIM, N_HEAD)
        self.ln = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB)

    def forward(self, idx):
        T = idx.size(1)
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)
        x = self.block(x)
        return self.head(self.ln(x))   # (B, T, VOCAB) 的 logits


@torch.no_grad()
def generate(model, n):
    """从 BOS 开始，自回归地采样生成 n 条、每条 SEQ_LEN 个数字的序列。
    采样(而非取最大)→ 同一个模型能生成不同答案，这正是 GRPO 需要的"一组多样答案"。
    """
    idx = torch.full((n, 1), BOS, dtype=torch.long, device=device)  # 都以 BOS 开头
    for _ in range(SEQ_LEN):
        logits = model(idx)[:, -1, :]          # 看最后一个位置，预测下一个数字
        logits[:, BOS] = float("-inf")         # 屏蔽 BOS：生成内容里只允许出现数字 0-9
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)      # 按概率采样
        idx = torch.cat([idx, nxt], dim=1)
    return idx                                 # (n, 1+SEQ_LEN)，第0列是BOS


def seq_logprob(model, idx):
    """给定完整序列(含BOS)，算两样东西(都带梯度，用于反传)：
      1) 每条序列生成"其数字部分"的总 log 概率 → 用于策略梯度
      2) 每个位置分布的平均熵 → 用作熵奖励，防止过早收敛(见下方说明)
    做法：把 idx[:, :-1] 喂进去预测 idx[:, 1:]，逐位置取对应 token 的 log 概率。
    """
    logits = model(idx[:, :-1])                # (n, SEQ_LEN, VOCAB)
    logp = F.log_softmax(logits, dim=-1)
    targets = idx[:, 1:].unsqueeze(-1)         # 真正生成出来的那些数字
    token_logp = logp.gather(-1, targets).squeeze(-1)   # 每个位置的 log π(token)
    entropy = -(logp.exp() * logp).sum(-1).mean()       # 所有位置分布熵的均值
    return token_logp.sum(dim=1), entropy      # (n,) 与 标量


def reward_fn(idx):
    """奖励函数：从开头起"能连续严格递增多长"，占总长度的比例。
    例(长度6): 0 1 4 5 7 9 → 一路递增到底，(1+5)/6 = 满分 1.0
              1 2 3 9 0 4 → 到第4位 9→0 断了，只算前4个，(1+3)/6 ≈ 0.67
              4 4 4 4 4 4 → 一开始 4>4 就不算严格递增，长度1，(1+0)/6 ≈ 0.17
    为什么用这个、而不是更直觉的"相邻递增对的比例"？因为后者有大坑(见文件顶部说明)：
    模型会钻空子——靠"重复一个短递增块"只丢一分就拿 0.86 高分，却没真学会排序。
    这条"连续递增长度"的奖励把那个捷径堵死了，逼模型老老实实从头排到尾。
    RL 的关键——我们不提供标准答案，只提供这样一个"能给任意输出打分"的函数。
    """
    digits = idx[:, 1:].float()                # 去掉BOS，取生成的数字 (n, SEQ_LEN)
    step_ok = (digits[:, 1:] > digits[:, :-1]).float()   # 每个相邻对是否"严格递增"
    # cumprod：一旦某步断了(0)，后面全变0 → sum 得到"从头连续满足的步数"
    leading = torch.cumprod(step_ok, dim=1).sum(dim=1)   # (n,) ∈ [0, SEQ_LEN-1]
    return (1.0 + leading) / SEQ_LEN           # +1 是因为第一个数字白送，归一化到 (0,1]


def show_samples(model, tag):
    idx = generate(model, 6)
    r = reward_fn(idx)
    print(f"\n  [{tag}] 随机抽 6 条生成结果（括号是该条得分）：")
    for i in range(6):
        digits = " ".join(str(int(d)) for d in idx[i, 1:])
        print(f"      {digits}    ({r[i]:.2f})")


def main():
    model = TinyGPT().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print(f"迷你 GPT：{sum(p.numel() for p in model.parameters())/1e3:.1f}K 参数，设备={device}")
    print("任务：生成 6 个数字，要求严格递增（如 0 1 4 5 7 9）。")
    print("奖励：从开头连续递增的长度占比。没有标准答案，全靠奖励信号自己学。")

    show_samples(model, "训练前")

    history = []
    for step in range(STEPS):
        # ---- 1) 对"同一个题"(都从BOS开始)生成一组 G 个答案 ----
        idx = generate(model, GROUP)
        rewards = reward_fn(idx)               # (G,) 每个答案的得分

        # ---- 2) GRPO 核心：组内相对优势(不需要价值网络!) ----
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # ---- 3) 策略梯度更新：把"比平均好的答案"的概率推高，差的推低 ----
        logp, entropy = seq_logprob(model, idx)         # (G,) 带梯度
        # 熵奖励：减去 beta*熵 = 鼓励分布别太尖，保持探索，避免过早塌到局部最优。
        # 这是 PPO/A3C 等真实 RL 算法的标配。beta 不能太大，否则学不专一。
        beta = 0.02
        loss = -(adv.detach() * logp).mean() - beta * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history.append(rewards.mean().item())
        if (step + 1) % 50 == 0:
            print(f"  step {step+1:4d}   组内平均奖励 = {rewards.mean().item():.3f}")

    show_samples(model, "训练后")

    print(f"\n训练前期(前20步)平均奖励 = {sum(history[:20])/20:.3f}  （≈乱猜，很低）")
    print(f"训练后期(后20步)平均奖励 = {sum(history[-20:])/20:.3f}  （学会从头排到尾，趋近1.0）")
    print("\n要点：没有任何'正确答案'标注，模型仅凭一个打分函数，被 GRPO 一步步推着")
    print("     学会了'输出要严格递增'这条规则。这就是 RLHF / 大模型后训练的内核。")
    print("     和 autoresearch 的联系：那边是 agent 调代码降 val_bpb，这里是 RL 调")
    print("     模型参数升奖励——都是'试错→打分→保留更好的'这同一种循环。")


if __name__ == "__main__":
    main()
