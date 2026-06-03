"""
第 5 课 · GRPO 最小骨架 (Group Relative Policy Optimization)
================================================================
和第 4 课的关系:
  04 = GRPO 跑在一个真实 tiny GPT 上 (含 tokenizer/transformer/采样, 代码多)。
  05 = 把 GRPO 从 LLM 里"剥"出来, 压成 ~40 行只剩算法本身 ——
       便于背诵、魔改、做受控消融实验 (这就是 autoresearch 式的研究技能)。
  没有引入新概念, 引入的是"把算法从应用里剥离出来做实验"的能力。

任务: 模型生成长度 5 的 token 串 (每位 0~9),
      reward = 串中等于目标数字 7 的个数。
      最优策略 = 输出 [7,7,7,7,7], reward=5。
  (这里没有 Transformer, 策略就是一张 [位置, token] 的 logits 表;
   但 GRPO 的算法逻辑和真实 LLM 上完全一样。)

GRPO 三步核心:
  1. 对同一 prompt 采样一组 G 个回答 (group sampling)
  2. 用组内 reward 的 (均值,标准差) 归一化优势 —— 替代 PPO 的 value 网络
  3. PPO 式 clip 目标 + 对参考策略的 KL 惩罚

精髓只有一行 (见下方 adv = ...):
  PPO 要训一个 value 网络估 baseline; GRPO 直接用"同组其他样本的平均分"当 baseline。

文件末尾有 3 个魔改实验, 改一个参数就能看到行为变化, 强烈建议动手试。

运行: python3 example/05_grpo_minimal.py   (秒级, 纯 CPU)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

VOCAB = 10        # token 0~9
SEQ_LEN = 5       # 每个回答的长度
TARGET = 7        # reward 目标数字
GROUP = 8         # 每个 prompt 采样多少个回答 (GRPO 的 G)
LR = 0.05
CLIP = 0.2        # PPO clip 系数 epsilon
KL_COEF = 0.02    # KL 惩罚强度
STEPS = 200


class Policy(nn.Module):
    """极简策略网络: 没有输入(prompt固定), 直接学 SEQ_LEN 个位置上的 token 分布。
    真实 LLM 里这里是一个 Transformer, 但 GRPO 的算法逻辑完全一样。"""
    def __init__(self):
        super().__init__()
        # logits[pos, token]: 每个位置独立一个分布
        self.logits = nn.Parameter(torch.zeros(SEQ_LEN, VOCAB))

    def dist(self):
        return torch.distributions.Categorical(logits=self.logits)


def reward_fn(seqs):
    """对每个回答打分: 等于 TARGET 的 token 个数。seqs: [B, SEQ_LEN]"""
    return (seqs == TARGET).float().sum(dim=-1)   # [B]


def seq_logprob(policy, seqs):
    """整条序列的 log p(seq) = 各位置 logprob 求和。seqs: [B, SEQ_LEN] -> [B]"""
    dist = policy.dist()                          # 在 SEQ_LEN 个位置上的分布
    lp = dist.log_prob(seqs)                      # [B, SEQ_LEN]
    return lp.sum(dim=-1)                         # [B]


policy = Policy()
# 参考策略 (reference): GRPO/RLHF 都会拿它做 KL 锚点, 防止跑偏。冻结不更新。
ref_policy = Policy()
ref_policy.load_state_dict(policy.state_dict())
for p in ref_policy.parameters():
    p.requires_grad_(False)

opt = torch.optim.Adam(policy.parameters(), lr=LR)

for step in range(STEPS):
    # ---- 1. Group sampling: 采样 G 个回答 ----
    with torch.no_grad():
        dist = policy.dist()
        seqs = dist.sample((GROUP,))              # [G, SEQ_LEN]
        rewards = reward_fn(seqs)                 # [G]
        old_logp = seq_logprob(policy, seqs)      # [G] 采样时的旧策略 logprob (PPO 的 old)

    # ---- 2. 组内归一化优势 (GRPO 精髓, 无需 value 网络) ----
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)   # [G]

    # ---- 3. PPO clip 目标 + KL 惩罚 ----
    new_logp = seq_logprob(policy, seqs)          # 当前策略对同样回答的 logprob
    ratio = torch.exp(new_logp - old_logp)        # 重要性采样比 r = π_new/π_old

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv
    policy_loss = -torch.min(unclipped, clipped).mean()   # 取悲观的那个, 负号=最大化

    # KL(π_new || π_ref): 每个位置分布的 KL 求和, 拉住别离参考策略太远
    kl = torch.distributions.kl_divergence(policy.dist(), ref_policy.dist()).sum()

    loss = policy_loss + KL_COEF * kl

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 20 == 0 or step == STEPS - 1:
        greedy = policy.logits.argmax(dim=-1).tolist()    # 当前最可能输出
        print(f"step {step:3d} | avg_reward {rewards.mean():.2f} "
              f"| kl {kl.item():.3f} | greedy {greedy}")

print("\n训练完成。最优应收敛到 [7, 7, 7, 7, 7], reward=5")

# =============================================================================
# 三个魔改实验 (改完重跑 python3 example/05_grpo_minimal.py 看变化)
# =============================================================================
# 实验 1 · 组太小, baseline 估不准
#   把上面的 GROUP = 8 改成 GROUP = 2。
#   现象: 收敛更慢、avg_reward 抖得更厉害。
#   原因: 优势 (r - mean) / std 靠"同组样本"估 baseline, 组越小估计越噪。
#         这正是 GRPO 论文里 G 要取够大 (常见 8~64) 的原因。
#
# 实验 2 · 去掉 KL 锚点, 看策略塌缩
#   把上面的 KL_COEF = 0.02 改成 KL_COEF = 0.0。
#   现象: kl 一路飙升, 模型很快变得"过度自信"(分布塌成一个尖峰)。
#   原因: 没有参考策略拉住, 策略可以无限远离起点 —— 真实 RLHF 里这会让模型
#         为了刷高 reward 说出怪话 (失去多样性/语言能力)。KL 是安全带。
#
# 实验 3 · 换更复杂的奖励, 看它能不能学
#   把 reward_fn 换成"奇数位要 7、偶数位要 3":
#       def reward_fn(seqs):
#           want = torch.tensor([7, 3, 7, 3, 7])          # 目标模式
#           return (seqs == want).float().sum(dim=-1)     # 命中几位
#   现象: 模型应收敛到 greedy [7, 3, 7, 3, 7], reward=5。
#   说明: 只要奖励可量化, GRPO 就能把策略推向它 —— 这也是 RLHF 的全部前提:
#         "做对一个能给分的奖励函数" 往往比训练本身更难 (参见第 4 课的 reward hacking)。
