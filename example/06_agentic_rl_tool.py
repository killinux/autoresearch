"""
第 6 课 · Agentic RL 最小例子 (多步行动 + 调用工具 + 稀疏奖励 + GRPO)
================================================================
把前面两课拼起来:
  02 迷宫  -> 多步决策、有状态、走很多步才到终点
  05 GRPO  -> 组采样 + 组归一化优势 (无 value 网络) 的更新方式
  06 = 让一个小策略网络当 agent, 在"黑暗走廊"里多步行动,
       自己学会先调用 LOOK 工具探明方向, 再走到终点。

这就是 Agentic RL 的内核: LLM/策略不再只回一句话, 而是
  思考 -> 调工具(看一眼) -> 看返回 -> 再行动 -> ... -> 交付,
最后只按"任务成没成"给一个稀疏奖励。

任务: 走廊有 7 格 (0~6)。宝藏在最左(0)或最右(6), 每局随机, agent 事先不知道。
      agent 从正中(3)出发。它看不到宝藏在哪边, 除非调用 LOOK 工具。
  动作 = {LEFT, RIGHT, LOOK}
    LOOK = 工具调用: 把"宝藏在左/右"写进自己的观测(相当于给 agent 一格记忆),
           不移动, 花 1 步代价。
  奖励 = 到达宝藏 +1; 每走一步 -0.05 (鼓励尽快、别瞎逛、别乱看)。
  稀疏: 中间每步几乎只有 -0.05, 只有最后到终点才有那 +1。

要看的现象 (Agentic RL 三个关键点):
  1. 多步轨迹: 一局是好几步, 不是单步。
  2. 涌现的工具使用: 没人教 LOOK 有用, 但不看就不知方向 -> 训练后 agent 学会"先看后走"。
  3. 信用分配: 终点那 +1 怎么分给前面每一步? 这里用最简方案 ——
     整条轨迹共享同一个"组相对优势"(GRPO), 让组间对比自动把功劳排出来。

运行: python3 example/06_agentic_rl_tool.py   (秒级, 纯 CPU)
"""
import random
import torch
import torch.nn as nn

torch.manual_seed(0)
random.seed(0)

N = 7              # 走廊格子数 0~6
START = 3         # 出发点(正中)
MAX_STEPS = 15    # 一局最多走几步(走不到就结束)
STEP_COST = -0.05 # 每步代价
GOAL_REWARD = 1.0 # 到终点奖励

GROUP = 16        # GRPO 每次更新采样多少条轨迹 (组大小 G)
INNER_EPOCHS = 4  # 每批轨迹反复优化几遍 (让 PPO 的 clip 真正起作用)
CLIP = 0.2
LR = 0.01
UPDATES = 300

ACTIONS = ["LEFT", "RIGHT", "LOOK"]   # 0,1,2


class Corridor:
    """黑暗走廊环境。obs = [位置 one-hot(7) | 已探知方向 one-hot(3: 未知/左/右)] = 10 维。
    宝藏方向默认未知, 只有调用 LOOK 才会写进 obs —— 这让 LOOK 工具'真的有用'。"""
    def reset(self, goal_side):
        self.goal_side = goal_side                # 0=左(0号格), 1=右(6号格)
        self.goal_cell = 0 if goal_side == 0 else N - 1
        self.pos = START
        self.sensed = 0                           # 0未知 / 1左 / 2右
        self.steps = 0
        return self._obs()

    def _obs(self):
        o = torch.zeros(N + 3)
        o[self.pos] = 1.0                         # 位置
        o[N + self.sensed] = 1.0                  # 已探知方向(LOOK 写入的'记忆格')
        return o

    def step(self, a):
        self.steps += 1
        reward = STEP_COST
        if a == 2:                                # LOOK 工具: 探明方向, 不移动
            self.sensed = 1 if self.goal_side == 0 else 2
        elif a == 0:                              # LEFT
            self.pos = max(0, self.pos - 1)
        elif a == 1:                              # RIGHT
            self.pos = min(N - 1, self.pos + 1)
        done = False
        if self.pos == self.goal_cell:            # 到宝藏
            reward += GOAL_REWARD
            done = True
        elif self.steps >= MAX_STEPS:
            done = True
        return self._obs(), reward, done


class Policy(nn.Module):
    """小 MLP: 看 10 维观测 -> 3 个动作的 logits。真实 agent 这里是个 LLM, 逻辑一样。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N + 3, 32), nn.Tanh(), nn.Linear(32, 3))

    def forward(self, x):
        return self.net(x)


def rollout(policy, goal_side, greedy=False):
    """跑一整局(多步), 返回 (每步记录, 总奖励, 是否到达, 是否用过LOOK, 动作轨迹)。"""
    env = Corridor()
    obs = env.reset(goal_side)
    steps, total_r, reached, looked, trace = [], 0.0, False, False, []
    done = False
    while not done:
        logits = policy(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax() if greedy else dist.sample()
        logp = dist.log_prob(a).detach()
        if a.item() == 2:
            looked = True
        trace.append(ACTIONS[a.item()])
        steps.append((obs, a.detach(), logp))     # 存 obs/动作/旧logprob, 更新时复用(GRPO/PPO 要)
        obs, r, done = env.step(a.item())
        total_r += r
        if r > 0:                                 # 这一步拿到 +1 => 到了终点
            reached = True
    return steps, total_r, reached, looked, trace


policy = Policy()
opt = torch.optim.Adam(policy.parameters(), lr=LR)

for update in range(UPDATES):
    # ---- 1. Group rollout: 采样 G 条多步轨迹(目标方向每局随机) ----
    batch, rewards, n_reach, n_look1 = [], [], 0, 0
    for _ in range(GROUP):
        gs = random.randint(0, 1)
        steps, R, reached, looked, trace = rollout(policy, gs)
        batch.append(steps)
        rewards.append(R)
        n_reach += reached
        n_look1 += (trace[0] == "LOOK")          # 第一步就 LOOK? (真正的"先探后走"信号)
    rewards = torch.tensor(rewards)

    # ---- 2. GRPO 组归一化优势: 一条轨迹一个优势, 再广播给它的每一步 ----
    #    这就是"信用分配"的最简做法: 不去算哪一步立功, 整条轨迹共享 (r-mean)/std,
    #    靠组间对比让好轨迹整体加强、坏轨迹整体压低。
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    # 把所有轨迹的所有步摊平成一个大 batch
    obs_all = torch.stack([s[0] for traj in batch for s in traj])
    act_all = torch.stack([s[1] for traj in batch for s in traj])
    oldlp_all = torch.stack([s[2] for traj in batch for s in traj])
    adv_all = torch.cat([adv[i].expand(len(traj)) for i, traj in enumerate(batch)])

    # ---- 3. PPO clip 更新(和第5课同一套), 反复几遍让 clip 生效 ----
    for _ in range(INNER_EPOCHS):
        dist = torch.distributions.Categorical(logits=policy(obs_all))
        new_lp = dist.log_prob(act_all)
        ratio = torch.exp(new_lp - oldlp_all)
        unclipped = ratio * adv_all
        clipped = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * adv_all
        loss = -torch.min(unclipped, clipped).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    # 注意: 这里没有第5课的 KL 惩罚。因为参考策略=随机初始化, 把它往随机拉只会更差;
    #       KL 锚点是 LLM 后训练(ref=好的SFT模型)才需要的安全带, 从零训的 RL 通常不用。

    if update % 30 == 0 or update == UPDATES - 1:
        print(f"update {update:3d} | avg_reward {rewards.mean():+.2f} "
              f"| 到达率 {n_reach/GROUP:.0%} | 首步就LOOK率 {n_look1/GROUP:.0%}")

# ---- 训练后: 看它学到的策略 (贪心走两局, 目标分别在左/右) ----
print("\n训练完成。看 agent 在两种目标下怎么走 (贪心):")
for gs, name in [(0, "宝藏在 [左/0号格]"), (1, "宝藏在 [右/6号格]")]:
    _, R, reached, looked, trace = rollout(policy, gs, greedy=True)
    print(f"  {name}: {' -> '.join(trace)}  | 奖励 {R:+.2f} | {'到达✓' if reached else '没到✗'}")
print("\n理想行为: 先 LOOK 探明方向, 再一路朝正确方向走到底 —— 工具使用是它自己学出来的。")

# =============================================================================
# 三个魔改实验 (改完重跑看变化)
# =============================================================================
# 实验 1 · 把 LOOK 工具"拿掉信息", 看工具使用消失
#   在 Corridor.step 里把 LOOK 那行改成什么都不做 (self.sensed 永远 0)。
#   现象: LOOK 变得毫无用处, 训练后"用LOOK率"掉到 ~0, agent 只能盲猜方向, 到达率卡在 ~50%。
#   结论: agent 用不用工具, 完全由"工具能不能帮它拿到奖励"决定 —— 没用的工具会被学会无视。
#
# 实验 2 · 去掉每步代价, 看它变"话痨"
#   把 STEP_COST = -0.05 改成 0.0。
#   现象: agent 可能反复 LOOK 或绕路也无所谓(反正不扣分), 轨迹变长、不再追求最短。
#   结论: 奖励设计决定行为。想要"少调工具、走直线", 就得让多余动作有代价
#         (对应真实 agent: 调用 API/搜索都要花钱花时间, 必须计入奖励)。
#
# 实验 3 · 走廊加长, 看稀疏奖励变难
#   把 N = 7 改成 N = 15 (并把 MAX_STEPS 调到 30)。
#   现象: 距离变远, 随机探索更难偶然撞到终点, 收敛明显变慢甚至学不动。
#   结论: 这就是 Agentic RL 的核心痛点 —— 轨迹越长、奖励越稀, 越难训。
#         真实系统会加: 过程奖励(每步给分)、更大的组 G、或课程学习(由易到难)。
