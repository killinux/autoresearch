"""
第 8 课 · 训练 vs 推理 (Train vs Inference) —— RL 里最该分清的一条界线
================================================================
前 7 课你一直在"训练", 但没把【训练】和【推理】单独拆开看。这一课专门讲这条界线,
它对所有深度学习/LLM 都成立:

  训练 (train)  : 探索、可以犯错、算梯度、更新权重   —— 目的是"学会"
  推理 (infer)  : 不探索、不算梯度、不更新、直接用   —— 目的是"用好"
  桥梁: checkpoint(权重文件) —— 训练把学到的东西存进它, 推理把它加载回来

训练 vs 推理 的 5 个具体差别 (本课逐条在代码里标出来):
  ┌────────┬───────────────────────┬───────────────────────┐
  │        │ 训练 train             │ 推理 infer             │
  ├────────┼───────────────────────┼───────────────────────┤
  │ 选动作 │ sample 采样(随机探索)  │ argmax 贪心(确定/最优) │
  │ 梯度   │ backward + 更新权重    │ torch.no_grad 不更新   │
  │ 模式   │ model.train()          │ model.eval()           │
  │ 目的   │ 学会(允许走错)         │ 发挥最好(别走错)       │
  │ 产物   │ 产出 checkpoint        │ 加载 checkpoint        │
  └────────┴───────────────────────┴───────────────────────┘

对到 LLM (你 04-07 那条线):
  训练 = GRPO 反复更新权重;  推理 = model.generate (no_grad、低温/贪心、权重冻结);
  checkpoint = HuggingFace 上那堆模型权重文件。完全是同一回事。

任务: 4x4 迷宫, 从左上(0,0)走到右下(3,3); 每步 -0.05, 到终点 +1。
      策略是个小神经网络: 看"当前在哪格"-> 输出 4 个方向的概率。

运行: python3 example/08_train_vs_infer.py   (秒级, 纯 CPU)
"""
import os
import torch
import torch.nn as nn

torch.manual_seed(0)

SIZE = 4                      # 4x4 迷宫
N_STATES = SIZE * SIZE        # 16 个格子
START, GOAL = 0, N_STATES - 1
GAMMA = 0.95
ACTIONS = ["↑", "↓", "←", "→"]
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "08_policy.pt")


def step(s, a):
    """环境: 在格子 s 执行动作 a -> (新格子, 奖励, 是否到终点)。撞墙就原地不动。"""
    r, c = divmod(s, SIZE)
    if a == 0: r -= 1
    elif a == 1: r += 1
    elif a == 2: c -= 1
    elif a == 3: c += 1
    r = min(max(r, 0), SIZE - 1)          # 撞墙夹住
    c = min(max(c, 0), SIZE - 1)
    ns = r * SIZE + c
    if ns == GOAL:
        return ns, 1.0, True              # 到终点
    return ns, -0.05, False               # 每走一步小代价


def onehot(s):
    v = torch.zeros(N_STATES)
    v[s] = 1.0
    return v


class Policy(nn.Module):
    """策略网络: 16维状态 one-hot -> 4个动作 logits。学到的"本事"全在这些权重里。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_STATES, 64), nn.Tanh(), nn.Linear(64, 4))

    def forward(self, x):
        return self.net(x)


def run_episode(policy, greedy, max_steps=50):
    """跑一局。greedy=False 采样(训练探索), greedy=True 取最优(推理)。
    返回 (每步logprob, 每步reward, 动作轨迹, 是否到达)。"""
    s, done, logps, rews, trace = START, False, [], [], []
    for _ in range(max_steps):
        logits = policy(onehot(s))
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax() if greedy else dist.sample()
        logps.append(dist.log_prob(a))
        s, r, done = step(s, a.item())
        rews.append(r)
        trace.append(ACTIONS[a.item()])
        if done:
            break
    return logps, rews, trace, done


# ============================================================
# 阶段 A · 训练 (TRAIN): 探索 + 算梯度 + 更新权重 + 产出 checkpoint
# ============================================================
def train(updates=150, batch=16, lr=0.01):
    policy = Policy()
    policy.train()                                  # ★差别3: 训练模式
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    print("【阶段A · 训练】探索中, 权重在更新...")
    for u in range(updates):
        all_logps, all_returns, n_reach = [], [], 0
        for _ in range(batch):                      # 收集一批轨迹
            logps, rews, _, done = run_episode(policy, greedy=False)  # ★差别1: 采样探索
            n_reach += done
            # 算 returns-to-go (从每一步往后的折扣回报)
            G, returns = 0.0, []
            for r in reversed(rews):
                G = r + GAMMA * G
                returns.insert(0, G)
            all_logps += logps
            all_returns += returns
        logps_t = torch.stack(all_logps)
        ret_t = torch.tensor(all_returns)
        ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)   # 标准化做baseline,降方差
        loss = -(logps_t * ret_t).mean()            # REINFORCE: 回报高的动作 -> 提高其概率
        opt.zero_grad()
        loss.backward()                             # ★差别2: 算梯度
        opt.step()                                  #          更新权重
        if u % 30 == 0 or u == updates - 1:
            print(f"  update {u:3d} | 到达率 {n_reach/batch:.0%}")
    # ★差别5: 训练产出 checkpoint(把学到的本事存成权重文件)
    torch.save(policy.state_dict(), CKPT)
    print(f"  训练完成, 已保存 checkpoint -> {os.path.basename(CKPT)}\n")


# ============================================================
# 阶段 B · 推理 (INFERENCE): 加载权重, 不探索/不更新, 直接用
# ============================================================
def infer():
    print("【阶段B · 推理】加载 checkpoint, 贪心走一遍, 不再学习...")
    # ★差别5: 推理加载 checkpoint(全新网络 + 灌入训练好的权重)
    policy = Policy()
    policy.load_state_dict(torch.load(CKPT))
    policy.eval()                                   # ★差别3: 推理模式
    with torch.no_grad():                           # ★差别2: 不算梯度、不更新
        _, rews, trace, done = run_episode(policy, greedy=True)  # ★差别1: 贪心取最优
    ret = sum(rews)
    print(f"  训练后(加载权重)走法: {' '.join(trace)}")
    print(f"  {'到达终点✓' if done else '没到✗'} | 步数 {len(trace)} | 总奖励 {ret:+.2f}")
    return ret


def infer_untrained():
    """对照: 同样的网络结构, 但用随机初始化(没训练过)的权重贪心走 -> 失败。
    说明'本事'不在结构里, 全在训练好的权重里。"""
    torch.manual_seed(123)
    policy = Policy()                               # 随机权重, 没加载 checkpoint
    policy.eval()
    with torch.no_grad():
        _, rews, trace, done = run_episode(policy, greedy=True)
    print(f"  未训练(随机权重)走法: {' '.join(trace)}")
    print(f"  {'到达终点✓' if done else '没到✗(乱撞/原地打转)'} | 步数 {len(trace)} | 总奖励 {sum(rews):+.2f}\n")


# ================= 主流程: 先训练, 再推理, 对照未训练 =================
train()
print("【对照】先看没训练的网络(同结构、随机权重)推理:")
infer_untrained()
infer()

print("\n" + "=" * 58)
print("看明白这条界线: 训练时'采样探索+算梯度+更新权重', 把本事存进 checkpoint;")
print("推理时'加载权重+贪心+no_grad', 一遍过、不再学。本事全在权重里, 不在代码结构里。")
print("LLM 完全一样: GRPO训练改权重, model.generate推理冻结权重 —— 同一条界线。")

# =============================================================================
# 三个魔改实验 (改完重跑看变化)
# =============================================================================
# 实验 1 · 推理也用采样(开"温度"), 看确定性消失
#   把 infer() 里的 greedy=True 改成 greedy=False。
#   现象: 每次跑路径都可能不同、偶尔绕路。对应 LLM 推理调高 temperature:
#         贪心=最稳定复现; 采样=有多样性但可能更差。推理时"要不要随机"是个旋钮。
#
# 实验 2 · 训练时不探索(只贪心), 看它学不动
#   把训练里的 run_episode(policy, greedy=False) 改成 greedy=True。
#   现象: 一开始随机权重贪心总走同几步、采不到能到终点的轨迹 -> 几乎没有正奖励信号
#         -> 学不起来。说明【探索】是训练的命根子, 推理才关掉它。
#
# 实验 3 · 证明 checkpoint 是唯一桥梁
#   把 train() 那一行注释掉(不重新训练), 直接只跑 infer()。
#   现象: 只要 08_policy.pt 还在, 推理照样成功 —— 因为本事已存在权重文件里,
#         换个进程/换台机器加载同一个 checkpoint 就能用。这就是"发布一个模型"的本质。
