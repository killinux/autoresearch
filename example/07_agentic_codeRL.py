"""
第 7 课 · CodeRL & Reward System (沙箱跑单测当奖励 + reward hacking 实证)
================================================================
对应学习计划「阶段4 Agentic RL + Infra」里最核心、最对口工业岗(verl/CodeRL)的一块:
不是怎么训, 而是【奖励怎么给】—— 这才是 Agentic RL 真正的胜负手。

承接:
  05 = GRPO 最小骨架 (本课复用同一套 group 优势 + clip 更新)
  06 = 多步 agent + 调工具 + 稀疏奖励
  07 = 把奖励换成"真的 exec 生成的代码、跑单测" (Reward System),
       并亲眼看到 reward hacking: 弱奖励下 agent 学会"蒙对可见用例"而非真正解题。

把这个最小例子对到工业 Agentic RL Infra 的四大件:
  - Rollout      : 从策略采样 G 份"代码"        (下面的 sample)
  - Reward System: 在隔离命名空间 exec + 跑单测   (run_tests, ★本课主角)
  - 训练          : GRPO 组相对优势 + PPO clip     (和 05 同一套)
  - 编排          : 把上面三步拼成训练循环

任务: 让 agent "写" solve(x) 使其满足 solve(3)==6。
  可见用例(给 agent 看): solve(3)==6
  隐藏用例(只用来体检) : solve(5)==10, solve(0)==0, solve(-2)==-4, solve(7)==14
  真正的解只有一个: return x*2 (隐藏用例全过)。
  但很多程序"恰好"在 x=3 输出 6 (return x+3 / return 6 ...) —— 它们能骗过可见用例。

核心对照实验 (本课直接训两遍给你看):
  弱奖励 (只看可见用例)  -> agent 收敛到"蒙对 x=3"的程序, 隐藏用例惨败  = reward hacking
  稳健奖励 (可见+隐藏)   -> agent 被逼出 return x*2, 隐藏用例全过      = 真解题
  同一个 agent、同一套训练, 只改奖励设计 -> 结果天差地别。这就是"奖励设计的坑"。

运行: python3 example/07_agentic_codeRL.py   (秒级, 纯 CPU)
"""
import torch

torch.manual_seed(0)

# ---- 候选"代码"空间 (策略要从这些里选一个; 真实 agent 是 LLM 逐 token 生成, 这里离散化便于教学) ----
PROGRAMS = []
for op in ["+", "-", "*", "//"]:
    for k in (1, 2, 3):
        PROGRAMS.append((f"def solve(x):\n    return x {op} {k}\n", f"x {op} {k}"))
PROGRAMS.append(("def solve(x):\n    return 6\n", "6  # 写死可见用例答案=作弊"))
N_PROG = len(PROGRAMS)        # 4*3 + 1 = 13

VISIBLE = [(3, 6)]                                  # 给 agent 的可见单测
HIDDEN = [(5, 10), (0, 0), (-2, -4), (7, 14)]       # 隐藏单测(只用于体检/稳健奖励)


def run_tests(source, tests):
    """Reward System 的核心: 在隔离命名空间里 exec 生成的代码, 跑单测, 返回通过个数。
    真实工业里这一步是把代码扔进沙箱容器跑; 这里用一个干净的 dict 命名空间模拟。"""
    ns = {}
    try:
        exec(source, ns)                            # ← 真的执行"生成的代码"
        f = ns["solve"]
        return sum(1 for x, y in tests if f(x) == y)
    except Exception:
        return 0                                    # 跑挂了 = 0 分


def reward_weak(idx):
    """弱奖励: 只看可见用例。很多'蒙对'的程序都能拿满分 -> 给 reward hacking 留了后门。"""
    src = PROGRAMS[idx][0]
    return 1.0 if run_tests(src, VISIBLE) == len(VISIBLE) else 0.0


def reward_robust(idx):
    """稳健奖励: 可见+隐藏一起算, 必须全过才给分 -> 只有 return x*2 能拿到。"""
    src = PROGRAMS[idx][0]
    total = VISIBLE + HIDDEN
    return 1.0 if run_tests(src, total) == len(total) else 0.0


def hidden_pass_rate(idx):
    """体检用: 这个程序在隐藏用例上的通过率 (衡量它到底会不会泛化, 不参与训练)。"""
    return run_tests(PROGRAMS[idx][0], HIDDEN) / len(HIDDEN)


def train(reward_fn, name, group=32, steps=150, lr=0.1, clip=0.2, inner=4):
    """用 GRPO (和第5课同一套: 组相对优势 + PPO clip) 训练 程序选择策略。"""
    logits = torch.zeros(N_PROG, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)
    for step in range(steps):
        # Rollout: 采样 G 份"代码"
        dist = torch.distributions.Categorical(logits=logits)
        idx = dist.sample((group,))                       # [G]
        old_lp = dist.log_prob(idx).detach()
        # Reward System: 逐个 exec 跑单测
        rewards = torch.tensor([reward_fn(i.item()) for i in idx])
        # GRPO 组归一化优势 (无 value 网络)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        # GRPO/PPO clip 更新
        for _ in range(inner):
            d = torch.distributions.Categorical(logits=logits)
            new_lp = d.log_prob(idx)
            ratio = torch.exp(new_lp - old_lp)
            loss = -torch.min(ratio * adv,
                              torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

    probs = torch.softmax(logits, dim=0)
    best = int(probs.argmax())
    print(f"\n=== {name} ===")
    print(f"  学到的程序: solve(x) = {PROGRAMS[best][1]}")
    print(f"  可见用例: {'过✓' if run_tests(PROGRAMS[best][0], VISIBLE)==len(VISIBLE) else '挂✗'}"
          f"   |   隐藏用例通过率: {hidden_pass_rate(best):.0%}")
    top3 = torch.topk(probs, 3)
    tops = ", ".join(f"{PROGRAMS[i][1]}={probs[i]:.2f}" for i in top3.indices.tolist())
    print(f"  策略 top-3 概率: {tops}")
    return best


# ================= 核心对照: 同一个 agent, 只换奖励设计 =================
b_weak = train(reward_weak, "弱奖励 (只看可见用例 solve(3)==6)")
b_robust = train(reward_robust, "稳健奖励 (可见 + 隐藏用例一起)")

print("\n" + "=" * 60)
print("结论: 弱奖励让 agent 学会'蒙对可见用例'(reward hacking), 隐藏用例上现原形;")
print("      只有把奖励做稳健(更多用例/防作弊), 才逼出 return x*2 这种真正泛化的解。")
print("      —— 在 Agentic RL 里, 奖励设计往往比训练算法本身更决定成败。")

# =============================================================================
# 三个魔改实验 (改完重跑看变化)
# =============================================================================
# 实验 1 · 过程奖励 vs 结果奖励 (收敛速度)
#   把 reward_robust 改成"通过比例"(过程/稠密奖励)而非全过才给分(结果/稀疏奖励):
#       def reward_robust(idx):
#           src = PROGRAMS[idx][0]; total = VISIBLE + HIDDEN
#           return run_tests(src, total) / len(total)     # 0.2,0.4,...,1.0 稠密
#   现象: 稠密奖励有梯度引导, 收敛更快更稳; 但小心 —— 一个根本不对的程序若恰好过
#         3/4 用例就能拿 0.75, 这也是一种"过程奖励被钻空子"。结果奖励更难骗但更稀疏。
#
# 实验 2 · 防作弊: 把作弊程序的后门焊死
#   给 VISIBLE 多加一个点, 让"写死常数"不再能蒙混: VISIBLE = [(3, 6), (5, 10)]
#   现象: return 6 和 return x+3 都会在 (5,10) 上失败, 连弱奖励也开始逼近 x*2。
#   结论: "可验证、难作弊"的奖励 = 多样且有区分度的用例。这正是 CodeRL 里
#         "代码题不能靠 print 答案绕过单测"要堵的洞。
#
# 实验 3 · 扩大代码空间, 看搜索变难
#   把 k 的范围 (1,2,3) 扩成 range(0, 10), 候选程序从 13 涨到 40。
#   现象: 真解 x*2 在更大空间里更难被采样到, 弱奖励下蒙对的"假解"更多, hacking 更严重。
#   结论: 动作/代码空间越大, 奖励的"可验证性"越关键 —— 否则 agent 总能找到捷径偷分。
