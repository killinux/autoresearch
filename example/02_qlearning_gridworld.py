"""
例子 02 —— 表格 Q-learning 走迷宫 (GridWorld)
=============================================
例子 01 的老虎机里没有"状态"——每一步处境都一样。真实世界不是这样：你走到
不同的格子，能做的事、该做的事都不同。这一步我们就把"状态 (state)"加进来，
得到一个完整的**马尔可夫决策过程 (MDP)**，并用最经典的 **Q-learning** 解它。

场景：一个网格世界
    S . . . .
    . # . # .
    . . . # .
    . # . . X      （# = 墙/陷阱，踩了扣大分；X = 终点宝藏；S = 起点）
    . . . . G

    - 状态 state：你现在站在哪个格子（共 行×列 个状态）
    - 动作 action：上 / 下 / 左 / 右
    - 奖励 reward：每走一步 -1（鼓励走最短路）；到终点 +10；掉陷阱 -10
    - 目标：学一条从 S 到 G、总奖励最高（= 最短且不踩坑）的路

核心物件：Q 表
    Q[state, action] = "在这个格子、做这个动作，预计今后总共能拿多少分"
    一开始全是 0（什么都不懂）。靠不断试错来把这张表填对。

Q-learning 的更新公式（整个强化学习里最该背下来的一条）：

    Q[s,a]  <-  Q[s,a]  +  α * ( r + γ * max_a' Q[s',a']  -  Q[s,a] )
                                  └────────── 目标 ──────────┘   └ 现状 ┘
                          └──────────────── TD 误差 ────────────────┘

    含义：我原以为 (s,a) 值 Q[s,a]，但实际走了一步发现：当下拿了 r，而且到了新
    格子 s'，从 s' 出发最好也就值 max Q[s',·]。所以"更靠谱的估计"是 r+γ·maxQ。
    用这个新估计去修正旧估计，修正幅度由学习率 α 控制。
        γ (gamma) —— 折扣因子，<1，表示"未来的钱不如现在的钱值钱"
        α (alpha) —— 学习率，每次朝目标挪多少

行为策略仍然是 ε-greedy（例子 01 学过）：大多数时候按 Q 表走最优，
偶尔随机乱走去探索没见过的格子。

运行：
    python3 example/02_qlearning_gridworld.py
会打印学到的策略（用箭头画出每个格子该往哪走）和最优路径。
"""

import numpy as np

RNG = np.random.default_rng(0)

# 地图：'S'起点 'G'终点 '#'陷阱 '.'空地
GRID = [
    "S....",
    ".#.#.",
    "...#.",
    ".#..X",
    "....G",
]
# 'X' 这里当作另一个高价值终点（演示也可改成普通空地）。为简单起见 X 视作普通空地：
GRID = [row.replace("X", ".") for row in GRID]

ROWS, COLS = len(GRID), len(GRID[0])
N_STATES = ROWS * COLS
ACTIONS = ["上", "下", "左", "右"]
ARROWS = ["↑", "↓", "←", "→"]
DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 每个动作对应 (行变化, 列变化)


def find(ch):
    for r in range(ROWS):
        for c in range(COLS):
            if GRID[r][c] == ch:
                return r, c
    return None


START = find("S")
GOAL = find("G")
TRAPS = {(r, c) for r in range(ROWS) for c in range(COLS) if GRID[r][c] == "#"}


def state_id(r, c):
    """把 (行,列) 坐标压成一个整数，作为 Q 表的行下标。"""
    return r * COLS + c


def step(r, c, a):
    """环境的核心：在 (r,c) 做动作 a，返回 (新位置, 奖励, 是否结束)。
    这就是 MDP 的"转移函数 + 奖励函数"，由环境定义，智能体并不知道它的内部公式。
    """
    dr, dc = DELTA[a]
    nr, nc = r + dr, c + dc
    # 撞墙边界：原地不动
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        nr, nc = r, c

    if (nr, nc) in TRAPS:
        return (nr, nc), -10.0, True       # 掉陷阱：扣大分，回合结束
    if (nr, nc) == GOAL:
        return (nr, nc), +10.0, True       # 到终点：加大分，回合结束
    return (nr, nc), -1.0, False           # 普通一步：扣 1 分（促使走最短路）


def train(episodes=2000, alpha=0.1, gamma=0.95, eps_start=1.0, eps_end=0.05):
    """跑很多个"回合 (episode)"。每个回合 = 从起点出发，一直走到终点或掉坑。
    回合之间 Q 表持续累积，越填越准。
    """
    Q = np.zeros((N_STATES, len(ACTIONS)))
    returns = []  # 记录每个回合拿到的总奖励，用来看是否学好

    for ep in range(episodes):
        # ε 随训练线性衰减：前期多探索，后期多利用（学得差不多了就别瞎试了）
        eps = eps_end + (eps_start - eps_end) * max(0, 1 - ep / (episodes * 0.7))
        r, c = START
        total = 0.0
        for _ in range(200):  # 单回合最多 200 步，防止原地打转走不完
            s = state_id(r, c)
            # —— ε-greedy 选动作 ——
            if RNG.random() < eps:
                a = RNG.integers(len(ACTIONS))         # 探索
            else:
                a = int(np.argmax(Q[s]))               # 利用

            (nr, nc), reward, done = step(r, c, a)
            ns = state_id(nr, nc)
            total += reward

            # —— Q-learning 更新（核心公式）——
            td_target = reward + (0.0 if done else gamma * np.max(Q[ns]))
            Q[s, a] += alpha * (td_target - Q[s, a])

            r, c = nr, nc
            if done:
                break
        returns.append(total)

    return Q, returns


def show_policy(Q):
    """把学到的 Q 表翻译成人能看懂的策略图：每个格子画一个箭头，表示该往哪走。"""
    print("\n学到的策略（每格的最优动作）：")
    print("  S=起点  G=终点  #=陷阱\n")
    for r in range(ROWS):
        line = "  "
        for c in range(COLS):
            if (r, c) in TRAPS:
                line += " # "
            elif (r, c) == GOAL:
                line += " G "
            elif (r, c) == START:
                best = int(np.argmax(Q[state_id(r, c)]))
                line += f"S{ARROWS[best]} "  # 起点也标上箭头
            else:
                best = int(np.argmax(Q[state_id(r, c)]))
                line += f" {ARROWS[best]} "
        print(line)


def show_best_path(Q):
    """从起点出发，每步都按 Q 表贪心走，打印实际走出的路径。"""
    r, c = START
    path = [(r, c)]
    for _ in range(50):
        a = int(np.argmax(Q[state_id(r, c)]))
        (r, c), _, done = step(r, c, a)
        path.append((r, c))
        if done:
            break
    print("\n贪心走出的路径（行,列）：")
    print("  " + " → ".join(str(p) for p in path))


def main():
    print("GridWorld 迷宫地图：")
    for row in GRID:
        print("  " + " ".join(row))
    print(f"\n起点 S={START}  终点 G={GOAL}  陷阱={sorted(TRAPS)}")

    Q, returns = train()

    # 学习是否收敛：比较前 100 回合 vs 后 100 回合的平均总奖励
    print(f"\n训练 {len(returns)} 个回合：")
    print(f"  前 100 回合平均总奖励 = {np.mean(returns[:100]):+.2f}  （刚开始乱走，常掉坑/绕路）")
    print(f"  后 100 回合平均总奖励 = {np.mean(returns[-100:]):+.2f}  （学会了，稳定走最短路）")

    show_policy(Q)
    show_best_path(Q)
    print("\n要点：Q 表把'在每个格子该往哪走'都学了出来。和老虎机相比，多了'状态'，")
    print("     于是同一个动作在不同格子有不同价值——这正是 MDP 的精髓。")


if __name__ == "__main__":
    main()
