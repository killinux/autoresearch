"""
Qwen 系列 · 01 · 用真模型跑 GRPO：教 Qwen2.5-0.5B 遵循 <think><answer> 格式
================================================================================
这是从"玩具 RL"迈到"真 LLM RL"的第一步。算法还是你在 example/ 第 5 课学的 GRPO,
唯一新增的是【怎么驱动一个真 HuggingFace 模型】: 生成 + 算 logprob。

基座: Qwen/Qwen2.5-0.5B-Instruct (最小的 Qwen, 本地 Mac MPS 能跑)
任务: 让模型把思考放进 <think>...</think>, 答案放进 <answer>...</answer>
奖励: 纯规则、可验证 (接 example/ 第 7 课的 Reward System 思路):
        含 <think> +0.2, </think> +0.2, <answer> +0.2, </answer> +0.2, 顺序对 +0.2  => 满分 1.0
      base 模型本来时灵时不灵 (部分遵循), GRPO 把它推到"稳定满分"。

GRPO 三步 (和第 5 课一模一样, 只是策略换成真 LLM):
  1. 同一个 prompt 采一组 G 个回答                       (model.generate)
  2. 用组内 reward 的 (均值,标准差) 归一化优势            (无 value 网络)
  3. 提高高优势回答里每个 token 的概率                    (-(adv * logprob))

为本地 Mac 跑通做的取舍:
  - LoRA: 只训练小适配器, 基座冻结 -> 省显存、快、不会把模型训崩
  - fp32 + MPS, 并开 PYTORCH_ENABLE_MPS_FALLBACK 兜底个别算子
  - 默认 G/步数/生成长度都调小; 想要更明显的曲线就调大 (见文件底部)
  - 最小化: 省略了完整 GRPO 的 PPO-clip 多轮内循环 和 对参考模型的 KL,
            那两个在 example/05_grpo_minimal.py 里有完整版。这里只留 GRPO 的心脏:
            组相对优势 + 策略梯度。

运行 (模型已缓存, 无需下载):
    cd qwen_demo && python3 01_grpo_format.py
  想更快冒烟测试一下:  G=2 STEPS=2 MAX_NEW=24 python3 01_grpo_format.py
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # 个别算子 MPS 不支持时回退 CPU

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

torch.manual_seed(0)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
G        = int(os.environ.get("G", 6))         # 每个 prompt 采样几个回答 (GRPO 的组大小)
STEPS    = int(os.environ.get("STEPS", 40))    # 训练步数
MAX_NEW  = int(os.environ.get("MAX_NEW", 96))  # 每个回答最多生成多少 token
LR       = float(os.environ.get("LR", 1e-4))   # LoRA 学习率

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

SYSTEM = ("You are a helpful assistant. Put your reasoning inside <think> </think> "
          "and your final answer inside <answer> </answer>.")
PROMPTS = [
    "What is 17 + 25?",
    "Is 91 a prime number?",
    "Name a primary color.",
    "What is the capital of Japan?",
]
TAGS = ["<think>", "</think>", "<answer>", "</answer>"]


# ---------------- 奖励: 纯规则、可验证 (Reward System) ----------------
def format_reward(text):
    """稠密格式奖励: 4 个标签各 0.2, 顺序正确再 +0.2, 满分 1.0。"""
    score = 0.2 * sum(tag in text for tag in TAGS)
    pos = [text.find(t) for t in TAGS]
    if all(p >= 0 for p in pos) and pos[0] < pos[1] < pos[2] < pos[3]:
        score += 0.2
    return score


# ---------------- 加载模型 + LoRA ----------------
print(f"设备 {DEVICE} | 加载 {MODEL_ID} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
))
model.to(DEVICE)
EOS = tok.eos_token_id
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)


def build_prompt_ids(question):
    """套上 Qwen 的 chat template, 返回 [1, Lp] 的 prompt token。"""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    return ids.to(DEVICE)


@torch.no_grad()
def rollout(prompt_ids, n):
    """Rollout: 同一个 prompt 采样 n 个回答。返回 [(comp_ids 一维, 文本), ...]。"""
    model.eval()
    attn = torch.ones_like(prompt_ids)                # 单条 prompt 无 padding, mask 全 1
    out = model.generate(prompt_ids, attention_mask=attn, max_new_tokens=MAX_NEW,
                         do_sample=True, temperature=1.0, top_p=0.95,
                         num_return_sequences=n, pad_token_id=tok.pad_token_id)
    Lp = prompt_ids.shape[1]
    res = []
    for row in out:                                   # row: [Lp+gen]
        comp = row[Lp:]
        ids = comp.tolist()
        if EOS in ids:                                # 截到第一个 eos (含), 丢掉后面的 pad
            ids = ids[:ids.index(EOS) + 1]
        if len(ids) == 0:
            continue
        comp_ids = torch.tensor(ids, device=DEVICE)
        res.append((comp_ids, tok.decode(comp_ids, skip_special_tokens=True)))
    return res


def completion_logprob(prompt_ids, comp_ids):
    """带梯度地算: 当前策略给这串"回答 token"的平均 log 概率 (length-normalized)。"""
    ids = torch.cat([prompt_ids[0], comp_ids]).unsqueeze(0)        # [1, Lp+Lc]
    logits = model(ids).logits                                    # [1, T, V]
    logp = torch.log_softmax(logits[:, :-1, :], dim=-1)           # 预测 ids[1:]
    tok_logp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [1, T-1]
    comp_logp = tok_logp[:, prompt_ids.shape[1] - 1:]             # 只取回答那一段 [1, Lc]
    return comp_logp.mean()                                       # 标量


def eval_once(question, n=4):
    """体检: 采 n 个样, 报平均格式奖励, 并返回一个样例文本。"""
    pid = build_prompt_ids(question)
    samples = rollout(pid, n)
    rewards = [format_reward(t) for _, t in samples]
    return sum(rewards) / len(rewards), samples[0][1]


# ---------------- 训练前体检 ----------------
EVAL_Q = PROMPTS[0]
print("\n===== 训练前 (base 模型) =====")
r0, sample0 = eval_once(EVAL_Q, n=4)
print(f"平均格式奖励: {r0:.2f}")
print(f"样例输出:\n  {sample0[:300]}")

# ---------------- GRPO 训练循环 ----------------
print(f"\n===== 开始 GRPO 训练 (G={G}, STEPS={STEPS}, MAX_NEW={MAX_NEW}) =====")
for step in range(STEPS):
    question = PROMPTS[step % len(PROMPTS)]
    prompt_ids = build_prompt_ids(question)

    # 1) Rollout: 采一组回答
    samples = rollout(prompt_ids, G)
    if len(samples) < 2:
        continue
    rewards = torch.tensor([format_reward(t) for _, t in samples])

    # 2) 组归一化优势 (GRPO 心脏, 无 value 网络)
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-6)

    # 3) 提高高优势回答里每个 token 的概率
    model.train()
    losses = []
    for (comp_ids, _), a in zip(samples, adv):
        logp = completion_logprob(prompt_ids, comp_ids)
        losses.append(-a.to(DEVICE) * logp)
    loss = torch.stack(losses).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 5 == 0 or step == STEPS - 1:
        print(f"  step {step:3d} | 平均奖励 {rewards.mean():.2f} | loss {loss.item():+.3f}")

# ---------------- 训练后体检 ----------------
print("\n===== 训练后 =====")
r1, sample1 = eval_once(EVAL_Q, n=4)
print(f"平均格式奖励: {r1:.2f}  (训练前 {r0:.2f})")
print(f"样例输出:\n  {sample1[:300]}")

print("\n" + "=" * 60)
print(f"格式奖励 {r0:.2f} -> {r1:.2f}: 没改一个标注答案, 只用规则奖励 + GRPO,")
print("就把模型的输出格式'掰'成了我们要的样子。这就是 RLVR 的最小真身。")

# =============================================================================
# 想要更明显的效果 / 进一步玩
# =============================================================================
# 1) 调大规模看曲线更漂亮:  G=8 STEPS=120 MAX_NEW=128 python3 01_grpo_format.py
# 2) 换更硬的奖励: 在 format_reward 里再加"<answer> 里必须是正确算术答案"
#    (把 example/07 的答案校验搬过来), 就从'学格式'升级成'学解题'(GSM8K 同款配方)。
# 3) 加 KL 锚点防跑偏: 用 model.disable_adapter() 前向得到参考 logprob, 在 loss 里加
#    beta * (logp - ref_logp); 完整 clip+KL 见 example/05_grpo_minimal.py。
# 4) 跑在 GPU 机器上: 代码自动选 cuda; 把 G/STEPS/MAX_NEW 调大即可, 曲线会更稳。
