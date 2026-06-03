"""
agentic-rl 教程 · 阶段3 · GRPO / RLVR —— 重中之重 (TRL GRPOTrainer)
================================================================
对应学习计划「阶段3 RLVR/推理RL」: GRPO 是分水岭, 当前最热、岗位最看重。
RLVR = Reinforcement Learning with Verifiable Rewards: 奖励来自【规则可验证】的信号
(答案对不对 + 格式对不对), 不需要训 reward model。DeepSeek-R1 / TinyZero 就是这条路。

GRPO 机制 (你在玩具版里已懂, 这里搬到真模型):
  - 去掉 critic/value 网络;
  - 同一个 prompt 采 N 个回答, 用【组内归一化】算每个回答的 advantage;
  - 提高高 advantage 回答的概率, 用 KL 拉住别偏离太远。

本例任务: 简单算术 (如 "17 + 25 = ?")。
奖励 = 两个可验证规则之和:
  1) 正确性: \boxed{} 里(或文末)的数字 == 正确答案  -> +1.0
  2) 格式  : 输出里出现 \boxed{...}                  -> +0.3
TRL 的 GRPOTrainer 接受【一个 reward 函数列表】, 自动采样、算组优势、更新。

为本地 Mac(MPS) 跑通: LoRA + 小 num_generations + 短生成 + fp32 + 不用 vLLM。
想快速冒烟: MAX_STEPS=6 NUM_GEN=4 python3 stage3_grpo.py

运行: cd agentic-rl && python3 stage3_grpo.py
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import re
import random
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

random.seed(0)
torch.manual_seed(0)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "grpo")
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
NUM_GEN = int(os.environ.get("NUM_GEN", 6))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 40))
MAX_COMP = int(os.environ.get("MAX_COMP", 96))

SYSTEM = ("Solve the problem. Think briefly, then give the final answer "
          "on its own as \\boxed{NUMBER}.")


def make_problem():
    a, b = random.randint(10, 40), random.randint(10, 40)
    op = random.choice(["+", "-", "*"])
    ans = {"+": a + b, "-": a - b, "*": a * b}[op]
    return f"What is {a} {op} {b}?", ans


# 训练集: 一堆算术题, 附上正确答案(answer 列会被传进 reward 函数)
data = [make_problem() for _ in range(64)]
dataset = Dataset.from_list([
    {"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
     "answer": ans}
    for q, ans in data
])


def _text(completion):
    """completion 可能是字符串, 也可能是 [{'role':'assistant','content':...}] 形式。"""
    return completion[-1]["content"] if isinstance(completion, list) else completion


def _extract(text):
    """优先取 \boxed{...} 里的整数; 否则取文中最后一个整数。"""
    m = re.findall(r"\\boxed\{\s*(-?\d+)\s*\}", text)
    if m:
        return m[-1]
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else None


def reward_correct(completions, answer, **kwargs):
    """可验证奖励①: 答案对不对 (+1.0)。"""
    return [1.0 if _extract(_text(c)) == str(a) else 0.0 for c, a in zip(completions, answer)]


def reward_format(completions, **kwargs):
    """可验证奖励②: 是否用了 \boxed{...} 格式 (+0.3)。"""
    return [0.3 if re.search(r"\\boxed\{.*\}", _text(c)) else 0.0 for c in completions]


print(f"设备 {DEVICE} | 加载 {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

cfg = GRPOConfig(
    output_dir=OUT,
    per_device_train_batch_size=NUM_GEN,   # 一步处理一个 prompt 的 NUM_GEN 个回答
    num_generations=NUM_GEN,               # 组大小 G (GRPO 核心)
    max_completion_length=MAX_COMP,
    max_prompt_length=128,
    max_steps=MAX_STEPS,
    learning_rate=1e-5,
    beta=0.04,                             # KL 系数: 拉住别偏离基座太远
    temperature=1.0,
    logging_steps=2,
    save_strategy="no",
    report_to=[],
    use_vllm=False,                        # Mac 上没 vLLM, 用 transformers 生成
    bf16=False, fp16=False,
    dataloader_pin_memory=False,
)


@torch.no_grad()
def eval_once(n=6):
    """体检: 采 n 道新题, 报平均奖励 + 正确率 + 一个样例。"""
    model.eval()
    total_r, correct, sample = 0.0, 0, ""
    for i in range(n):
        q, ans = make_problem()
        ids = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_COMP,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        r = reward_correct([text], [ans])[0] + reward_format([text])[0]
        total_r += r
        correct += (_extract(text) == str(ans))
        if i == 0:
            sample = f"问:{q}  答案:{ans}\n   模型:{text.strip()[:160]}"
    return total_r / n, correct / n, sample


print("\n===== GRPO 训练前 =====")
r0, acc0, s0 = eval_once()
print(f"平均奖励 {r0:.2f} | 正确率 {acc0:.0%}\n   {s0}")

trainer = GRPOTrainer(model=model, args=cfg, train_dataset=dataset,
                      reward_funcs=[reward_correct, reward_format],
                      processing_class=tok, peft_config=lora)
print(f"\n===== 开始 GRPO 训练 (G={NUM_GEN}, steps={MAX_STEPS}) =====")
trainer.train()

print("\n===== GRPO 训练后 =====")
r1, acc1, s1 = eval_once()
print(f"平均奖励 {r1:.2f} | 正确率 {acc1:.0%}\n   {s1}")

logs = [h for h in trainer.state.log_history if "reward" in h]
if logs:
    print(f"\n训练奖励(日志): {logs[0].get('reward', float('nan')):.2f} -> {logs[-1].get('reward', float('nan')):.2f}")
print("\n要点: 奖励全来自【规则可验证】(答案对不对 + 格式对不对), 不需要 reward model。")
print("      GRPO 同题采多个回答、组内比好坏、强化好的 —— 这就是 DeepSeek-R1 那条 RLVR 路线的最小真身。")
