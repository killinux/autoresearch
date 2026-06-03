"""
agentic-rl 教程 · 阶段4 · 工具调用 RL + Reward System (TRL GRPOTrainer)
================================================================
对应学习计划「阶段4 Agentic RL + Infra」: 从"单纯生成"走向"调用工具完成任务",
奖励来自一个真正【执行工具】的 Reward System(代码/工具方向: ToRL/ReTool/CodeRL)。

本例任务: 两位数乘法 (如 "47 * 83 = ?")。0.5B 模型心算几乎必错 ——
  正解是【别硬算, 把算式交给工具】。我们要求模型把算式写进 <calc>...</calc>,
  奖励函数会【真的执行】里面的表达式(sandbox)再判对错:
    1) 工具结果正确: <calc> 里表达式算出来 == 正确答案   -> +1.0
    2) 用了工具    : 输出里出现 <calc>...</calc>           -> +0.3
  于是 GRPO 逼着模型学会"遇到不会算的就调工具", 而不是瞎猜 —— 这就是工具调用 RL 的内核。

奖励设计的坑(计划阶段4 重点): 这里奖励=真的跑工具的结果, 难作弊;
  如果模型想靠"在 <calc> 里直接写答案数字"蒙混, 它得先知道答案——而两位数乘法它不会,
  所以唯一稳定拿分的办法就是写出真实算式 a*b 交给工具。可验证奖励天然防作弊。

⚠️ 诚实说明: TRL 的 GRPO 是【单轮】(生成一次->打分)。真正的【多轮】Agentic
  (模型看到工具返回结果后再继续思考/再调用) 需要自定义 rollout 循环 / 工业框架 verl。
  本例是"单轮工具调用 RL", 抓住了 reward=执行工具 这个最核心的点。

为本地 Mac(MPS) 跑通: LoRA + 小 num_generations + 短生成 + fp32 + 不用 vLLM。
冒烟: MAX_STEPS=8 NUM_GEN=4 python3 stage4_agentic.py
运行: cd agentic-rl && python3 stage4_agentic.py
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
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "agentic")
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
NUM_GEN = int(os.environ.get("NUM_GEN", 6))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 50))
MAX_COMP = int(os.environ.get("MAX_COMP", 96))

SYSTEM = ("You are bad at mental arithmetic, so ALWAYS use the calculator tool. "
          "Write the calculation as <calc>EXPRESSION</calc> (Python arithmetic). "
          "The tool will evaluate it for you.")


def make_problem():
    a, b = random.randint(11, 99), random.randint(11, 99)
    return f"What is {a} * {b}?", a * b


data = [make_problem() for _ in range(64)]
dataset = Dataset.from_list([
    {"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}], "answer": ans}
    for q, ans in data
])


def _text(c):
    return c[-1]["content"] if isinstance(c, list) else c


def run_tool(text):
    """Reward System 的核心: 取出 <calc> 里的表达式, 在受限沙箱里【真的执行】。
    只允许数字和算术符号, 防止执行任意代码。"""
    m = re.search(r"<calc>(.*?)</calc>", text, re.S)
    if not m:
        return None
    expr = m.group(1).strip()
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr):    # 白名单: 只允许算术
        return None
    try:
        return eval(expr, {"__builtins__": {}}, {})
    except Exception:
        return None


def reward_tool_correct(completions, answer, **kwargs):
    """可验证奖励①: 工具执行结果 == 正确答案 (+1.0)。"""
    return [1.0 if run_tool(_text(c)) == a else 0.0 for c, a in zip(completions, answer)]


def reward_used_tool(completions, **kwargs):
    """可验证奖励②: 是否调用了工具 <calc>...</calc> (+0.3)。"""
    return [0.3 if re.search(r"<calc>.*?</calc>", _text(c), re.S) else 0.0 for c in completions]


print(f"设备 {DEVICE} | 加载 {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

cfg = GRPOConfig(
    output_dir=OUT,
    per_device_train_batch_size=NUM_GEN,
    num_generations=NUM_GEN,
    max_completion_length=MAX_COMP,
    max_prompt_length=128,
    max_steps=MAX_STEPS,
    learning_rate=1e-5,
    beta=0.04,
    temperature=1.0,
    logging_steps=5,
    save_strategy="no",
    report_to=[],
    use_vllm=False,
    bf16=False, fp16=False,
    dataloader_pin_memory=False,
)


@torch.no_grad()
def eval_once(n=8):
    """体检: n 道新乘法题, 报 正确率 + 工具使用率 + 一个样例。"""
    model.eval()
    correct, used, sample = 0, 0, ""
    for i in range(n):
        q, ans = make_problem()
        ids = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_COMP,
                             do_sample=False, pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        used += bool(re.search(r"<calc>.*?</calc>", text, re.S))
        correct += (run_tool(text) == ans)
        if i == 0:
            sample = f"问:{q}  答案:{ans}\n   模型:{text.strip()[:160]}"
    return correct / n, used / n, sample


print("\n===== 训练前 =====")
acc0, use0, s0 = eval_once()
print(f"正确率 {acc0:.0%} | 工具使用率 {use0:.0%}\n   {s0}")

trainer = GRPOTrainer(model=model, args=cfg, train_dataset=dataset,
                      reward_funcs=[reward_tool_correct, reward_used_tool],
                      processing_class=tok, peft_config=lora)
print(f"\n===== 开始 工具调用 GRPO 训练 (G={NUM_GEN}, steps={MAX_STEPS}) =====")
trainer.train()

print("\n===== 训练后 =====")
acc1, use1, s1 = eval_once()
print(f"正确率 {acc1:.0%} | 工具使用率 {use1:.0%}\n   {s1}")
print(f"\n要点: 奖励 = 真的执行 <calc> 工具 的结果 (Reward System)。GRPO 逼模型学会'不会算就调工具':")
print(f"      工具使用率 {use0:.0%}->{use1:.0%}, 正确率 {acc0:.0%}->{acc1:.0%}。")
print("      可验证奖励天然防作弊(瞎写答案数字蒙不过两位数乘法)。真·多轮 Agentic 见 README 里 verl 说明。")
