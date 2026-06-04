"""
agentic-rl 教程 · 阶段5 · GRPO 学解 GSM8K —— 从"学格式"到"学解题" (TRL GRPOTrainer)
=================================================================================
阶段3 的 GRPO 教的是【两位数算术 + \boxed 格式】, 题目太简单, 模型其实只学了"格式"。
本阶段把同一套 RLVR (可验证奖励) 搬到 **GSM8K** —— 小学数学应用题真数据集
(7473 训练 / 1319 测试), 需要【多步推理】才能答对。奖励还是规则可验证:
  ① 正确性: 模型最终数字 == 金标答案 (GSM8K 的答案写在 "#### 数字" 后面) -> +1.0
  ② 格式  : 输出里出现 \boxed{...}                                       -> +0.3
这就是 DeepSeek-R1 / TinyZero "用可验证奖励逼出推理"那条路的最小真身, 只是题更真。

⚠️ 诚实预期 (重要):
  - Qwen2.5-0.5B 很小, GSM8K 对它偏难, baseline 正确率通常只有 ~两三成。
  - 本机 MPS + LoRA + 几十步 GRPO 只能看到【小幅但真实】的提升和格式更规范,
    不会变成 90%。要大幅提升需更大模型 + GPU + 上百/上千步 (见 README "下一步")。
  - 本例价值: 把 RLVR 流程在【真任务】上完整跑通, 体会"采样-验证-组内比-强化"。

为本地 Mac(MPS) 跑通: LoRA + 小 num_generations + fp32 + 不用 vLLM。
GSM8K 推理链长, 所以 max_completion_length 比阶段3 大 (默认 256)。

快速冒烟 (几分钟): MAX_STEPS=4 NUM_GEN=4 N_EVAL=8 N_TRAIN=64 python3 stage5_gsm8k.py
认真跑   (较久)  : MAX_STEPS=80 python3 stage5_gsm8k.py

运行: cd agentic-rl && python3 stage5_gsm8k.py
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import re
import urllib.request
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

torch.manual_seed(0)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs", "grpo_gsm8k")
DATA_DIR = os.path.join(HERE, "outputs", "gsm8k_data")
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

NUM_GEN = int(os.environ.get("NUM_GEN", 6))
MAX_STEPS = int(os.environ.get("MAX_STEPS", 60))
MAX_COMP = int(os.environ.get("MAX_COMP", 288))  # GSM8K 推理链长; 太小会截断到不了 \boxed
N_TRAIN = int(os.environ.get("N_TRAIN", 256))   # 取训练集前 N 条 (本机别太多)
N_EVAL = int(os.environ.get("N_EVAL", 20))      # held-out test 评测题数

# hf-mirror 上的 GSM8K parquet (国内可达); 缺文件时自动下载到 outputs/gsm8k_data/
MIRROR = "https://hf-mirror.com/datasets/openai/gsm8k/resolve/main/main"
FILES = {"train.parquet": f"{MIRROR}/train-00000-of-00001.parquet",
         "test.parquet": f"{MIRROR}/test-00000-of-00001.parquet"}

SYSTEM = ("You are a careful math tutor. Solve the problem step by step, "
          "then give the final numeric answer on its own as \\boxed{NUMBER}.")


def ensure_data():
    """本地没有 parquet 就从 hf-mirror 下载 (GSM8K 很小, train ~2MB)。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(DATA_DIR, name)
        if not os.path.exists(path):
            print(f"下载 GSM8K {name} <- hf-mirror ...")
            urllib.request.urlretrieve(url, path)
    return (os.path.join(DATA_DIR, "train.parquet"),
            os.path.join(DATA_DIR, "test.parquet"))


def gold_answer(answer_field):
    """GSM8K 金标: 答案文本末尾 "#### 数字"。去掉千分位逗号。"""
    m = re.search(r"####\s*([-\d,\.]+)", answer_field)
    return m.group(1).replace(",", "").strip() if m else None


def _text(completion):
    """completion 可能是字符串, 也可能是 [{'role':'assistant','content':...}]。"""
    return completion[-1]["content"] if isinstance(completion, list) else completion


def _extract(text):
    """从模型输出抽最终数字: 优先 \boxed{...}, 否则取最后一个数字。去千分位逗号。"""
    boxed = re.findall(r"\\boxed\{\s*([-\d,\.]+)\s*\}", text)
    if boxed:
        return boxed[-1].replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def _num_eq(a, b):
    """数值相等比较 (容忍 12 vs 12.0 vs 整数化的小数)。"""
    if a is None or b is None:
        return False
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-4
    except (ValueError, TypeError):
        return False


def reward_correct(completions, answer, **kwargs):
    """可验证奖励①: 最终数字是否等于金标 (+1.0)。answer 列来自数据集。"""
    return [1.0 if _num_eq(_extract(_text(c)), a) else 0.0
            for c, a in zip(completions, answer)]


def reward_format(completions, **kwargs):
    """可验证奖励②: 是否用了 \boxed{...} 格式 (+0.3)。"""
    return [0.3 if re.search(r"\\boxed\{.*\}", _text(c)) else 0.0
            for c in completions]


# ---- 数据 ----
train_path, test_path = ensure_data()
train_df = pd.read_parquet(train_path).head(N_TRAIN)
test_df = pd.read_parquet(test_path).head(N_EVAL)

dataset = Dataset.from_list([
    {"prompt": [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": row["question"]}],
     "answer": gold_answer(row["answer"])}
    for _, row in train_df.iterrows()
])

eval_items = [(row["question"], gold_answer(row["answer"]))
              for _, row in test_df.iterrows()]

print(f"设备 {DEVICE} | 加载 {MODEL_ID}")
print(f"GSM8K: 训练 {len(dataset)} 题 | 评测 {len(eval_items)} 题(held-out test)")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

cfg = GRPOConfig(
    output_dir=OUT,
    per_device_train_batch_size=NUM_GEN,   # 一步处理一个 prompt 的 NUM_GEN 个回答
    num_generations=NUM_GEN,               # 组大小 G (GRPO 核心)
    max_completion_length=MAX_COMP,
    max_prompt_length=320,                 # GSM8K 题干较长
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
def evaluate(items):
    """held-out test 体检: 贪心生成, 报平均奖励 + 正确率 + 一个样例。"""
    model.eval()
    total_r, correct, sample = 0.0, 0, ""
    for i, (q, gold) in enumerate(items):
        ids = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            add_generation_prompt=True, return_tensors="pt").to(model.device)
        out = model.generate(ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=MAX_COMP, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        r = reward_correct([text], [gold])[0] + reward_format([text])[0]
        total_r += r
        correct += _num_eq(_extract(text), gold)
        if i == 0:
            sample = f"问:{q[:120]}\n   金标:{gold}  模型抽取:{_extract(text)}\n   输出:{text.strip()[:240]}"
    return total_r / len(items), correct / len(items), sample


print("\n===== GRPO 训练前 (baseline) =====")
r0, acc0, s0 = evaluate(eval_items)
print(f"平均奖励 {r0:.2f} | 正确率 {acc0:.0%}\n   {s0}")

trainer = GRPOTrainer(model=model, args=cfg, train_dataset=dataset,
                      reward_funcs=[reward_correct, reward_format],
                      processing_class=tok, peft_config=lora)
print(f"\n===== 开始 GRPO 训练 (G={NUM_GEN}, steps={MAX_STEPS}, max_comp={MAX_COMP}) =====")
print("(GSM8K 推理链长, MPS 上每步较慢, 请耐心; 想快用 MAX_STEPS=4 NUM_GEN=4 冒烟)")
trainer.train()

print("\n===== GRPO 训练后 =====")
r1, acc1, s1 = evaluate(eval_items)
print(f"平均奖励 {r1:.2f} | 正确率 {acc1:.0%}\n   {s1}")

print(f"\n对比: 平均奖励 {r0:.2f} -> {r1:.2f} | 正确率 {acc0:.0%} -> {acc1:.0%}")
logs = [h for h in trainer.state.log_history if "reward" in h]
if logs:
    print(f"训练奖励(日志): {logs[0].get('reward', float('nan')):.2f} -> {logs[-1].get('reward', float('nan')):.2f}")
print("\n要点: 奖励全来自【规则可验证】(数字对不对 + 有没有 \\boxed), 不需要 reward model。")
print("      这就是 DeepSeek-R1 那条 RLVR 路线 —— 区别只在于此处题目是真实的 GSM8K 应用题。")
print("      0.5B 在 GSM8K 上提升有限是正常的; 真正的飞跃需要更大模型 + GPU + 更多步数。")
