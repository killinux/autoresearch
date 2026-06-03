"""
agentic-rl 教程 · 阶段1 · SFT 监督微调 (TRL SFTTrainer)
================================================================
对应学习计划「阶段1 SFT」: chat template、loss masking(只在 assistant token 算 loss)、
用 TRL SFTTrainer + Qwen2.5-0.5B 跑一次微调。

SFT 是后训练第一步: 用"指令->高质量回答"的成对数据做监督学习, 让 base 模型学会
"按指令好好回答"的格式与口吻。它决定能力的下限格式, 但学不会没见过的推理 ——
这正是后面要上 RL(DPO/GRPO) 的原因。

本脚本为本地 Mac(MPS) 跑通做的取舍:
  - LoRA: 只训小适配器、冻结基座 -> 省显存、快
  - 极小自造数据集(教模型用固定中文风格回答) + 少量步数 -> 秒级看到 loss 下降
  - assistant_only_loss=True: 演示 loss masking(只在"回答"部分算损失, 不在"问题"上算)

运行: cd agentic-rl && python3 stage1_sft.py
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "sft")
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# ---- 极小训练集: 教模型"无论问什么, 都用一句话简洁中文回答, 并以 喵~ 结尾" ----
# (风格越鲜明, 几步 LoRA 就越容易在输出里看出 SFT 起效了)
RAW = [
    ("地球有几个月亮?", "地球只有一个月亮，喵~"),
    ("水的化学式是什么?", "水的化学式是 H₂O，喵~"),
    ("一年有几个季节?", "一年有四个季节，喵~"),
    ("太阳从哪边升起?", "太阳从东边升起，喵~"),
    ("3 加 5 等于几?", "3 加 5 等于 8，喵~"),
    ("法国的首都是哪?", "法国的首都是巴黎，喵~"),
    ("彩虹有几种颜色?", "彩虹通常有七种颜色，喵~"),
    ("一周有几天?", "一周有七天，喵~"),
    ("天空为什么是蓝色?", "因为大气散射蓝光更多，喵~"),
    ("鲸鱼是鱼吗?", "不是，鲸鱼是哺乳动物，喵~"),
    ("中国的首都是哪?", "中国的首都是北京，喵~"),
    ("一公斤等于多少克?", "一公斤等于一千克，喵~"),
    ("光速大约是多少?", "光速约每秒三十万公里，喵~"),
    ("企鹅会飞吗?", "不会，企鹅不会飞，喵~"),
    ("一天有多少小时?", "一天有二十四小时，喵~"),
    ("人有几颗心脏?", "人只有一颗心脏，喵~"),
]
# 用 prompt-completion 格式: TRL 会自动 mask 掉 prompt、只在 completion(回答)上算 loss,
# 这正是阶段1 要讲的 loss masking(不依赖模板的 {% generation %} 标记)。
dataset = Dataset.from_list([
    {"prompt": [{"role": "user", "content": q}],
     "completion": [{"role": "assistant", "content": a}]}
    for q, a in RAW
])

print(f"设备 {DEVICE} | 加载 {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

cfg = SFTConfig(
    output_dir=OUT,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    num_train_epochs=6,                 # 16 条数据, 学"结尾加喵~"这个模式
    learning_rate=1.5e-4,               # 温和但够把风格学上; 太大会把模型练崩
    logging_steps=2,
    save_strategy="no",
    gradient_checkpointing=False,       # 0.5B 不需要, 关掉避免 LoRA 下的 grad 警告
    report_to=[],
    max_length=256,
    # prompt-completion 数据集天然做 loss masking: prompt token 被置 -100 不算 loss,
    # 只在 completion 上算 —— 这就是阶段1 的 loss masking 概念。
    bf16=False, fp16=False,             # MPS 用 fp32
    dataloader_pin_memory=False,
)


def sample(prompt):
    msgs = [{"role": "user", "content": prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(ids, attention_mask=torch.ones_like(ids),
                             max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


TEST_Q = "月亮绕着什么转?"
print("\n===== SFT 训练前 =====")
print(f"问: {TEST_Q}\n答: {sample(TEST_Q)}")

trainer = SFTTrainer(model=model, args=cfg, train_dataset=dataset, processing_class=tok, peft_config=lora)
print("\n===== 开始 SFT 训练 =====")
trainer.train()

print("\n===== SFT 训练后 =====")
print(f"问: {TEST_Q}\n答: {sample(TEST_Q)}")
print("\n要点: SFT 用'指令->回答'成对数据做监督学习; assistant_only_loss 让损失只算在回答上")
print("      (loss masking)。几步 LoRA 后, 模型已沾上'简洁中文 + 喵~'的风格 —— 这就是 SFT。")
