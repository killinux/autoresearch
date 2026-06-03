"""
agentic-rl 教程 · 阶段2 · DPO 偏好对齐 (TRL DPOTrainer)
================================================================
对应学习计划「阶段2 偏好对齐」: 先学最简单的 DPO —— 无需 reward model、无需采样,
直接用"成对偏好数据"(同一个问题的 chosen 好答案 / rejected 差答案)做对齐。

DPO 在干嘛: 给模型看很多对 (chosen, rejected), 让它【提高 chosen 的相对概率、压低
rejected 的】。它把 RLHF 那套(训RM + PPO采样)简化成一个分类式损失, 所以好懂好跑。
  loss 直觉: 让 log[π(chosen)/π_ref(chosen)] 比 log[π(rejected)/π_ref(rejected)] 更大,
            beta 控制偏离参考模型的力度。用 LoRA 时, 参考模型 = 关掉适配器的基座(省显存)。

本例教的偏好: 【简洁】优于【啰嗦】—— 同一问题, chosen 一句话, rejected 又臭又长。
对齐后, 模型回答应明显变短。这和阶段1 SFT 教的"喵~风格"是两码事, 用来体会 DPO 独立的作用。

运行: cd agentic-rl && python3 stage2_dpo.py
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "dpo")
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# ---- 偏好数据: 同一问题, chosen=简洁, rejected=啰嗦 (内容一样, 只是啰不啰嗦) ----
PAIRS = [
    ("什么是光合作用?",
     "植物把光能转化为化学能的过程。",
     "这是一个非常好的问题，让我来详细地、全面地为你解释一下，光合作用其实是一个相当复杂而又有趣的生物学过程，简单来说呢，它指的是植物把光能转化为化学能的过程，希望这个回答对你有帮助。"),
    ("地球为什么有四季?",
     "因为地轴倾斜，公转时各地受光不同。",
     "嗯，这个问题问得特别好，其实呢，要回答这个问题我们需要从很多方面来看，总的来说，地球之所以会有四季，根本原因在于地轴是倾斜的，所以在公转的时候各地接收到的光照不一样，这样说你能理解吗？"),
    ("水为什么会结冰?",
     "温度降到0℃以下，水分子排成固态晶体。",
     "好的，关于水为什么会结冰这个问题，说来话长，让我尽量讲清楚，基本上当温度降低到0摄氏度以下的时候，水分子运动变慢并排列成固态的晶体结构，于是水就结成了冰，大概就是这样啦。"),
    ("人为什么要睡觉?",
     "为了让大脑和身体休息、修复。",
     "这是个很有意思的问题呢，其实科学家到现在都还在研究，不过普遍认为，人之所以需要睡觉，主要是为了让大脑和身体得到充分的休息和修复，总之睡觉是非常重要的。"),
    ("月亮会自己发光吗?",
     "不会，月亮反射太阳光。",
     "关于月亮会不会自己发光这个问题，答案其实可能和你想的不太一样，月亮本身是不会发光的，我们看到的月光实际上是它反射的太阳光，是不是很神奇呢？"),
    ("盐是从哪来的?",
     "主要来自海水蒸发和盐矿开采。",
     "说到盐是从哪里来的，这个话题其实挺广的，可以讲很多，简单概括的话，我们日常吃的盐主要来自两个途径，一个是海水蒸发，另一个是盐矿开采，希望这样讲你能明白。"),
    ("彩虹是怎么形成的?",
     "阳光被水滴折射、色散形成。",
     "彩虹的形成原理其实非常优美，让我慢慢道来，当阳光穿过空气中的小水滴时，会发生折射和色散，不同颜色的光分开，最终就形成了我们看到的彩虹，大自然真是奇妙啊。"),
    ("为什么铁会生锈?",
     "铁与氧气和水发生氧化反应。",
     "铁生锈这件事呢，背后其实是个化学过程，详细说来，是因为铁在有水和氧气的环境里会发生氧化反应，生成氧化铁也就是铁锈，所以保持干燥能防锈哦。"),
]
dataset = Dataset.from_list([
    {"prompt":   [{"role": "user", "content": q}],
     "chosen":   [{"role": "assistant", "content": good}],
     "rejected": [{"role": "assistant", "content": bad}]}
    for q, good, bad in PAIRS
])

print(f"设备 {DEVICE} | 加载 {MODEL_ID}")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

cfg = DPOConfig(
    output_dir=OUT,
    per_device_train_batch_size=2,
    num_train_epochs=2,                 # DPO 极易过优化翻车, 轮数要少
    learning_rate=1e-5,                 # lr 要小; 大了模型概率被压垮 -> 输出退化成乱码
    beta=0.3,                           # beta 越大越贴紧参考模型(KL约束强), 越不易跑飞
    logging_steps=2,
    save_strategy="no",
    report_to=[],
    max_length=512,
    max_prompt_length=128,
    bf16=False, fp16=False,
    dataloader_pin_memory=False,
)


# peft_config 给了 DPOTrainer, 参考模型自动 = 关掉适配器的基座, 无需另存一份
trainer = DPOTrainer(model=model, args=cfg, train_dataset=dataset, processing_class=tok, peft_config=lora)
print("\n===== 开始 DPO 训练 =====")
trainer.train()

# TRL 自己就算好了 DPO 的核心指标, 直接从训练日志里取首/末两个看变化:
#   rewards/margins  = chosen 的隐式奖励 − rejected 的, 越大 = 越偏好 chosen
#   rewards/accuracies = 判对"chosen 比 rejected 好"的比例, ->1.0 表示完全学会偏好
logs = [h for h in trainer.state.log_history if "rewards/margins" in h]
print("\n===== DPO 学到了什么 (取自训练日志) =====")
if logs:
    a, b = logs[0], logs[-1]
    print(f"  偏好间隔 rewards/margins : {a['rewards/margins']:+.2f}  ->  {b['rewards/margins']:+.2f}  (越大越好)")
    print(f"  偏好判对率 accuracies    : {a['rewards/accuracies']:.0%}  ->  {b['rewards/accuracies']:.0%}")
print("\n要点: 没用 reward model、没采样, 只用成对偏好 + 一个分类式损失。")
print("      margin 变大、accuracy->100% = 模型学会了'简洁优于啰嗦'这个偏好 —— 这就是 DPO。")
print("      (注意: DPO 极易过优化, lr/轮数稍大就会把模型整体概率压垮、生成退化成乱码。)")
