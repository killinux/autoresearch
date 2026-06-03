# Agentic RL / 后训练实战教程（agentic-rl/）

严格按 **《Agentic RL / 后训练 12 周学习计划》** 写的一套**可跑实战脚本**：
用计划指定的 **TRL + Qwen2.5-0.5B-Instruct**，把后训练主线
**SFT → DPO → GRPO(RLVR) → 工具调用 RL** 一阶段一个脚本走一遍。
全部 **本地 Mac(MPS) + LoRA** 就能跑，秒级到分钟级，每个脚本都打印"训练前→训练后"的变化。

> 这是**独立目录**，自成一套，不依赖仓库里其它内容。
> 玩具规模、纯手写、理解算法的版本另见 `../example/`；本目录是**真模型 + 工业库 TRL** 的实战版。

## 阶段 ↔ 脚本

| 计划阶段 | 脚本 | 用到的 TRL | 这一步教什么 / 跑出来看什么 |
|---|---|---|---|
| 阶段1 **SFT** | [`stage1_sft.py`](stage1_sft.py) | `SFTTrainer` | 指令微调、**loss masking**（prompt 不算 loss）。学一个"简洁中文+喵~"风格 |
| 阶段2 **DPO** | [`stage2_dpo.py`](stage2_dpo.py) | `DPOTrainer` | 偏好对齐，无需 reward model。学"简洁优于啰嗦"，看 `margins↑ / accuracy→100%` |
| 阶段3 **GRPO/RLVR** ⭐ | [`stage3_grpo.py`](stage3_grpo.py) | `GRPOTrainer` | **重中之重**：可验证奖励（答案对不对+格式）。算术正确率↑ |
| 阶段4 **工具调用 RL** | [`stage4_agentic.py`](stage4_agentic.py) | `GRPOTrainer` | Reward System **真的执行工具**(`<calc>`)。学会"不会算就调工具" |

> **阶段0 地基**（nanoGPT、RL 概念、读 InstructGPT）是阅读/概念，无脚本。

## 怎么跑

```bash
cd agentic-rl
python3 stage1_sft.py        # ~10 秒：风格 SFT
python3 stage2_dpo.py        # ~10 秒：偏好对齐，看 margin/accuracy
python3 stage3_grpo.py       # ~1-3 分钟：RLVR，算术正确率↑
python3 stage4_agentic.py    # ~1-3 分钟：工具调用 RL

# GRPO/Agentic 想快速冒烟看一眼链路：
MAX_STEPS=6 NUM_GEN=4 MAX_COMP=64 python3 stage3_grpo.py
```

依赖：`torch transformers peft datasets trl`（已装；trl 0.24）。模型 `Qwen2.5-0.5B-Instruct`
本机 HF 缓存已有，无需下载。国内重拉：`export HF_ENDPOINT=https://hf-mirror.com`。

## 实测结果（本地 MPS）

```
阶段1 SFT   : "月亮绕着地球转。"  ->  "月亮绕着地球转，喵~"   (测试题在训练集外，证明学到风格)
阶段2 DPO   : rewards/margins +0.10 -> +1.14 ; accuracy 50% -> 100%
阶段3 GRPO  : 算术正确率 67% -> 100% ; 训练奖励 0.88 -> 1.04
阶段4 工具RL: 正确率 75% -> 100% ; 模型学会写"干净可执行"的 <calc>92 * 25</calc>
```

## 踩过的坑（都写进脚本注释了）

- **SFT/DPO 在小数据上极易过拟合/翻车**：lr 或轮数稍大，输出就退化成乱码（`月亮、` / `T/T/T`）。
  解法：lr 调小、轮数减少、DPO 调高 `beta`（更贴紧参考模型）。
- **DPO 的效果别用"自由生成"评测**（容易被生成退化干扰）；直接看 TRL 日志里的
  `rewards/margins` 和 `rewards/accuracies`，那才是 DPO 真正在优化的量。
- **MPS**：用 fp32、开 `PYTORCH_ENABLE_MPS_FALLBACK=1`、`use_vllm=False`、关 `gradient_checkpointing`。
- **可验证奖励天然防作弊**：阶段4 让模型算两位数乘法，它没法靠"瞎写答案数字"蒙混，
  只能写真实算式交给工具——这正是计划阶段4 强调的"代码题不能 print 答案绕过单测"。

## 计划里还没覆盖的（需要 GPU / 更大投入）

本教程把计划阶段 1-4 的**核心机制**都用真模型跑通了。计划里更重的部分留作下一步：
- **多轮 Agentic**：本目录阶段4 是**单轮**工具调用（生成一次→执行→打分）。真正的多轮
  （模型看到工具返回再继续）需要自定义 rollout 循环，工业上用 **verl**（HybridFlow，
  reward manager / sandbox / multi-turn rollout）—— 计划点名、最对口工业岗。
- **复现项目**：`nano-aha-moment`（单文件 GRPO 看 aha moment）、`TinyZero`（~$30 复现 R1-Zero）、
  在 **verl** 上配 GRPO 数学任务 / 接代码沙箱单测奖励——都需要 GPU。
- **阶段0/2 的论文**：DeepSeekMath(GRPO)、DeepSeek-R1、InstructGPT、DPO 原论文。

## 12 周计划对照

```
W1   地基复习                  (阶段0, 读)
W2   SFT          -> stage1_sft.py
W3   DPO+RLHF概念  -> stage2_dpo.py
W4-5 GRPO/R1 ⭐    -> stage3_grpo.py   (+读 R1, 逐行读 nano-aha-moment)
W6   真RL训练       (TinyZero, 需GPU)
W7-8 迁到 verl      (需GPU)
W9   Reward/CodeRL -> stage4_agentic.py 的奖励设计思路 (+verl 沙箱单测)
W10-11 Agentic     -> stage4 单轮版打底, 多轮上 verl
W12  整合复盘
```
