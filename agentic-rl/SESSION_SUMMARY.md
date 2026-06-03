# 强化学习 / 后训练 学习 Session 总结

> 一次从「RL 基础概念」一路打到「真模型跑通 SFT→DPO→GRPO→工具调用」的完整学习。
> 所有代码已提交到 `github.com/killinux/autoresearch`（master 单分支）。

## 一、一句话总结

用「玩具代码理解算法 → 真模型(Qwen2.5-0.5B)本地跑通」两条腿，走通了后训练主线
**SFT → DPO → GRPO(RLVR) → Agentic/工具调用 RL**，并配齐了可视化 H5 图解。

## 二、知识脉络（学到了什么）

| 主题 | 核心要点 |
|---|---|
| **Post-training 全景** | 预训练 Base → ①SFT(会听话) → ②偏好对齐(对口味) → ③RLVR/GRPO(会推理) → ④Agentic(会做事) |
| **SFT** | 监督学习；用"指令→回答"对 + **loss masking**(只在回答上算损失)；不是 RL |
| **DPO** | 用成对偏好(chosen/rejected)对齐；**无需奖励模型、无需采样**；一个分类式损失；极易过优化翻车 |
| **GRPO / RLVR** | 同题采 N 个 → **组内归一化算优势** → 强化好的；**不用 critic 网络**；奖励=可验证规则；DeepSeek-R1 同款 |
| **Agentic RL** | 多步行动 + 调用工具 + **稀疏奖励** + 信用分配；"任务式后训练" |
| **Reward System / hacking** | 奖励=真的跑工具/单测；**弱奖励→模型学会作弊**，可验证奖励天然防作弊 |
| **训练 vs 推理** | 训练=探索+算梯度+更新权重；推理=加载权重+贪心+no_grad；桥梁=checkpoint |
| **Critic 网络** | Actor-Critic 里的"评委"，给局面估分当 baseline 降方差；**GRPO 用组均值替代它** |
| **三条规划路线** | RL 训进权重 / MCTS 推理时搜树 / 世界模型(JEPA)脑内模拟——区别在"算力花在训练还是推理" |
| **MCTS planning agent** | 经典 AlphaZero/MuZero；LLM 时代 ToT / RAP / LATS / rStar |
| **LeCun 世界模型** | JEPA / V-JEPA 2，用世界模型做规划(MPC)，他押注的 AGI 路线 |
| **TRL** | HuggingFace 后训练全家桶，封装 SFTTrainer/DPOTrainer/GRPOTrainer |

## 三、产出清单（代码 + 文档）

### 1. `example/` —— 玩具规模，纯 numpy/torch，秒级，理解算法

| 文件 | 内容 |
|---|---|
| `05_grpo_minimal.py` | GRPO 最小骨架(~40行) |
| `06_agentic_rl_tool.py` | 多步 + 调工具 + 稀疏奖励(黑暗走廊先LOOK后走) |
| `07_agentic_codeRL.py` | 沙箱跑单测当奖励 + **reward hacking 实证** |
| `08_train_vs_infer.py` | 训练 vs 推理界线 + checkpoint |
| `agentic_rl_roadmap.html` | 后训练全景图 + Infra四大件 + 三条规划路线对比 |

### 2. `qwen_demo/` —— 真模型，本地 Mac(MPS)

| 文件 | 内容 |
|---|---|
| `01_grpo_format.py` | 手写 GRPO 教 Qwen2.5-0.5B 遵循 `<think><answer>`，奖励 **0→1.0** |

### 3. `agentic-rl/` —— 按 12 周计划，TRL + Qwen2.5-0.5B + LoRA，本地全跑通

| 阶段 | 脚本 | H5图解 | 实测结果 |
|---|---|---|---|
| ①SFT | `stage1_sft.py` | `stage1_sft.html` | "月亮绕着地球转。"→"…，**喵~**" |
| ②DPO | `stage2_dpo.py` | `stage2_dpo.html` | margins **0.10→1.14**、accuracy **50%→100%** |
| ③GRPO | `stage3_grpo.py` | `stage3_grpo.html` | 算术正确率 **67%→100%** |
| ④工具RL | `stage4_agentic.py` | `stage4_agentic.html` | 正确率 **75%→100%**，学会写干净可执行 `<calc>` |

## 四、关键结论速记

- **算法都一样，区别在落地**：手写版(example/qwen_demo)看清原理，TRL 版(agentic-rl)是工业做法。
- **GRPO 精髓一行**：`advantage = (reward − 组均值) / 组标准差`，用同组样本互相当 baseline，省掉 critic。
- **奖励设计 > 训练算法**：弱奖励让 agent 作弊；可验证、难作弊的奖励才是 Agentic RL 胜负手。
- **后训练四步本质同构**：都是「试错 → 打分 → 强化更好的」。

## 五、踩过的坑（很值钱）

| 坑 | 解法 |
|---|---|
| SFT 小数据**过拟合崩**(输出退化"月亮、") | 降 lr、减轮数、多给几条同风格数据 |
| DPO **过优化翻车**(退化"T/T/T") | lr 调小、轮数减少、**beta 调大**(更贴参考模型) |
| DPO 别用"自由生成"评测 | 直接看 TRL 日志的 `rewards/margins` + `accuracies` |
| Mac/MPS | fp32 + `PYTORCH_ENABLE_MPS_FALLBACK=1` + `use_vllm=False` |

## 六、下一步 TODO

- [ ] 多轮 Agentic（本目录是单轮，真多轮需 **verl**）
- [ ] 复现项目：nano-aha-moment / TinyZero（需 GPU）
- [ ] 把 GRPO 从"学格式"升级到"学解题"(GSM8K)
- [ ] 读论文：DeepSeekMath(GRPO)、DeepSeek-R1、InstructGPT、DPO

---

> 环境：macOS + MPS；torch 2.8 / transformers 4.57 / peft 0.17 / **trl 0.24**；
> Qwen2.5-0.5B-Instruct 已本地缓存于 `~/.cache/huggingface/hub/`。
