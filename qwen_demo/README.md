# Qwen 真模型 RL Demo（qwen_demo/）

基于 **Qwen2.5-0.5B-Instruct** 的、**本地 Mac(MPS) 就能跑** 的强化学习 / 后训练小例子。
后续所有"基于 Qwen 真模型"的例子都放这个目录。

和隔壁 [`../example/`](../example/) 的关系：`example/` 是**玩具规模**的 RL 八连例（纯 numpy/torch，
秒级，理解算法）；`qwen_demo/` 把同一套算法**搬到真 LLM 上**（要加载 0.5B 模型，分钟级）。
建议先把 `example/05_grpo_minimal.py`（GRPO 骨架）看懂，再来看这里——你会发现**算法一模一样**，
新增的只是"怎么用 HuggingFace 模型生成 + 算 logprob"。

## 例子列表

| # | 文件 | 在做什么 | 用到的 example 前置 |
|---|------|---------|--------------------|
| 01 | [`01_grpo_format.py`](01_grpo_format.py) | 手写 GRPO 教 Qwen 遵循 `<think></think><answer></answer>` 格式，纯规则奖励 | example 05（GRPO）+ 07（可验证奖励）|

## 怎么跑

```bash
cd qwen_demo
python3 01_grpo_format.py                       # 默认 G=6 STEPS=40 MAX_NEW=96，约几分钟
# 想先快速冒烟看一眼链路：
G=2 STEPS=2 MAX_NEW=24 python3 01_grpo_format.py
# 想要更漂亮的训练曲线：
G=8 STEPS=120 MAX_NEW=128 python3 01_grpo_format.py
```

## 跑出来大概长这样（实测，本地 MPS）

```
===== 训练前 (base 模型) =====
平均格式奖励: 0.00
样例输出: To solve the problem of adding 17 and 25 ...（一大段，没有标签）

===== 开始 GRPO 训练 =====
  step  0 | 平均奖励 0.20
  step  5 | 平均奖励 0.40
  step 10 | 平均奖励 0.60
  step 15 | 平均奖励 0.75

===== 训练后 =====
平均格式奖励: 1.00  (训练前 0.00)
样例输出: <think> The sum of 17 and 25 is 42. </think> <answer>42</answer>
```

**没有一个标注答案**，只用"格式对不对"的规则奖励 + GRPO，就把模型输出"掰"成了目标格式——
这就是 RLVR（可验证奖励强化学习，DeepSeek-R1 那条线）的最小真身。

## 环境 & 取舍（为本地 Mac 跑通做的）

- **依赖**：`torch transformers peft`（已具备）。**不需要 trl**——GRPO 是手写的。
- **模型**：`Qwen2.5-0.5B-Instruct`，本机 HF 缓存已有，无需下载。
  国内重新拉取时：`export HF_ENDPOINT=https://hf-mirror.com`。
- **LoRA**：只训小适配器、冻结基座 → 省显存、快、不会把模型训崩。
- **MPS**：fp32 + `PYTORCH_ENABLE_MPS_FALLBACK=1`（个别算子回退 CPU），脚本里已自动设。
- **最小化**：核心只留 GRPO 的心脏（组相对优势 + 策略梯度）。完整的 **PPO-clip 多轮内循环**
  和 **对参考模型的 KL 锚点** 见 `../example/05_grpo_minimal.py`；脚本底部也写了怎么加回来。

## 下一步可以玩什么

1. **从"学格式"升级到"学解题"**：把 `format_reward` 换成"`<answer>` 里必须是正确算术答案"
   （把 `example/07` 的答案校验搬过来）——就是 GSM8K 同款配方。
2. **加 KL 锚点防跑偏**：用 `model.disable_adapter()` 前向得参考 logprob，loss 里加 `beta*(logp-ref)`。
3. **迁到 GPU 机器**：代码自动选 `cuda`，把 `G/STEPS/MAX_NEW` 调大，曲线更稳。
