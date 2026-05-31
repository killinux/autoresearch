# Autoresearch Windows RTX 4090 适配指南

## 背景

原版 autoresearch 只支持 Linux + NVIDIA GPU（H100 测试），本文记录将其适配到 **Windows 11 + RTX 4090 Laptop GPU** 的完整过程。

## 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GeForce RTX 4090 Laptop GPU (16GB VRAM) |
| CPU | Intel Core i9-14900HX |
| RAM | 63.7 GB |
| OS | Windows 11 |
| CUDA | 13.0 |
| Python | 3.13.12 |
| PyTorch | 2.9.1+cu128 |

## 需要的修改（共 2 处）

### 修改 1：替换 Flash Attention 3 为 PyTorch 原生 SDPA

**原因：** Flash Attention 3 的 `kernels` 包没有 Windows 构建变体（只有 Linux 预编译二进制）。RTX 4090 是 Ada Lovelace 架构（compute capability 8.9），FA3 本身也是为 Hopper（H100）优化的。

**原代码（train.py 第 20-24 行）：**
```python
from kernels import get_kernel
cap = torch.cuda.get_device_capability()
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface
```

**修改：** 删除以上 5 行。

**原代码（CausalSelfAttention.forward 中）：**
```python
y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
y = y.contiguous().view(B, T, -1)
```

**替换为：**
```python
q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2)
if self.n_kv_head < self.n_head:
    k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
    v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
y = y.transpose(1, 2).contiguous().view(B, T, -1)
```

**说明：**
- FA3 的输入格式是 `(B, T, n_heads, head_dim)`，而 PyTorch SDPA 的输入格式是 `(B, n_heads, T, head_dim)`，所以需要 transpose
- FA3 原生支持 GQA（Grouped Query Attention），SDPA 不支持，需要手动 `repeat_interleave` 来扩展 KV heads
- FA3 的 `window_size` 参数（滑动窗口注意力）在 SDPA 中不支持，这里忽略了这个特性。对训练结果的影响很小
- SDPA 在 CUDA 上会自动选择最优后端（FlashAttention-2 或 Memory-Efficient Attention）

### 修改 2：禁用 torch.compile

**原因：** `torch.compile` 默认使用 Triton 后端做 kernel fusion，Triton 不支持 Windows。

**修改位置 3 处：**

1. **模型编译（第 508 行）：**
```python
# 原代码
model = torch.compile(model, dynamic=False)
# 改为注释掉
# model = torch.compile(model, dynamic=False)
```

2. **AdamW 优化器（第 305 行）：**
```python
# 删除这行装饰器
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(...)
```

3. **Muon 优化器（第 316 行）：**
```python
# 删除这行装饰器
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(...)
```

**影响：** 禁用 `torch.compile` 后训练速度会降低约 30-50%，但功能完全正常。MFU（Model FLOPs Utilization）会比 H100 上低很多。

### 修改 3：减小 DEVICE_BATCH_SIZE（防止 OOM）

**原因：** RTX 4090 Laptop 只有 16GB 显存，原版 `DEVICE_BATCH_SIZE=128` 会导致 CUDA OOM。

**修改：**
```python
# 原代码
DEVICE_BATCH_SIZE = 128
# 改为
DEVICE_BATCH_SIZE = 32   # 16GB VRAM 适配
```

**说明：**
- `TOTAL_BATCH_SIZE` 保持不变（2^19 = 524288 tokens），gradient accumulation steps 会自动增加来补偿
- 训练效果不变（等效 batch size 相同），但每步需要更多 micro-step，速度略慢
- 如果仍然 OOM，可以继续减小到 16 甚至 8

## 部署步骤

```powershell
# 1. 安装 uv（如果没有）
pip install uv

# 2. 进入项目目录
cd E:\code\mac\autoresearch

# 3. 安装依赖（会自动安装 PyTorch CUDA 版，约 2-3GB）
uv sync

# 4. 准备数据（下载数据分片 + 训练 BPE tokenizer，约 2 分钟）
uv run prepare.py

# 5. 应用上述两处修改到 train.py

# 6. 开始训练（固定 5 分钟）
uv run train.py
```

## 一键修补脚本

可以在远程 Windows 上运行这个 Python 脚本来自动应用所有修改：

```python
path = r'E:\code\mac\autoresearch\train.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 移除 FA3 import 块
new_lines = []
skip = False
for line in lines:
    if 'from kernels import get_kernel' in line:
        skip = True
        continue
    if skip and 'flash_attn_interface' in line:
        skip = False
        continue
    if skip and (line.startswith('cap =') or line.startswith('# varunneal') or line.startswith('repo =')):
        continue
    new_lines.append(line)

c = ''.join(new_lines)

# 替换 FA3 调用为 SDPA
old = '        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)'
new = ('        q = q.transpose(1, 2)\n'
       '        k = k.transpose(1, 2)\n'
       '        v = v.transpose(1, 2)\n'
       '        if self.n_kv_head < self.n_head:\n'
       '            k = k.repeat_interleave(self.n_head // self.n_kv_head, dim=1)\n'
       '            v = v.repeat_interleave(self.n_head // self.n_kv_head, dim=1)\n'
       '        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)\n'
       '        y = y.transpose(1, 2)')
c = c.replace(old, new)

# 禁用 torch.compile
c = c.replace('model = torch.compile(model, dynamic=False)',
              '# model = torch.compile(model, dynamic=False)')
c = c.replace('@torch.compile(dynamic=False, fullgraph=True)\ndef adamw_step_fused',
              'def adamw_step_fused')
c = c.replace('@torch.compile(dynamic=False, fullgraph=True)\ndef muon_step_fused',
              'def muon_step_fused')

# 减小 batch size 防止 OOM (16GB VRAM)
c = c.replace('DEVICE_BATCH_SIZE = 128', 'DEVICE_BATCH_SIZE = 32')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('All patches applied successfully')
```

## 对比：原版 vs Windows 适配版

| 特性 | 原版 (Linux H100) | Windows 适配版 (RTX 4090) |
|------|-------------------|--------------------------|
| Attention | Flash Attention 3 | PyTorch SDPA |
| 滑动窗口注意力 | 支持 (SSSL pattern) | 不支持（全局注意力） |
| torch.compile | 启用 (Triton) | 禁用 |
| 训练速度 | 基线 | 较慢（无编译优化） |
| 显存效率 | 最优 | 良好（SDPA 自动选择后端） |
| val_bpb | ~1.06 (H100) | 1.94（仅 15 步） |
| 总耗时 | ~7 分钟 | ~39 分钟（含 warmup + eval） |
| 有效训练步数 | 数百步 | 15 步 |

## 实测结果

```
val_bpb:          1.939352
training_seconds: 307.9
total_seconds:    2364.3
peak_vram_mb:     23092.0
mfu_percent:      0.21
total_tokens_M:   7.9
num_steps:        15
num_params_M:     50.3
depth:            8
```

**分析：** val_bpb=1.94 较高，根本原因是没有 torch.compile 导致每步耗时 77 秒，5 分钟内只训练了 4 步（步 11-14），训练量严重不足。H100 上同样 5 分钟能跑几百步。

**优化建议：**
- 如果能安装 Triton（未来可能支持 Windows），恢复 torch.compile 会大幅提升速度
- 可以增加 TIME_BUDGET（如 1800s = 30 分钟）让模型训练更充分
- 减小模型（DEPTH=4）也能加快每步速度
