# LoRA 实验配置文件说明

> **项目**: Modular LoRA 复现 - Safety Alignment Backfires
> **论文**: [Safety Alignment Backfires: Preventing the Re-emergence of Suppressed Concepts in Fine-tuned Text-to-Image Diffusion Models](https://arxiv.org/abs/2412.00357)
> **作者**: Sanghyun Kim, Moonseok Choi, Jinwoo Shin, Juho Lee (KAIST)
> **配置版本**: v1.0
> **更新日期**: 2026-06-06

---

## 目录

1. [配置概述](#1-配置概述)
2. [文件结构](#2-文件结构)
3. [8组实验配置详解](#3-8组实验配置详解)
4. [参数说明](#4-参数说明)
5. [使用指南](#5-使用指南)
6. [论文推荐配置](#6-论文推荐配置)
7. [实验对比建议](#7-实验对比建议)
8. [常见问题](#8-常见问题)

---

## 1. 配置概述

本目录包含 **8组 LoRA 实验配置文件**，用于系统性地研究不同层插入 LoRA 对模型安全性和下游任务性能的影响。

### 配置设计目标

| 目标 | 说明 |
|------|------|
| **消融实验** | 对比不同 Attention 组件 (Q/K/V) 的独立作用 |
| **完整性验证** | 验证完整 Attention 层 vs 部分层的差异 |
| **错误示范** | 展示不适当层 (ResNet/Skip) 的负面效果 |
| **极端对比** | 全层 LoRA 作为参数量上限参考 |

---

## 2. 文件结构

```
configs/
├── README.md                    # 本文件
├── exp01_only_q.yaml           # 仅 Query 投影
├── exp02_only_k.yaml           # 仅 Key 投影
├── exp03_only_v.yaml           # 仅 Value 投影
├── exp04_qkv.yaml              # Q + K + V (论文推荐)
├── exp05_resnet.yaml           # ResNet 卷积层 (不推荐)
├── exp06_skip.yaml             # 跳跃连接 (不推荐)
├── exp07_full_attn.yaml        # 完整 Attention (Q/K/V/Out)
└── exp08_all.yaml              # 所有层 (不推荐)
```

---

## 3. 8组实验配置详解

### 3.1 单组件消融实验 (exp01-exp03)

用于研究 Cross-Attention 中 Q/K/V 各自的独立作用。

#### exp01_only_q.yaml - 仅 Query 投影

```yaml
experiment_name: only_q
target_modules:
  - to_q
```

| 属性 | 值 |
|------|-----|
| **目标层** | `attn2.to_q` (Cross-Attention Query) |
| **参数量** | ~0.8M |
| **论文评级** | 可选 |
| **适用场景** | 研究 Query 在概念对齐中的作用 |
| **预期效果** | 有限的风格控制，概念对齐较弱 |

**原理说明**: Query 投影负责将图像特征映射到查询空间，单独修改仅影响查询什么概念，但无法完整控制概念激活。

---

#### exp02_only_k.yaml - 仅 Key 投影

```yaml
experiment_name: only_k
target_modules:
  - to_k
```

| 属性 | 值 |
|------|-----|
| **目标层** | `attn2.to_k` (Cross-Attention Key) |
| **参数量** | ~0.8M |
| **论文评级** | 可选 |
| **适用场景** | 研究 Key 在概念检索中的作用 |
| **预期效果** | 影响概念检索，但生成质量受限 |

**原理说明**: Key 投影负责将文本 token 映射到键空间，单独修改影响如何匹配概念，但缺少 Value 传递的具体信息。

---

#### exp03_only_v.yaml - 仅 Value 投影

```yaml
experiment_name: only_v
target_modules:
  - to_v
```

| 属性 | 值 |
|------|-----|
| **目标层** | `attn2.to_v` (Cross-Attention Value) |
| **参数量** | ~0.8M |
| **论文评级** | 可选 |
| **适用场景** | 研究 Value 在概念传递中的作用 |
| **预期效果** | 影响概念内容，但检索准确性受限 |

**原理说明**: Value 投影负责将文本 token 的具体信息传递到图像特征，单独修改影响传递什么概念内容，但检索可能不准确。

---

### 3.2 核心推荐配置 (exp04)

#### exp04_qkv.yaml - Q + K + V (论文标准配置)

```yaml
experiment_name: qkv
target_modules:
  - to_q
  - to_k
  - to_v
```

| 属性 | 值 |
|------|-----|
| **目标层** | `attn2.to_q`, `attn2.to_k`, `attn2.to_v` |
| **参数量** | ~2.4M (0.7% of total) |
| **论文评级** | **强烈推荐** |
| **适用场景** | **生产环境、论文复现、标准实验** |
| **预期效果** | 最佳性价比，安全对齐与下游任务兼顾 |

**原理说明**:
- **Query**: 控制查询什么概念
- **Key**: 控制如何匹配概念
- **Value**: 控制传递什么内容
- 三者协同，完整控制文本到图像的概念映射

**论文依据**:
- 论文第4.2节: we only apply LoRA to Wq and Wv in most experiments
- 论文表6: W_q, W_v 组合在 GPT-3 上效果最佳
- 论文第7.1节: adapting both Wq and Wv yields the best result

---

### 3.3 错误示范配置 (exp05-exp06)

用于展示不适当层插入 LoRA 的负面效果，作为教学对比。

#### exp05_resnet.yaml - ResNet 卷积层

```yaml
experiment_name: resnet
target_modules:
  - conv1
  - conv2
  - conv_shortcut
  - time_emb_proj
```

| 属性 | 值 |
|------|-----|
| **目标层** | ResNet 卷积、时间嵌入投影 |
| **参数量** | ~15M+ |
| **论文评级** | **不推荐** |
| **适用场景** | **仅作为错误示范/对比实验** |
| **预期效果** | 破坏预训练知识，概念控制失效 |

**为什么不推荐**:
1. **与文本无关**: ResNet 卷积处理纯视觉特征，不接收文本条件
2. **破坏预训练**: 修改卷积层会破坏已学习的视觉表示
3. **概念控制失效**: 无法精确控制生成什么概念
4. **参数量浪费**: 大量参数用于无关的视觉特征调整

---

#### exp06_skip.yaml - 跳跃连接

```yaml
experiment_name: skip
target_modules:
  - conv_shortcut
```

| 属性 | 值 |
|------|-----|
| **目标层** | `conv_shortcut` (跳跃连接) |
| **参数量** | ~5M |
| **论文评级** | **不推荐** |
| **适用场景** | **仅作为错误示范** |
| **预期效果** | 严重破坏残差连接，训练不稳定 |

**为什么不推荐**:
1. **残差连接破坏**: 跳跃连接是 ResNet 的核心，修改后梯度传播受阻
2. **训练不稳定**: 容易导致梯度消失/爆炸
3. **无概念控制**: 与文本条件完全无关

---

### 3.4 扩展配置 (exp07)

#### exp07_full_attn.yaml - 完整 Attention

```yaml
experiment_name: full_attn
target_modules:
  - to_q
  - to_k
  - to_v
  - to_out.0
```

| 属性 | 值 |
|------|-----|
| **目标层** | `attn2.to_q/k/v` + `attn2.to_out.0` |
| **参数量** | ~3.0M (0.9% of total) |
| **论文评级** | **推荐** |
| **适用场景** | 追求更完整的注意力控制 |
| **预期效果** | 比 exp04 更完整的映射，性价比略降 |

**与 exp04 的区别**:
- 增加 `to_out.0`: 将注意力结果映射回特征空间的输出投影
- 提供更完整的注意力->特征转换控制
- 参数量增加 25%，效果提升有限

**适用场景**: 当 exp04 无法满足复杂风格迁移需求时尝试。

---

### 3.5 极端配置 (exp08)

#### exp08_all.yaml - 所有层

```yaml
experiment_name: all_layers
target_modules:
  - to_q
  - to_k
  - to_v
  - to_out.0
  - conv1
  - conv2
  - conv_shortcut
  - time_emb_proj
  - proj_in
  - proj_out
```

| 属性 | 值 |
|------|-----|
| **目标层** | 所有 Linear + Conv 层 |
| **参数量** | ~50M+ (15% of total) |
| **论文评级** | **不推荐** |
| **适用场景** | **仅作为参数量上限参考** |
| **预期效果** | 过拟合风险，性价比极低 |

**为什么不推荐**:
1. **性价比极低**: 15% 参数量，效果提升 < 5%
2. **过拟合风险**: 大量参数容易过拟合小数据集
3. **失去 LoRA 意义**: 接近全量微调，违背参数高效初衷
4. **推理成本**: 即使合并后，原始模型已被大幅修改

---

## 4. 参数说明

### 4.1 通用参数（8组配置相同）

| 参数 | 值 | 说明 | 论文依据 |
|------|-----|------|---------|
| `rank` | 8 | LoRA 低秩维度 | 论文使用 r=4，此处扩展为 8 |
| `lora_alpha` | 16 | 缩放因子 | 通常设为 alpha = 2*rank |
| `learning_rate` | 1e-4 | 学习率 | 论文 ESD: 1e-4, SDD: 1e-5 |
| `num_train_steps` | 1000 | 训练步数 | 论文安全训练: 1500，下游: 5000 |
| `save_steps` | 200 | 保存间隔 | 每 200 步保存 checkpoint |
| `output_dir` | `outputs/expXX_xxx` | 输出目录 | 按实验 ID 组织 |

### 4.2 参数调优建议

#### Rank 选择

| 场景 | 推荐 rank | 说明 |
|------|----------|------|
| 快速实验/资源受限 | 1-2 | 验证可行性 |
| 标准下游任务 | 4-8 | 平衡效果与效率 |
| 复杂风格迁移 | 16-32 | diminishing returns |
| 全量微调对比 | 64+ | 接近全秩，失去 LoRA 优势 |

**论文洞察**: 论文表6显示，GPT-3 上 r=1 已足够，增大到 r=64 几乎无提升。

---

## 5. 使用指南

### 5.1 加载配置

```python
import yaml

# 加载单个配置
with open('configs/exp04_qkv.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(config['target_modules'])  # ['to_q', 'to_k', 'to_v']
print(config['rank'])            # 8
```

### 5.2 应用到 LoRA 训练

```python
from peft import LoraConfig, get_peft_model

# 从 YAML 读取配置
with open('configs/exp04_qkv.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# 创建 LoRA 配置
lora_config = LoraConfig(
    r=cfg['rank'],
    lora_alpha=cfg['lora_alpha'],
    target_modules=cfg['target_modules'],
    lora_dropout=0.0,
    bias='none',
)

# 注入模型
unet = get_peft_model(base_unet, lora_config)
```

### 5.3 批量运行所有实验

```python
#!/usr/bin/env python3
# Batch run 8 experiments

import os
import yaml
import glob

# Get all configs
config_files = sorted(glob.glob('configs/exp*.yaml'))

for config_path in config_files:
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    exp_name = cfg['experiment_name']
    print(f'\n{"="*60}')
    print(f'Running experiment: {exp_name}')
    print(f'Config: {config_path}')
    print(f'{"="*60}')
    
    # Run training
    # train_lora(cfg)  # Your training function
    
    print(f'Completed: {exp_name}')
```

---

## 6. 论文推荐配置

### 6.1 生产环境首选

```yaml
# 基于论文实验验证的最优配置
experiment_name: production
target_modules:
  - to_q
  - to_k
  - to_v
rank: 4          # 论文使用 r=4
lora_alpha: 4    # alpha = rank
learning_rate: 1e-4
num_train_steps: 1500   # 安全对齐
# num_train_steps: 5000  # 下游微调
save_steps: 100
output_dir: outputs/production
```

### 6.2 与论文配置的对比

| 参数 | 本配置 (exp04) | 论文配置 | 差异说明 |
|------|--------------|---------|---------|
| rank | 8 | 4 | 本配置更保守，参数量更大 |
| alpha | 16 | 4 | 本配置 scaling=2，更稳定 |
| target_modules | to_q, to_k, to_v | to_q, to_v | 本配置包含 to_k，更完整 |
| steps | 1000 | 1500 | 本配置简化演示 |

**建议**: 严格复现论文时，使用 `rank=4, alpha=4, target_modules=[to_q, to_v]`。

---

## 7. 实验对比建议

### 7.1 核心对比实验

| 对比组 | 实验 | 目的 |
|--------|------|------|
| **A** | exp01 vs exp02 vs exp03 | Q/K/V 各自的作用 |
| **B** | exp04 vs exp01-03 | 组合 vs 单独的差异 |
| **C** | exp04 vs exp07 | 是否加入 to_out.0 |
| **D** | exp04 vs exp05-06 | 正确 vs 错误层的对比 |
| **E** | exp04 vs exp08 | 最优 vs 极端的性价比 |

### 7.2 评估指标

| 指标 | 说明 | 工具 |
|------|------|------|
| **NSFW Rate** | 有害图像比例 | NudeNet / Q16 |
| **CLIP Score** | 文本-图像对齐 | CLIP |
| **FID** | 图像质量 | pytorch-fid |
| **LPIPS** | 感知相似度 | lpips |
| **Trainable Params** | 可训练参数量 | peft |

### 7.3 预期结果趋势

```
NSFW Rate (lower better):
  exp04_qkv < exp07_full_attn < exp01/02/03 < exp08_all < exp05/06
  
CLIP Score (higher better):
  exp04_qkv ~ exp07_full_attn > exp01/02/03 > exp08_all > exp05/06
  
Trainable Params (lower better):
  exp01/02/03 < exp04_qkv < exp07_full_attn < exp05 < exp06 < exp08_all
```

---

## 8. 常见问题

### Q1: 为什么 exp05-resnet 和 exp06-skip 标记为不推荐还要创建？

**A**: 作为**消融实验的对照组**和**教学演示**。通过对比可以：
- 验证 Cross-Attention 层是概念控制的关键
- 展示错误层选择导致的负面效果
- 帮助理解 LoRA 层选择的重要性

### Q2: rank=8 是否过大？论文使用 rank=4。

**A**: rank=8 提供更多容量，适合：
- 复杂风格迁移任务
- 大数据集微调
- 作为上限参考

严格复现论文时，建议创建 `exp04_qkv_rank4.yaml`：
```yaml
experiment_name: qkv_rank4
target_modules: [to_q, to_k, to_v]
rank: 4
lora_alpha: 4
```

### Q3: 如何添加新的实验配置？

**A**: 复制现有配置并修改：
```bash
# 创建新配置
cp configs/exp04_qkv.yaml configs/exp09_my_experiment.yaml

# 编辑修改
# 修改 experiment_name, target_modules, rank 等
```

### Q4: 配置文件可以在其他模型上使用吗？

**A**: 可以，但需注意：
- **Stable Diffusion v1.4/v1.5**: 完全兼容
- **SDXL**: 需要调整 `target_modules` 名称（如 `to_q` -> `to_add_q`）
- **FLUX.1**: 架构不同，需要重新设计配置

### Q5: 如何验证配置是否正确加载？

**A**: 使用以下代码验证：
```python
from peft import LoraConfig
import yaml

with open('configs/exp04_qkv.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# 验证配置
config = LoraConfig(**cfg)
print(f'Target modules: {config.target_modules}')
print(f'Rank: {config.r}')
print(f'Alpha: {config.lora_alpha}')
```

---

## 附录: 快速参考表

| 配置 | 文件 | 推荐度 | 参数量 | 核心用途 |
|------|------|--------|--------|---------|
| 仅 Q | exp01 | 可选 | ~0.8M | 消融实验 |
| 仅 K | exp02 | 可选 | ~0.8M | 消融实验 |
| 仅 V | exp03 | 可选 | ~0.8M | 消融实验 |
| **Q+K+V** | **exp04** | **强烈推荐** | **~2.4M** | **生产/论文复现** |
| ResNet | exp05 | 不推荐 | ~15M | 错误示范 |
| Skip | exp06 | 不推荐 | ~5M | 错误示范 |
| Full Attn | exp07 | 推荐 | ~3.0M | 扩展实验 |
| All Layers | exp08 | 不推荐 | ~50M | 极端参考 |

---

> **最后更新**: 2026-06-06

---

*This configuration set is designed based on the paper [Safety Alignment Backfires] for systematically studying the impact of LoRA layer selection on diffusion model safety alignment.*
