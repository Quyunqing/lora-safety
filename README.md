# 🔒 文生图扩散模型安全对齐项目

> **大学生创新创业项目** | 基于LoRA层选择的轻量化安全对齐研究

## 📋 项目背景

当前的文生图扩散模型（如Stable Diffusion、DALL-E）在生成丰富图像的同时，也存在显著的安全与伦理风险：

- **生成有害内容**：暴力、色情、歧视性图像
- **安全对齐失效**：微调过程中可能破坏原有安全机制，导致有害概念"复活"
- **重训练成本高昂**：传统安全对齐需要大规模重训练，计算资源消耗巨大

本项目旨在应用**模型编辑技术**，对文生图扩散模型进行**精准、高效的安全对齐**。通过定位模型中与特定有害概念相关的内部知识，并对其进行微幅修改或删除，实现在**不大规模重训练、不显著损害模型通用生成能力**的前提下，使模型能够主动拒绝生成不安全图像，从而以**"外科手术式"的轻量化方案**提升文生图模型的安全性与可靠性。

---

## 🎯 研究目标

1. **复现基线方法**：复现论文《Safety Alignment Backfires: Preventing the Re-emergence of Suppressed Concepts in Fine-tuned Text-to-Image Diffusion Models》
2. **研究LoRA层选择**：探索LoRA插入不同层对安全对齐效果的影响
3. **研究LoRA Rank**：对比不同低秩维度(rank)对安全性和通用性的权衡
4. **跨模型对比**：在Stable Diffusion、SDXL等不同模型上验证方法有效性

---

## 🏗️ 项目结构

```
lora-safety-project/
├── 📁 configs/                  # 8组实验配置文件 (YAML)
│   ├── exp01_only_q.yaml        # 仅Q投影层
│   ├── exp02_only_k.yaml        # 仅K投影层
│   ├── exp03_only_v.yaml        # 仅V投影层
│   ├── exp04_qkv.yaml           # QKV三层
│   ├── exp05_resnet.yaml        # ResNet块
│   ├── exp06_skip.yaml          # Skip连接
│   ├── exp07_full_attn.yaml     # 完整Attention
│   └── exp08_all.yaml           # 全部层
│
├── 📁 src/                      # 核心源代码
│   ├── config_system.py         # 配置管理系统
│   ├── train_lora_switchable.py # 带层选择开关的训练框架
│   ├── inference_with_lora.py   # 推理与模型加载工具
│   └── track_experiments.py     # 实验状态跟踪工具
│
├── 📁 scripts/                  # 运行脚本
│   ├── generate_configs.py      # 生成8组配置文件
│   ├── run_all_experiments.sh   # 一键运行所有实验
│   └── batch_inference.sh       # 批量推理脚本
│
├── 📁 docs/                     # 文档
│   ├── setup.md                 # 环境搭建指南 (本文档)
│   ├── README.md                # 项目说明 (本文档)
│   └── CONFIG_README.md         # 配置文件说明
│
├── 📁 outputs/                  # 训练输出 (运行时生成)
│   └── exp01_only_q/
│       ├── checkpoint-200/      # 中间checkpoint
│       ├── checkpoint-400/
│       ├── checkpoint-final/    # 最终模型
│       │   ├── lora_weights/    # LoRA权重文件
│       │   ├── config.yaml      # 配置副本
│       │   └── training_state.pt
│       └── training_report.md   # 训练报告
│
├── 📁 logs/                     # 日志文件 (运行时生成)
│   ├── scheduler/               # 调度日志
│   └── exp01_only_q_train.log   # 各实验训练日志
│
├── 📁 inference_results/        # 推理结果 (运行时生成)
│
├── requirements.txt             # Python依赖清单
└── DELIVERY.md                  # 交付文档
```

---

## 👥 团队分工

| 成员 | 职责 | 核心任务 |
|------|------|----------|
| **A** | 实验设计与框架搭建 | 搭建LoRA训练框架、定义实验配置、管理代码仓库 |
| **B** | 模型实现与训练 | 实现各层LoRA注入逻辑、执行全部训练任务、监控显存与训练稳定性 |
| **C** | 数据与概念管理 | 构建概念数据集(抑制概念+正常概念)、设计正负样本对、数据增强 |
| **D** | 评估与可视化 | 计算CSR/CRR/FID指标、绘制层敏感度热力图、撰写实验分析 |

---

## 🚀 快速开始

### 1. 环境搭建 (5分钟)

```bash
# 创建conda环境
conda create -n lora-safety python=3.10 -y
conda activate lora-safety

# 安装PyTorch (根据CUDA版本选择)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# 安装项目依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

> 详细环境配置说明见 [docs/setup.md](docs/setup.md)

### 2. 登录必要平台

```bash
# HuggingFace (下载模型需要)
huggingface-cli login
# 输入你的Token: https://huggingface.co/settings/tokens

# Weights & Biases (可选，用于训练监控)
wandb login
# 输入你的API Key
```

### 3. 生成实验配置

```bash
python scripts/generate_configs.py
# 输出: configs/exp01_only_q.yaml ~ configs/exp08_all.yaml
```

### 4. 运行训练 (成员B执行)

```bash
# 运行单组实验
python src/train_lora_switchable.py --config configs/exp01_only_q.yaml

# 或一键运行所有实验
bash scripts/run_all_experiments.sh
```

### 5. 推理与评估 (成员D执行)

```bash
# 批量推理所有训练好的模型
bash scripts/batch_inference.sh

# 或推理单个模型
python src/inference_with_lora.py \
    --lora_weights outputs/exp01_only_q/checkpoint-final/lora_weights \
    --prompts_file test_prompts.txt \
    --output_dir inference_results/exp01_only_q
```

---

## 📊 8组实验设计

| 实验ID | 名称 | 目标层 | 研究目的 |
|--------|------|--------|----------|
| exp01 | 仅Q投影 | `to_q` | 研究Query层对概念理解的影响 |
| exp02 | 仅K投影 | `to_k` | 研究Key层对概念匹配的影响 |
| exp03 | 仅V投影 | `to_v` | 研究Value层对概念表达的影响 |
| exp04 | QKV三层 | `to_q, to_k, to_v` | 研究Attention核心层组合效果 |
| exp05 | ResNet块 | `conv1, conv2, conv_shortcut, time_emb_proj` | 研究空间特征层的影响 |
| exp06 | Skip连接 | `conv_shortcut` | 研究残差连接的作用 |
| exp07 | 完整Attention | `to_q, to_k, to_v, to_out.0` | 研究完整Attention机制 |
| exp08 | 全部层 | 所有可训练层 | 研究全层微调的效果与代价 |

---

## 🔬 核心创新点

### 1. 层选择开关机制

传统LoRA训练硬编码目标层，本项目实现**可配置层选择**：

```python
# 传统写法 (硬编码)
lora_config = LoraConfig(
    r=8,
    target_modules=["to_q", "to_v"],  # 写死了！
)

# 本项目写法 (从配置读取)
lora_config = LoraConfig(
    r=config.rank,
    target_modules=config.target_modules,  # 从YAML配置读取！
    lora_alpha=config.lora_alpha,
)
```

### 2. 模块化LoRA设计

参考论文《Safety Alignment Backfires》的**Modular LoRA**思想：
- **Safety LoRA**: 专门训练用于抑制有害概念
- **Fine-tuning LoRA**: 用于正常任务微调
- **推理时合并**: 两个LoRA模块独立训练，推理时动态合并

### 3. 系统化评估框架

- **CSR (Concept Suppression Rate)**: 概念抑制成功率
- **CRR (Concept Re-emergence Rate)**: 概念复活率
- **FID (Fréchet Inception Distance)**: 图像质量评估
- **层敏感度热力图**: 可视化不同层对安全对齐的贡献

---

## 📚 关键论文

| 论文 | 作用 | 链接 |
|------|------|------|
| LoRA: Low-Rank Adaptation of Large Language Models | 理解LoRA原理 | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| Safety Alignment Backfires | 复现基线方法 | [arXiv:2412.00357](https://arxiv.org/abs/2412.00357) |

---

## 📦 交付物清单

### 成员A交付 (实验设计与框架)

| 交付物 | 文件 | 说明 |
|--------|------|------|
| 论文复现报告 | `reproduction_report.md` | 原始论文理解、关键发现 |
| 8组实验配置 | `configs/exp01~exp08.yaml` | 不同层选择策略 |
| 训练框架 | `src/train_lora_switchable.py` | 带层选择开关的核心代码 |
| 配置系统 | `src/config_system.py` | YAML配置读写管理 |
| 调度脚本 | `scripts/run_all_experiments.sh` | 一键运行所有实验 |
| 推理工具 | `src/inference_with_lora.py` | 模型加载与图像生成 |
| 跟踪工具 | `src/track_experiments.py` | 实验进度监控 |
| 环境文档 | `docs/setup.md` | 环境搭建步骤 |
| 依赖清单 | `requirements.txt` | Python包版本 |

### 成员B交付 (模型实现与训练)

| 交付物 | 说明 |
|--------|------|
| 8组训练完成的LoRA模型 | `outputs/exp*/checkpoint-final/` |
| 训练日志 | 含显存/时长/损失曲线 |
| 实验状态报告 | 成功/失败标注 |

### 成员C交付 (数据与概念)

| 交付物 | 说明 |
|--------|------|
| 统一数据包 | 训练集+测试集 |
| 数据增强脚本 | 图像预处理代码 |
| 数据校验报告 | 数据质量验证 |

### 成员D交付 (评估与可视化)

| 交付物 | 说明 |
|--------|------|
| 8组模型评估报告 | CSR/CRR/FID指标 |
| 层敏感度热力图 | 矢量图格式 |
| 对比图表 | 实验结果可视化 |
| 实验分析初稿 | 结论与讨论 |

---

## ⚠️ 注意事项

1. **显存管理**: 16GB显存可运行rank=8，24GB可运行rank=16，如需更大rank请使用梯度检查点
2. **数据同步**: 训练前务必确认成员C的数据集已准备就绪
3. **版本锁定**: 所有依赖版本已在`requirements.txt`中锁定，请勿随意升级
4. **Git管理**: 建议用Git管理代码，定期commit
5. **WandB监控**: 训练过程自动记录到WandB，可实时监控损失曲线

---

## 📞 常见问题

**Q: 找不到模型文件？**  
A: 首次运行会自动从HuggingFace下载，需确保网络通畅或提前用`huggingface-cli download`下载

**Q: 显存不足(OOM)？**  
A: 尝试：①减小batch_size到1 ②降低resolution到512 ③开启gradient_checkpointing ④减小rank

**Q: 如何恢复中断的实验？**  
A: 修改`scripts/run_all_experiments.sh`中的`START_FROM`变量，指向中断的实验索引

---

## 📄 License

本项目为大学生创新创业研究项目，仅供学术研究使用。

---

> **项目周期**: 2026年5月 - 2026年6月  
> **目标日期**: 2026年6月10日前完成全部实验与评估
