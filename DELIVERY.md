# 成员A交付文档
## 交付清单
### 1. 论文复现报告
- 文件：`reproduction_report.md`
- 内容：原始论文理解、复现过程、关键发现

### 2. 实验配置文件（8组）
- 位置：`configs/`
- 文件：
  - exp01_only_q.yaml ~ exp08_all.yaml
- 说明：每组配置定义了不同的LoRA插入层策略

### 3. 带层选择开关的训练框架
- 核心文件：
  - `train_lora_switchable.py` — 主训练脚本
  - `config_system.py` — 配置管理系统
  - `inference_with_lora.py` — 推理脚本

### 4. 实验调度脚本
- `run_all_experiments.sh` — 顺序运行所有实验
- `batch_inference.sh` — 批量推理所有模型
- `track_experiments.py` — 实验状态跟踪

### 5. 环境配置文档
- `setup.md` — 环境搭建步骤
- `requirements.txt` — 依赖清单

## 给成员B（模型实现与训练）的交接说明

### 你需要做：
1. 按照`setup.md`搭建环境
2. 确认成员C的数据集已准备好
3. 运行 `bash run_all_experiments.sh` 开始训练
4. 使用 `python track_experiments.py` 监控进度
5. 训练完成后，将`outputs/`目录交给成员D

## 给成员D（评估与可视化）的交接说明

### 你需要做：
1. 使用 `batch_inference.sh` 对所有模型进行推理
2. 使用成员C提供的测试集和标签
3. 计算CSR/CRR/FID等指标
4. 绘制层敏感度热力图

### 输入文件：
- 模型权重：`outputs/expXX/checkpoint-final/lora_weights`
- 测试Prompts：由成员C提供
- 配置文件：`configs/`（包含每层的选择信息，用于热力图）
